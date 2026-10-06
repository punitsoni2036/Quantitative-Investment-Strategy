# region imports
from AlgorithmImports import *
# endregion
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# UNIVERSE SELECTION (Week 4 Day 5 spec)
# ─────────────────────────────────────────────

def fundamental_filter_function(fundamental):
    filtered = [
        f for f in fundamental
        if f.has_fundamental_data
        and f.price > 10
        and f.market_cap > 2_000_000_000
    ]
    sorted_by_dollar_volume = sorted(filtered, key=lambda f: f.dollar_volume, reverse=True)
    return [f.symbol for f in sorted_by_dollar_volume[:50]]

def get_universe_snapshot(qb, date):
    universe = qb.add_universe(fundamental_filter_function)

    start_date = date - pd.Timedelta(days=7)

    history = list(
        qb.universe_history(
            universe,
            start_date,
            date
        )
    )

    if not history:
        return []

    return history[-1]


# ─────────────────────────────────────────────
# WINSORISE / Z-SCORE HELPERS (Week 4 Day 2 spec)
# ─────────────────────────────────────────────

def winsorise(series, lower=0.05, upper=0.95):
    lo, hi = series.quantile(lower), series.quantile(upper)
    return series.clip(lo, hi)

def zscore(series):
    return (series - series.mean()) / series.std(ddof=1)


# ─────────────────────────────────────────────
# VALUE FACTOR (Week 3 Day 1 spec)
# ─────────────────────────────────────────────

def compute_value(snapshot):
    records = []
    for f in snapshot:
        pe = f.valuation_ratios.pe_ratio
        pb = f.valuation_ratios.pb_ratio
        records.append({
            'symbol': str(f.symbol),
            'earnings_yield': 1 / pe if pe and pe > 0 else np.nan,
            'book_yield': 1 / pb if pb and pb > 0 else np.nan,
        })
    df = pd.DataFrame(records)
    df['raw_value'] = df[['earnings_yield', 'book_yield']].mean(axis=1)
    return df[['symbol', 'raw_value']]


# ─────────────────────────────────────────────
# QUALITY FACTOR (Week 3 Day 2 spec)
# ─────────────────────────────────────────────

def compute_quality(snapshot):
    records = []
    for f in snapshot:
        roe = f.operation_ratios.roe.value if f.operation_ratios.roe else np.nan
        de = f.operation_ratios.total_debt_equity_ratio.value if f.operation_ratios.total_debt_equity_ratio else np.nan
        margin = f.operation_ratios.net_margin.value if f.operation_ratios.net_margin else np.nan
        records.append({
            'symbol': str(f.symbol),
            'roe': roe,
            'inv_debt_to_equity': 1 / de if de and de > 0 else np.nan,
            'net_margin': margin,
        })
    df = pd.DataFrame(records)
    df['raw_quality'] = df[['roe', 'inv_debt_to_equity', 'net_margin']].mean(axis=1)
    return df[['symbol', 'raw_quality']]


# ─────────────────────────────────────────────
# GROWTH FACTOR (Week 3 Day 4 spec)
# ─────────────────────────────────────────────

def compute_growth(snapshot):
    records = []
    for f in snapshot:
        rev_g = f.operation_ratios.revenue_growth.value if f.operation_ratios.revenue_growth else np.nan
        earn_g = f.operation_ratios.net_income_growth.value if f.operation_ratios.net_income_growth else np.nan
        records.append({'symbol': str(f.symbol), 'revenue_growth': rev_g, 'earnings_growth': earn_g})
    df = pd.DataFrame(records)
    df['earnings_growth_capped'] = winsorise(df['earnings_growth'], 0.02, 0.98)
    df['raw_growth'] = df[['revenue_growth', 'earnings_growth_capped']].mean(axis=1)
    return df[['symbol', 'raw_growth']]


# ─────────────────────────────────────────────
# MOMENTUM FACTOR (Week 3 Day 3 spec) — 12-1 month
# ─────────────────────────────────────────────

def compute_momentum(qb, symbols, date, lookback=252, skip=21, buffer_days=390):
    history = qb.history(symbols, buffer_days, Resolution.DAILY)
    records = []
    for sym in symbols:
        try:
            prices = history.loc[sym]['close']
            if len(prices) < lookback:
                records.append({'symbol': str(sym), 'momentum_12_1': np.nan})
                continue
            price_start = prices.iloc[-lookback]
            price_end = prices.iloc[-skip]
            records.append({'symbol': str(sym), 'momentum_12_1': (price_end / price_start) - 1})
        except KeyError:
            records.append({'symbol': str(sym), 'momentum_12_1': np.nan})
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# COMBINE INTO v1.0 SCORE (Week 3 Day 5 + Week 4 Day 2 spec)
# Equal-weighted z-scores, winsorised raw factors first
# ─────────────────────────────────────────────

def compute_combined_score(qb, symbols, date):
    snapshot = get_universe_snapshot(qb, date)
    snapshot = [f for f in snapshot if str(f.symbol) in [str(s) for s in symbols]]

    df_value = compute_value(snapshot)
    df_quality = compute_quality(snapshot)
    df_growth = compute_growth(snapshot)
    df_momentum = compute_momentum(qb, symbols, date)

    merged = df_value.merge(df_quality, on='symbol') \
                      .merge(df_growth, on='symbol') \
                      .merge(df_momentum, on='symbol')

    for raw_col, z_col in [('raw_value', 'z_value'), ('raw_quality', 'z_quality'),
                            ('raw_growth', 'z_growth'), ('momentum_12_1', 'z_momentum')]:
        winsorised = winsorise(merged[raw_col])
        merged[z_col] = zscore(winsorised)

    merged['combined_score'] = merged[['z_value', 'z_quality', 'z_growth', 'z_momentum']].mean(axis=1)
    return merged


# ─────────────────────────────────────────────
# PORTFOLIO CONSTRUCTION (Week 4 Day 5 spec)
# Top 15, equal-weighted
# ─────────────────────────────────────────────

def construct_portfolio(scored_df, n_holdings=15):
    top_n = scored_df.nlargest(n_holdings, 'combined_score').copy()
    top_n['target_weight'] = 1.0 / n_holdings
    return top_n[['symbol', 'combined_score', 'target_weight']]


# ─────────────────────────────────────────────
# ONE-CALL ENTRY POINT
# ─────────────────────────────────────────────

def get_frozen_v1_portfolio(qb, date, n_universe=50, n_holdings=15):
    
    universe = qb.add_universe(fundamental_filter_function)
    start_date = date - pd.Timedelta(days=7)

    snapshot = list(
        qb.universe_history(
            universe,
            start_date,
            date
        )
    )

    if not snapshot:
        print("No universe history found")
        return None

    snapshot = snapshot[0]

    symbols = [f.symbol for f in snapshot]
        
    

    scored = compute_combined_score(qb, symbols, date)
    portfolio = construct_portfolio(scored, n_holdings=n_holdings)
    return portfolio


# Usage:
# qb = QuantBook()
# portfolio = get_frozen_v1_portfolio(qb, pd.Timestamp("2023-12-01"))
# print(portfolio)



from AlgorithmImports import *
import numpy as np
import pandas as pd

class MultiFactorStrategy(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2018, 1, 1)
        self.set_end_date(2020, 12, 31)   
        self.set_cash(10000)
        self.set_benchmark("SPY")

        self.universe_settings.resolution = Resolution.DAILY
        self.add_universe(self.fundamental_filter_function)


        self.schedule.on(
            self.date_rules.month_start(),
            self.time_rules.after_market_open("SPY", 10),
            self.rebalance
        )

    def fundamental_filter_function(self, fundamental):
        filtered = [
            f for f in fundamental
            if f.has_fundamental_data
            and f.price > 10
            and f.market_cap > 2_000_000_000
        ]
        sorted_by_dollar_volume = sorted(filtered, key=lambda f: f.dollar_volume, reverse=True)
        return [f.symbol for f in sorted_by_dollar_volume[:50]]

    def on_data(self, data):
        pass  


#rebalancing method

    def rebalance(self):

        symbols = [
            security.symbol
            for security in self.active_securities.values()
        ]

        self.debug(f"Rebalancing with {len(symbols)} symbols")

        if len(symbols) < 20:
            self.debug("Not enough symbols to rebalance")
            return

        scored_df = self.compute_combined_score(symbols)

        if scored_df is None:
            self.debug("compute_combined_score returned None")
            return

        self.debug(f"Scored {len(scored_df)} stocks")

        if len(scored_df) < 20:
            self.debug("Fewer than 20 stocks received scores")
            return

        top_20 = scored_df.nlargest(
            20,
            "combined_score"
        )

        target_weight = 1.0 / 20

        # Liquidate old positions
        held_symbols = [
            s for s in self.portfolio.keys()
            if self.portfolio[s].invested
        ]

        new_symbols = set(
            top_20["symbol"].tolist()
        )

        for s in held_symbols:

            if str(s) not in [
                str(x) for x in new_symbols
            ]:
                self.liquidate(s)

        # Buy new positions
        for _, row in top_20.iterrows():

            self.set_holdings(
                row["symbol"],
                target_weight
            )



    def compute_combined_score(self, symbols):

        # ---------------------------------------
        # 1. Get historical price data
        # ---------------------------------------

        history = self.history(
            symbols,
            390,
            Resolution.DAILY
        )

        if history.empty:
            return None


        # ---------------------------------------
        # 2. Calculate factor values for each stock
        # ---------------------------------------

        records = []

        for sym in symbols:

            try:
                fundamentals = self.securities[sym].fundamentals

                if not fundamentals.has_fundamental_data:
                    continue

                # -------------------------------
                # Fundamental data
                # -------------------------------

                pe = fundamentals.valuation_ratios.pe_ratio
                pb = fundamentals.valuation_ratios.pb_ratio

                roe = (
                    fundamentals.operation_ratios.roe.value
                    if fundamentals.operation_ratios.roe
                    else np.nan
                )

                margin = (
                    fundamentals.operation_ratios.net_margin.value
                    if fundamentals.operation_ratios.net_margin
                    else np.nan
                )

                rev_g = (
                    fundamentals.operation_ratios.revenue_growth.value
                    if fundamentals.operation_ratios.revenue_growth
                    else np.nan
                )

                earn_g = (
                    fundamentals.operation_ratios.net_income_growth.value
                    if fundamentals.operation_ratios.net_income_growth
                    else np.nan
                )


                # -------------------------------
                # Value factor
                # -------------------------------

                value_pe = 1 / pe if pe > 0 else np.nan
                value_pb = 1 / pb if pb > 0 else np.nan

                raw_value = np.nanmean([
                    value_pe,
                    value_pb
                ])


                # -------------------------------
                # Quality factor
                # -------------------------------

                raw_quality = np.nanmean([
                    roe,
                    margin
                ])


                # -------------------------------
                # Growth factor
                # -------------------------------

                raw_growth = np.nanmean([
                    rev_g,
                    earn_g
                ])


                # -------------------------------
                # Momentum factor
                # -------------------------------

                prices = history.loc[sym]["close"].dropna()

                if len(prices) < 252:
                    continue

                price_start = prices.iloc[-252]
                price_end = prices.iloc[-21]

                momentum = (
                    price_end / price_start
                ) - 1


                # -------------------------------
                # Store all four factors
                # -------------------------------

                records.append({
                    "symbol": str(sym),
                    "raw_value": raw_value,
                    "raw_quality": raw_quality,
                    "raw_growth": raw_growth,
                    "momentum": momentum
                })


            except Exception as e:
                self.Debug(f"Error calculating factors for {sym}: {e}")
                continue


        # ---------------------------------------
        # 3. Create DataFrame
        # ---------------------------------------

        df = pd.DataFrame(records)

        if df.empty:
            return None


        # ---------------------------------------
        # 4. Winsorise and z-score the factors
        # ---------------------------------------

        raw_cols = [
            "raw_value",
            "raw_quality",
            "momentum",
            "raw_growth"
        ]

        for col in raw_cols:

            # Winsorisation
            low = df[col].quantile(0.05)
            high = df[col].quantile(0.95)

            df[f"{col}_w"] = df[col].clip(
                low,
                high
            )

            # Z-score
            mean = df[f"{col}_w"].mean()
            std = df[f"{col}_w"].std(ddof=1)

            if std == 0 or pd.isna(std):
                df[f"z_{col}_w"] = 0.0
            else:
                df[f"z_{col}_w"] = (
                    df[f"{col}_w"] - mean
                ) / std


        # ---------------------------------------
        # 5. Calculate combined score
        # ---------------------------------------

        df["combined_score"] = (
            df["z_raw_value_w"]
            + df["z_raw_quality_w"]
            + df["z_momentum_w"]
            + df["z_raw_growth_w"]
        )
        return df
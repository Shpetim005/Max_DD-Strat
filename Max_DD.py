import streamlit as st
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import plotly.graph_objects as go

st.set_page_config(page_title="DD Cash-Wait All-in vs HOLD DCA (Stooq)", layout="wide")


# =========================================================
# Data (Stooq)
# =========================================================
def _to_stooq_symbol(ticker: str) -> str:
    return ticker.lower() + ".us"

@st.cache_data(show_spinner=False)
def download_prices_stooq(tickers, start, end):
    frames = []
    failed = []
    for t in tickers:
        try:
            sym = _to_stooq_symbol(t)
            df = pdr.DataReader(sym, "stooq", start=start, end=end).sort_index()
            s = df["Close"].rename(t)
            frames.append(s)
        except Exception:
            failed.append(t)

    out = pd.concat(frames, axis=1) if frames else pd.DataFrame()
    if not out.empty:
        out.index = pd.to_datetime(out.index)
    return out, failed

def get_sample_tickers():
    # Tu peux remplacer par tes top100
    return [
        "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA",
        "AVGO","JPM","V","XOM","MA","UNH","COST","HD",
        "PG","MRK","ABBV","KO","PEP"
    ]


# =========================================================
# Utils
# =========================================================
def compute_drawdown(price: pd.Series) -> pd.Series:
    peak = price.cummax()
    return price / peak - 1.0

def month_starts(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # premier jour de trading présent par mois
    s = pd.Series(1, index=index)
    ms = s.groupby([index.year, index.month]).apply(lambda x: x.index.min())
    return pd.to_datetime(ms.values)

def perf_stats(equity: pd.Series):
    equity = equity.dropna()
    if len(equity) < 2:
        return {"Total return": np.nan, "Vol (ann.)": np.nan, "Max DD": np.nan, "Sharpe (rf=0)": np.nan}

    rets = equity.pct_change().dropna()
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    vol = rets.std() * np.sqrt(252)
    mdd = (equity / equity.cummax() - 1.0).min()
    sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252)) if rets.std() != 0 else np.nan
    return {"Total return": total, "Vol (ann.)": vol, "Max DD": mdd, "Sharpe (rf=0)": sharpe}


# =========================================================
# Portfolio Backtests
# =========================================================
def backtest_hold_dca_portfolio(prices: pd.DataFrame, monthly_contrib: float):
    """
    HOLD (DCA) portefeuille:
    - Début de chaque mois: cash += X puis on investit immédiatement X
    - Répartition: égal-weight sur tous les assets disponibles ce jour-là
    """
    prices = prices.dropna(how="all")
    if prices.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    idx = prices.index
    ms = set(month_starts(idx))

    cash = 0.0
    shares = {c: 0.0 for c in prices.columns}

    values = []
    entries = []

    for dt in idx:
        # contrib mensuelle + investissement immédiat
        if dt in ms:
            cash += monthly_contrib

            today_prices = prices.loc[dt].dropna()
            available = list(today_prices.index)

            if cash > 0 and len(available) > 0:
                alloc = cash / len(available)
                for t in available:
                    p = float(today_prices[t])
                    shares[t] += alloc / p

                entries.append({
                    "Date": dt,
                    "Type": "HOLD_DCA",
                    "Tickers": ",".join(available),
                    "CashInvested": float(cash),
                    "N_assets": len(available),
                })
                cash = 0.0

        # portfolio value
        today_prices = prices.loc[dt]
        port_val = cash
        for t, sh in shares.items():
            p = today_prices.get(t, np.nan)
            if pd.notna(p):
                port_val += sh * float(p)

        values.append((dt, port_val))

    equity = pd.DataFrame(values, columns=["Date", "Value"]).set_index("Date")["Value"]
    entries_df = pd.DataFrame(entries).set_index("Date") if entries else pd.DataFrame(
        columns=["Type", "Tickers", "CashInvested", "N_assets"]
    )
    return equity, entries_df


def backtest_dd_cash_wait_allin_portfolio(prices: pd.DataFrame, monthly_contrib: float, threshold: float):
    """
    Stratégie DD (cash-wait, all-in) portefeuille:
    - Début de mois: cash += X
    - Chaque jour: si cash > 0 et au moins un asset a DD <= threshold (peak-to-trough),
      on investit 100% du cash disponible.
    - Si plusieurs assets trigger le même jour: on répartit le cash également entre eux.
    - Après investissement: cash = 0 -> tu ne réinvestis que le mois suivant (nouveau X).
    """
    prices = prices.dropna(how="all")
    if prices.empty:
        return pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame()

    # drawdowns par asset
    dd = prices.apply(lambda s: compute_drawdown(s.dropna()).reindex(prices.index), axis=0)

    idx = prices.index
    ms = set(month_starts(idx))

    cash = 0.0
    shares = {c: 0.0 for c in prices.columns}

    values = []
    entries = []          # log portfolio-level (date, tickers)
    entries_by_asset = [] # log asset-level (pour marqueurs sur graphe DD)

    for dt in idx:
        # épargne mensuelle
        if dt in ms:
            cash += monthly_contrib

        # triggers du jour (assets dont DD <= threshold et prix dispo)
        if cash > 0:
            today_prices = prices.loc[dt].dropna()
            today_dd = dd.loc[dt].dropna()

            candidates = list(set(today_prices.index).intersection(set(today_dd.index)))
            triggered = [t for t in candidates if float(today_dd[t]) <= threshold]

            if len(triggered) > 0:
                invest_total = cash
                alloc = invest_total / len(triggered)

                for t in triggered:
                    p = float(today_prices[t])
                    shares[t] += alloc / p
                    entries_by_asset.append({
                        "Date": dt,
                        "Ticker": t,
                        "Invest": float(alloc),
                        "Price": float(p),
                        "Drawdown": float(today_dd[t]),
                    })

                entries.append({
                    "Date": dt,
                    "Type": "DD_ALLIN",
                    "Tickers": ",".join(triggered),
                    "CashInvested": float(invest_total),
                    "N_assets": len(triggered),
                })
                cash = 0.0

        # portfolio value
        today_prices = prices.loc[dt]
        port_val = cash
        for t, sh in shares.items():
            p = today_prices.get(t, np.nan)
            if pd.notna(p):
                port_val += sh * float(p)

        values.append((dt, port_val))

    equity = pd.DataFrame(values, columns=["Date", "Value"]).set_index("Date")["Value"]
    entries_df = pd.DataFrame(entries).set_index("Date") if entries else pd.DataFrame(
        columns=["Type", "Tickers", "CashInvested", "N_assets"]
    )
    entries_asset_df = pd.DataFrame(entries_by_asset)
    if not entries_asset_df.empty:
        entries_asset_df["Date"] = pd.to_datetime(entries_asset_df["Date"])
        entries_asset_df = entries_asset_df.set_index("Date").sort_index()
    else:
        entries_asset_df = pd.DataFrame(columns=["Ticker","Invest","Price","Drawdown"])

    return equity, entries_df, entries_asset_df, dd


# =========================================================
# UI
# =========================================================
st.title("📉 DD (peak-to-trough) cash-wait all-in vs HOLD (DCA) — Portfolio multi-assets")

with st.sidebar:
    st.header("Paramètres")
    start = st.date_input("Start", value=pd.to_datetime("2015-01-01"))
    end = st.date_input("End", value=pd.to_datetime("2025-12-31"))

    monthly_contrib = st.number_input("Épargne mensuelle X ($)", min_value=0.0, value=200.0, step=50.0)

    dd_pct = st.number_input("Seuil drawdown (%)", min_value=1, max_value=90, value=20)
    threshold = -dd_pct / 100.0

    tickers = get_sample_tickers()
    run = st.button("Lancer", type="primary")

if not run:
    st.info("Régle les paramètres à gauche puis clique **Lancer**.")
    st.stop()

prices, failed = download_prices_stooq(tickers, str(start), str(end))
if prices.empty:
    st.error("Aucune donnée Stooq téléchargée (dates/tickers à vérifier).")
    st.stop()

if failed:
    st.warning(f"Tickers échoués (Stooq): {failed}")

# align: garder uniquement les colonnes avec assez de data
min_obs = 300
keep = [c for c in prices.columns if prices[c].dropna().shape[0] >= min_obs]
prices = prices[keep].copy()
if prices.shape[1] < 2:
    st.error("Pas assez d'assets avec historique suffisant. Réduis min_obs ou change la liste.")
    st.stop()

# =========================================================
# Run portfolio backtests
# =========================================================
hold_eq, hold_entries = backtest_hold_dca_portfolio(prices, monthly_contrib=monthly_contrib)
dd_eq, dd_entries, dd_entries_by_asset, dd_matrix = backtest_dd_cash_wait_allin_portfolio(
    prices, monthly_contrib=monthly_contrib, threshold=threshold
)

df_eq = pd.concat([hold_eq.rename("HOLD_DCA"), dd_eq.rename("DD_ALLIN")], axis=1).dropna()

# =========================================================
# Graph 1: Portfolio equity curves
# =========================================================
st.subheader("1) Portefeuille — courbes de valeur")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["HOLD_DCA"], name="HOLD (DCA mensuel)"))
fig.add_trace(go.Scatter(x=df_eq.index, y=df_eq["DD_ALLIN"], name=f"DD cash-wait all-in (seuil {dd_pct}%)"))
fig.update_layout(xaxis_title="Date", yaxis_title="Valeur ($)")
st.plotly_chart(fig, use_container_width=True)

# Perf summary
st.subheader("Résumé perf (portefeuille)")
summary = pd.DataFrame({
    "HOLD_DCA": perf_stats(df_eq["HOLD_DCA"]),
    "DD_ALLIN": perf_stats(df_eq["DD_ALLIN"]),
})
st.dataframe(summary, use_container_width=True)

# Entries tables
c1, c2 = st.columns(2)
with c1:
    st.write("Entrées HOLD (mensuel, réparti sur tous les assets)")
    st.dataframe(hold_entries, use_container_width=True)
with c2:
    st.write("Entrées DD all-in (investit tout sur les assets qui trigger ce jour)")
    st.dataframe(dd_entries, use_container_width=True)

# =========================================================
# Drill-down: single asset drawdown + entry markers (for that asset)
# =========================================================
st.subheader("2) Drawdown d'un asset + moments d'entrée (quand il reçoit une allocation)")

choice = st.selectbox("Asset", options=list(prices.columns), index=0)

dd_series = dd_matrix[choice].dropna()
# entries for that asset
asset_entries = dd_entries_by_asset[dd_entries_by_asset["Ticker"] == choice].copy()

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values, name="Drawdown"))

fig2.add_hline(y=threshold, line_dash="dash")

if not asset_entries.empty:
    dates = asset_entries.index.intersection(dd_series.index)
    fig2.add_trace(go.Scatter(
        x=dates,
        y=asset_entries.loc[dates, "Drawdown"],
        mode="markers",
        name="Entrées (alloc reçue)",
        marker=dict(size=10, symbol="circle"),
        text=[
            f"Invest: {asset_entries.loc[d,'Invest']:.0f}<br>"
            f"Price: {asset_entries.loc[d,'Price']:.2f}<br>"
            f"DD: {asset_entries.loc[d,'Drawdown']:.2%}"
            for d in dates
        ],
        hoverinfo="text"
    ))

fig2.update_layout(
    title=f"{choice} — Drawdown (peak-to-trough) + entrées",
    xaxis_title="Date",
    yaxis_title="Drawdown"
)
st.plotly_chart(fig2, use_container_width=True)

st.write("Détails des entrées pour cet asset (allocation reçue quand il trigger avec d'autres)")
st.dataframe(asset_entries, use_container_width=True)

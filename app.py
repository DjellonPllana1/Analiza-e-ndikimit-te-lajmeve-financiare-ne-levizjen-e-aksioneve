from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind

from model import train_and_evaluate


@dataclass(frozen=True)
class Paths:
    merged_csv: Path


def default_paths() -> Paths:
    root = Path(__file__).resolve().parent
    merged_csv = root / "DataPreparation" / "datasets" / "cleanedData" / "Combined_News_DJIA_Merged.csv"
    return Paths(merged_csv=merged_csv)


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    df["daily_return"] = df["Close"].pct_change()
    df["abs_return"] = df["daily_return"].abs()
    return df


def welch_ttest_by_label(df: pd.DataFrame) -> dict[str, float]:
    d = df.dropna(subset=["daily_return", "Label"]).copy()
    r0 = d.loc[d["Label"] == 0, "daily_return"]
    r1 = d.loc[d["Label"] == 1, "daily_return"]
    if len(r0) < 5 or len(r1) < 5:
        return {
            "n0": float(len(r0)),
            "n1": float(len(r1)),
            "mean0": float(r0.mean()),
            "mean1": float(r1.mean()),
            "pvalue": float("nan"),
        }

    stat = ttest_ind(r1, r0, equal_var=False, nan_policy="omit")
    return {
        "n0": float(len(r0)),
        "n1": float(len(r1)),
        "mean0": float(r0.mean()),
        "mean1": float(r1.mean()),
        "pvalue": float(stat.pvalue),
    }


def fig_close_timeseries(df: pd.DataFrame, ma_days: int) -> go.Figure:
    d = df[["Date", "Close"]].dropna().copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], mode="lines", name="Close", line=dict(width=1.5)))
    if ma_days > 1:
        d["ma"] = d["Close"].rolling(ma_days).mean()
        fig.add_trace(go.Scatter(x=d["Date"], y=d["ma"], mode="lines", name=f"MA({ma_days})", line=dict(width=2.5)))
    fig.update_layout(
        title="DJIA Close (kohë)",
        xaxis_title="Date",
        yaxis_title="Close",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def fig_returns_hist(df: pd.DataFrame, bins: int) -> go.Figure:
    d = df.dropna(subset=["daily_return"]).copy()
    d["Label"] = d["Label"].astype("Int64").astype(str)
    fig = px.histogram(
        d,
        x="daily_return",
        color="Label",
        nbins=bins,
        histnorm="probability density",
        barmode="overlay",
        opacity=0.55,
        title="Shpërndarja e kthimeve ditore sipas Label",
    )
    fig.update_layout(
        xaxis_title="daily_return (Close % change)",
        yaxis_title="Density",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend_title_text="Label",
    )
    return fig


def fig_volatility(df: pd.DataFrame, window: int) -> go.Figure:
    d = df[["Date", "daily_return"]].dropna().copy()
    d["vol"] = d["daily_return"].rolling(window).std()
    d = d.dropna(subset=["vol"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["vol"], mode="lines", name=f"STD({window})", line=dict(width=2)))
    fig.update_layout(
        title=f"Volatilitet (STD e kthimeve), dritare = {window} ditë",
        xaxis_title="Date",
        yaxis_title="Rolling std(daily_return)",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def _safe_float(x: float | int | None, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return float(default)
        return x
    except Exception:
        return float(default)


def _annualization_factor(trading_days_per_year: int) -> float:
    return float(np.sqrt(trading_days_per_year))


def compute_backtest(
    df: pd.DataFrame,
    proba_up: np.ndarray,
    threshold: float,
    allow_short: bool,
    fee_bps: float,
    start_equity: float,
    trading_days_per_year: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Very simple daily backtest on the provided rows (assumes one decision per row/day).

    Position rule:
      - long  (+1) if proba_up >= threshold
      - short (-1) if allow_short and proba_up <= (1 - threshold)
      - cash   (0) otherwise

    Returns are based on `daily_return` in the same row (educational / simplified).
    """
    d = df[["Date", "Close", "Label", "daily_return"]].copy()
    d = d.dropna(subset=["daily_return"]).reset_index(drop=True)

    p = np.asarray(proba_up, dtype=float)
    if len(p) != len(df):
        raise ValueError("proba_up length must match df length.")
    p = p[: len(d)]  # align after dropna

    pos = np.zeros(len(d), dtype=float)
    pos[p >= threshold] = 1.0
    if allow_short:
        pos[p <= (1.0 - threshold)] = -1.0

    trade = np.zeros(len(d), dtype=float)
    trade[0] = abs(pos[0])
    trade[1:] = np.abs(np.diff(pos))

    fee = (fee_bps / 10_000.0) * trade
    strat_ret = pos * d["daily_return"].to_numpy(dtype=float) - fee

    equity = np.empty(len(d), dtype=float)
    equity[0] = float(start_equity) * (1.0 + strat_ret[0])
    for i in range(1, len(d)):
        equity[i] = equity[i - 1] * (1.0 + strat_ret[i])

    d["proba_up"] = p
    d["position"] = pos
    d["trade"] = trade
    d["fee"] = fee
    d["strategy_return"] = strat_ret
    d["equity"] = equity

    # Metrics
    total_return = equity[-1] / float(start_equity) - 1.0
    mean = float(np.mean(strat_ret)) if len(strat_ret) else 0.0
    std = float(np.std(strat_ret, ddof=1)) if len(strat_ret) > 1 else 0.0
    sharpe = 0.0 if std <= 0 else (mean / std) * _annualization_factor(trading_days_per_year)

    rolling_max = np.maximum.accumulate(equity)
    drawdown = equity / rolling_max - 1.0
    max_dd = float(np.min(drawdown)) if len(drawdown) else 0.0

    win_rate = float(np.mean(strat_ret > 0)) if len(strat_ret) else 0.0
    trades = float(np.sum(trade > 0))

    stats = {
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "trades": float(trades),
    }
    return d, stats


def fig_equity_curve(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["Date"], y=bt["equity"], mode="lines", name="Equity", line=dict(width=2)))
    fig.update_layout(
        title="Equity curve (strategy)",
        xaxis_title="Date",
        yaxis_title="Equity",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def fig_positions(bt: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["Date"], y=bt["position"], mode="lines", name="Position", line=dict(width=1.5)))
    fig.update_layout(
        title="Position over time (-1 short, 0 cash, +1 long)",
        xaxis_title="Date",
        yaxis_title="Position",
        height=280,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def page_header() -> None:
    st.set_page_config(
        page_title="Trading Assistant (News x DJIA)",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Trading Assistant: News → Signal → Backtest (DJIA)")
    st.caption(
        "Demo edukative (jo këshillë financiare): sinjale nga lajmet + analiza + backtest i thjeshtë mbi DJIA."
    )


def main() -> int:
    page_header()

    p = default_paths()
    if not p.merged_csv.exists():
        st.error(
            "Nuk u gjet `Combined_News_DJIA_Merged.csv`.\n\n"
            "Ekzekuto fillimisht:\n"
            "`python DataPreparation/dataPreparation.py`"
        )
        st.stop()

    with st.sidebar:
        st.subheader("Kontrolle")
        st.write("Ndrysho filtrat dhe grafiqet rifreskohen menjëherë.")

        df = load_data(str(p.merged_csv))

        min_d = df["Date"].min().date()
        max_d = df["Date"].max().date()
        date_range = st.date_input("Range i datave", value=(min_d, max_d), min_value=min_d, max_value=max_d)

        label_filter = st.multiselect("Label", options=[0, 1], default=[0, 1])

        st.divider()
        ma_days = st.slider("Moving average (ditë)", 1, 60, 14)
        bins = st.slider("Bins (histogram)", 20, 120, 60)
        vol_window = st.slider("Vol window (ditë)", 5, 90, 20)
        st.divider()
        st.subheader("Strategy / Backtest")
        split_year = st.number_input("Split year (train < year, test ≥ year)", min_value=2000, max_value=2020, value=2015, step=1)
        threshold = st.slider("Signal threshold P(up)", 0.50, 0.90, 0.60, step=0.01)
        allow_short = st.toggle("Allow short", value=False, help="Nëse ON, strategjia merr edhe short kur P(up) është shumë e ulët.")
        fee_bps = st.slider("Transaction cost (bps per trade)", 0.0, 50.0, 5.0, step=0.5)
        start_equity = st.number_input("Start equity", min_value=1000.0, max_value=10_000_000.0, value=10_000.0, step=1000.0)
        trading_days = st.slider("Trading days/year (for Sharpe)", 200, 260, 252, step=1)

    # Normalize date input output types (can be single date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_d, end_d = date_range
    else:
        start_d = min_d
        end_d = max_d

    dff = df[(df["Date"].dt.date >= start_d) & (df["Date"].dt.date <= end_d)].copy()
    if label_filter:
        dff = dff[dff["Label"].isin(label_filter)]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Rreshta", f"{len(dff):,}".replace(",", " "))
    kpi2.metric("Data (nga)", str(start_d))
    kpi3.metric("Data (deri)", str(end_d))
    kpi4.metric("Label", ", ".join(map(str, label_filter)) if label_filter else "Asnjë")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Market", "Stats", "Data", "Signal & Backtest", "About"])

    with tab1:
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.subheader("Close (kohë)")
            fig1 = fig_close_timeseries(dff, ma_days=ma_days)
            st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": True})
        with c2:
            st.subheader("Daily return (sipas Label)")
            fig2 = fig_returns_hist(dff, bins=bins)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": True})

        st.subheader("Volatilitet")
        fig3 = fig_volatility(dff, window=vol_window)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": True})

    with tab2:
        st.subheader("Welch t-test: Label=1 vs Label=0 (daily_return)")
        s = welch_ttest_by_label(dff)

        a, b, c = st.columns(3)
        a.metric("mean(Label=0)", f"{s['mean0']:.6f}")
        b.metric("mean(Label=1)", f"{s['mean1']:.6f}")
        c.metric("p-value", f"{s['pvalue']:.6g}" if pd.notna(s["pvalue"]) else "NaN")

        st.caption(
            "Shënim: ky është test bazik mbi kthimet ditore; rezultatet nuk nënkuptojnë shkakësi dhe varen nga periudha/filtrat."
        )

    with tab3:
        st.subheader("Dataset (i filtruar)")
        show_cols = st.multiselect(
            "Kolonat",
            options=list(dff.columns),
            default=[c for c in ["Date", "Label", "Open", "High", "Low", "Close", "Volume", "daily_return"] if c in dff.columns],
        )
        st.dataframe(dff[show_cols] if show_cols else dff, use_container_width=True, height=420)

        csv_bytes = dff.to_csv(index=False).encode("utf-8")
        st.download_button("Shkarko CSV (filtered)", data=csv_bytes, file_name="filtered_data.csv", mime="text/csv")

    with tab4:
        st.subheader("Signal (baseline) → Strategy backtest")
        st.write(
            "Këtu e shohim modelin si **signal** për tregtim: trajnojmë TF‑IDF + Logistic Regression dhe e përdorim "
            "probabilitetin (P(up)) për të marrë pozicion (long/cash/short)."
        )

        cA, cB, cC = st.columns(3)
        with cA:
            max_features = st.slider("max_features (TF-IDF)", 5000, 120000, 50000, step=5000)
        with cB:
            ngram_max = st.selectbox("ngram max", options=[1, 2, 3], index=1)
        with cC:
            C = st.slider("C (LogReg)", 0.25, 10.0, 2.0, step=0.25)

        with st.spinner("Duke trajnuar modelin dhe duke bërë backtest..."):
            res = train_and_evaluate(
                df,
                split_year=int(split_year),
                max_features=int(max_features),
                ngram_max=int(ngram_max),
                C=float(C),
            )

        # Model quality KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Train rows", f"{res['n_train']:,}".replace(",", " "))
        k2.metric("Test rows", f"{res['n_test']:,}".replace(",", " "))
        k3.metric("Accuracy", f"{res['accuracy']:.3f}")
        k4.metric("F1", f"{res['f1']:.3f}")
        k5.metric("ROC-AUC", f"{res['roc_auc']:.3f}" if pd.notna(res["roc_auc"]) else "NaN")

        cm = res["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
        fig_cm = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": True})

        test_df = res["test_df"].copy()
        test_proba = res["test_proba"]
        if test_proba is None:
            st.warning("Modeli nuk ktheu probabilitete; backtest kërkon `predict_proba`.")
            st.stop()

        bt, stats = compute_backtest(
            test_df,
            proba_up=np.asarray(test_proba, dtype=float),
            threshold=float(threshold),
            allow_short=bool(allow_short),
            fee_bps=float(fee_bps),
            start_equity=float(start_equity),
            trading_days_per_year=int(trading_days),
        )

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total return", f"{stats['total_return']*100:.2f}%")
        s2.metric("Sharpe (simple)", f"{stats['sharpe']:.2f}")
        s3.metric("Max drawdown", f"{stats['max_drawdown']*100:.2f}%")
        s4.metric("Win rate", f"{stats['win_rate']*100:.1f}%")
        s5.metric("Trades", f"{int(stats['trades']):,}".replace(",", " "))

        st.plotly_chart(fig_equity_curve(bt), use_container_width=True, config={"displayModeBar": True})
        st.plotly_chart(fig_positions(bt), use_container_width=True, config={"displayModeBar": False})

        st.subheader("Paper trading (simple)")
        st.caption("Ruhet vetëm në këtë sesion të browser-it (Streamlit session_state).")
        if "paper" not in st.session_state:
            st.session_state.paper = {"cash": float(start_equity), "shares": 0.0, "last_price": None, "log": []}

        last_row = bt.dropna(subset=["Close"]).tail(1)
        if not last_row.empty:
            last_price = float(last_row["Close"].iloc[0])
            last_date = last_row["Date"].iloc[0]
            st.write(f"Latest available close: **{last_price:,.2f}** on **{last_date.date()}**".replace(",", " "))

            pstate = st.session_state.paper
            cash = float(pstate["cash"])
            shares = float(pstate["shares"])
            equity_now = cash + shares * last_price

            p1, p2, p3 = st.columns(3)
            p1.metric("Cash", f"{cash:,.2f}".replace(",", " "))
            p2.metric("Shares", f"{shares:,.4f}".replace(",", " "))
            p3.metric("Equity", f"{equity_now:,.2f}".replace(",", " "))

            c_buy, c_sell, c_flat = st.columns(3)
            with c_buy:
                qty = st.number_input("Qty (shares)", min_value=0.0, value=1.0, step=1.0)
                if st.button("Buy", use_container_width=True):
                    cost = qty * last_price
                    if cost <= cash and qty > 0:
                        pstate["cash"] = cash - cost
                        pstate["shares"] = shares + qty
                        pstate["log"].append({"Date": str(last_date.date()), "Action": "BUY", "Qty": qty, "Price": last_price})
                        st.rerun()
                    else:
                        st.error("Not enough cash (or qty=0).")
            with c_sell:
                qty_s = st.number_input("Qty to sell", min_value=0.0, value=1.0, step=1.0, key="qty_sell")
                if st.button("Sell", use_container_width=True):
                    if qty_s <= shares and qty_s > 0:
                        pstate["cash"] = cash + qty_s * last_price
                        pstate["shares"] = shares - qty_s
                        pstate["log"].append({"Date": str(last_date.date()), "Action": "SELL", "Qty": qty_s, "Price": last_price})
                        st.rerun()
                    else:
                        st.error("Not enough shares (or qty=0).")
            with c_flat:
                if st.button("Close position (sell all)", use_container_width=True):
                    if shares > 0:
                        pstate["cash"] = cash + shares * last_price
                        pstate["shares"] = 0.0
                        pstate["log"].append({"Date": str(last_date.date()), "Action": "CLOSE", "Qty": shares, "Price": last_price})
                        st.rerun()

            if pstate["log"]:
                st.subheader("Trade log")
                st.dataframe(pd.DataFrame(pstate["log"]), use_container_width=True, height=220)

        st.subheader("Examples (test set)")
        show_n = st.slider("Sa rreshta me shfaq", 5, 50, 10)
        bt_show = bt.copy()
        bt_show["pred"] = (bt_show["proba_up"] >= 0.5).astype(int)
        cols = [c for c in ["Date", "Label", "pred", "proba_up", "Close", "daily_return", "position", "strategy_return", "equity"] if c in bt_show.columns]
        st.dataframe(bt_show[cols].tail(int(show_n)), use_container_width=True, height=320)

    with tab5:
        st.subheader("What’s innovative here (for a trading user)")
        st.write(
            "- **Signal from text**: probabilitet (P(up)) nga lajmet\n"
            "- **Explainable knobs**: threshold, short on/off, fee bps\n"
            "- **Fast iteration**: ndrysho parametrat dhe shih equity curve menjëherë\n"
            "- **Paper trading**: një mënyrë e thjeshtë për të simuluar “klikimet” e një user-i\n\n"
            "Kufizime: ky dataset është për **DJIA index**, jo për aksione individuale; backtest është i thjeshtuar "
            "(nuk modelon slippage, orare tregu, delay të lajmeve, etj.)."
        )
        st.subheader("Si të përditësosh të dhënat")
        st.code("python DataPreparation/dataPreparation.py", language="bash")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


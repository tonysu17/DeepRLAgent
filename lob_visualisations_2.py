"""
LOB Environment Diagnostic Visualisations
==========================================
Comprehensive visual analysis of the multi-agent LOB simulation.
Run AFTER the Phase 1 environment code (lob_environment.py).

Dependencies: plotly, numpy, scipy, statsmodels
    pip install plotly numpy scipy statsmodels kaleido

Usage:
    python lob_visualisations.py          # interactive HTML plots
    python lob_visualisations.py --static # static PNG fallback
"""

import numpy as np
import sys, os, warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

warnings.filterwarnings("ignore")

# ── Import simulation engine ───────────────────────────────────
# Adjust path if needed
from lob_environment import (
    SimConfig, LOBConfig, HawkesConfig, FundamentalConfig, AgentConfig,
    LimitOrderBook, FundamentalValue, HawkesNoiseTrader, FundamentalTrader,
    MultivariateHawkes, AvellanedaStoikovMM, Portfolio,
    Side, EventType, Fill, MarketMakerEnv,
)

# ── Plotting backend ──────────────────────────────────────────
USE_PLOTLY = "--static" not in sys.argv
if USE_PLOTLY:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        import plotly.io as pio
        pio.templates.default = "plotly_dark"
        BACKEND = "plotly"
    except ImportError:
        USE_PLOTLY = False

if not USE_PLOTLY:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    BACKEND = "matplotlib"

print(f"[Visualisation backend: {BACKEND}]")


# ════════════════════════════════════════════════════════════════
# §1  DATA COLLECTION ENGINE
# ════════════════════════════════════════════════════════════════

@dataclass
class SimData:
    """Container for all recorded simulation time series."""
    timestamps: np.ndarray
    mid_prices: np.ndarray
    fundamentals: np.ndarray
    spreads: np.ndarray
    volumes: np.ndarray
    bid_depth_top5: np.ndarray     # (T, 5)
    ask_depth_top5: np.ndarray     # (T, 5)
    bid_prices_top5: np.ndarray    # (T, 5)
    ask_prices_top5: np.ndarray    # (T, 5)
    # Hawkes
    hawkes_events: List[Tuple[float, int]]
    hawkes_intensities: np.ndarray  # (T, 6)
    # Market maker (A-S)
    mm_inventories: np.ndarray
    mm_equities: np.ndarray
    mm_bid_offsets: np.ndarray
    mm_ask_offsets: np.ndarray
    mm_spreads: np.ndarray
    # Trade tape
    trade_prices: np.ndarray
    trade_sizes: np.ndarray
    trade_sides: np.ndarray  # 0=bid aggressor, 1=ask aggressor


def collect_simulation_data(n_steps: int = 20000, seed: int = 42) -> SimData:
    """Run the environment and record everything for visualisation."""
    cfg = SimConfig(seed=seed, n_market_makers=2, dt=0.1)
    rng = np.random.default_rng(seed)

    lob = LimitOrderBook(cfg.lob)
    lob.seed_book(cfg.lob.initial_mid, rng=rng)
    fund = FundamentalValue(cfg.fundamental, rng)
    noise = HawkesNoiseTrader(cfg.hawkes, cfg.lob, rng)

    mms = [AvellanedaStoikovMM(agent_id=100+i, sigma=0.05,
                                kappa=100.0, gamma=0.1+rng.uniform(-0.02, 0.02))
           for i in range(cfg.n_market_makers)]
    fund_trader = FundamentalTrader(threshold=0.5, intensity=0.15,
                                     max_qty=2, rng=rng)

    # Storage
    ts_list, mids, funds, sprs, vols = [], [], [], [], []
    bd5, ad5, bp5, ap5 = [], [], [], []
    h_events_all, h_intensities = [], []
    mm_inv, mm_eq, mm_bo, mm_ao, mm_spr = [], [], [], [], []
    tp, tsz, tsd = [], [], []

    t = 0.0
    for step in range(n_steps):
        t += cfg.dt

        # Record Hawkes intensities BEFORE events
        h_int = noise.hawkes.intensities().copy()
        h_intensities.append(h_int)

        # Market makers act
        all_fills = []
        for mm in mms:
            f = mm.act(lob, t)
            all_fills.extend(f)

        # Noise trader — FIX: pass fundamental for anchored orders
        nf = noise.generate_orders(lob, cfg.dt, t,
                                    fundamental_price=fund.v)
        all_fills.extend(nf)

        # Fundamental trader — FIX v3: price anchoring
        ft_fills = fund_trader.act(lob, fund.v, t)
        all_fills.extend(ft_fills)

        # Fundamental
        fund.step(cfg.dt)

        # FIX: process ALL fills for MMs (including noise hits on MM quotes)
        for mm in mms:
            mm.process_external_fills(all_fills, lob.mid_price)

        # Record state
        ts_list.append(t)
        mid = lob.mid_price
        mids.append(mid)
        funds.append(fund.v)
        s = lob.spread
        sprs.append(s if s < 1e6 else np.nan)
        vols.append(sum(f.quantity for f in all_fills))

        # LOB depth snapshot
        snap = lob.depth_snapshot(5)
        bd5.append(snap['bid_volumes'])
        ad5.append(snap['ask_volumes'])
        bp5.append(snap['bid_prices'])
        ap5.append(snap['ask_prices'])

        # MM diagnostics (use first MM)
        mm0 = mms[0]
        mm_inv.append(mm0.portfolio.inventory)
        mm_eq.append(mm0.portfolio.equity(mid))
        if mm0.active_oids and len(mm0.active_oids) >= 2:
            # bid and ask offsets from mid
            bo = mid - (lob.order_map.get(mm0.active_oids[0], (type('', (), {'price': mid}),))[0].price) if mm0.active_oids[0] in lob.order_map else 0
            ao = (lob.order_map.get(mm0.active_oids[1], (type('', (), {'price': mid}),))[0].price) - mid if mm0.active_oids[1] in lob.order_map else 0
            mm_bo.append(max(0, bo))
            mm_ao.append(max(0, ao))
            mm_spr.append(max(0, bo) + max(0, ao))
        else:
            mm_bo.append(np.nan)
            mm_ao.append(np.nan)
            mm_spr.append(np.nan)

        # Trade tape
        for f in all_fills:
            tp.append(f.price)
            tsz.append(f.quantity)
            tsd.append(f.aggressor_side)

        if step % 5000 == 0:
            print(f"  Collecting data... step {step}/{n_steps}")

    return SimData(
        timestamps=np.array(ts_list),
        mid_prices=np.array(mids),
        fundamentals=np.array(funds),
        spreads=np.array(sprs),
        volumes=np.array(vols),
        bid_depth_top5=np.array(bd5),
        ask_depth_top5=np.array(ad5),
        bid_prices_top5=np.array(bp5),
        ask_prices_top5=np.array(ap5),
        hawkes_events=h_events_all,
        hawkes_intensities=np.array(h_intensities),
        mm_inventories=np.array(mm_inv),
        mm_equities=np.array(mm_eq),
        mm_bid_offsets=np.array(mm_bo),
        mm_ask_offsets=np.array(mm_ao),
        mm_spreads=np.array(mm_spr),
        trade_prices=np.array(tp) if tp else np.array([]),
        trade_sizes=np.array(tsz) if tsz else np.array([]),
        trade_sides=np.array(tsd) if tsd else np.array([]),
    )


# ════════════════════════════════════════════════════════════════
# §2  DERIVED STATISTICS
# ════════════════════════════════════════════════════════════════

def compute_returns(prices: np.ndarray) -> np.ndarray:
    p = prices[prices > 0]
    return np.diff(np.log(p + 1e-10))

def acf(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Compute autocorrelation function."""
    x = x - x.mean()
    n = len(x)
    var = np.var(x)
    if var < 1e-16:
        return np.zeros(max_lag + 1)
    result = np.correlate(x, x, mode='full')
    result = result[n-1:n+max_lag] / (var * n)
    return result


# ════════════════════════════════════════════════════════════════
# §3  PLOTLY VISUALISATIONS
# ════════════════════════════════════════════════════════════════

from pathlib import Path

def save_fig(fig, name: str):
    save_dir = Path.home() / "Documents" / "Papers" / "DeepRL Agent" / "Plots v4"
    
    if BACKEND == "plotly":
        file_path = save_dir / f"{name}.html"
        fig.write_html(str(file_path), include_plotlyjs='cdn')
        print(f"  Saved {file_path}")
    else:
        file_path = save_dir / f"{name}.png"
        fig.savefig(str(file_path), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {file_path}")


# ── 3.1 Price dynamics ─────────────────────────────────────────

def plot_price_dynamics(d: SimData):
    """Mid-price, fundamental value, and spread over time."""
    if BACKEND == "plotly":
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04,
                            subplot_titles=("Mid-Price vs Fundamental Value",
                                            "Bid-Ask Spread", "Trade Volume"))
        fig.add_trace(go.Scatter(x=d.timestamps, y=d.mid_prices,
                                 name='Mid Price', line=dict(width=1, color='#00d4ff')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=d.timestamps, y=d.fundamentals,
                                 name='Fundamental', line=dict(width=1, color='#ff6b6b', dash='dot')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=d.timestamps, y=d.spreads,
                                 name='Spread', line=dict(width=1, color='#ffd93d'),
                                 fill='tozeroy', fillcolor='rgba(255,217,61,0.1)'),
                      row=2, col=1)
        fig.add_trace(go.Bar(x=d.timestamps, y=d.volumes,
                             name='Volume', marker_color='rgba(0,212,255,0.4)'),
                      row=3, col=1)
        fig.update_layout(height=900, title_text="<b>Market Dynamics Overview</b>",
                          template="plotly_dark", showlegend=True,
                          legend=dict(x=0.01, y=0.99))
        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Spread", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=3, col=1)
    else:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        axes[0].plot(d.timestamps, d.mid_prices, lw=0.8, label='Mid Price')
        axes[0].plot(d.timestamps, d.fundamentals, lw=0.8, ls='--', label='Fundamental')
        axes[0].set_ylabel("Price"); axes[0].legend(); axes[0].set_title("Mid-Price vs Fundamental")
        axes[1].fill_between(d.timestamps, 0, d.spreads, alpha=0.3)
        axes[1].plot(d.timestamps, d.spreads, lw=0.5); axes[1].set_ylabel("Spread")
        axes[2].bar(d.timestamps[::10], d.volumes[::10], width=d.timestamps[1]-d.timestamps[0], alpha=0.6)
        axes[2].set_ylabel("Volume"); axes[2].set_xlabel("Time (s)")
        fig.suptitle("Market Dynamics Overview", fontsize=14)
        fig.tight_layout()
    save_fig(fig, "01_price_dynamics")


# ── 3.2 LOB depth heatmap ──────────────────────────────────────

def plot_lob_heatmap(d: SimData):
    """Animated LOB depth over time as a heatmap."""
    # Subsample for performance
    step = max(1, len(d.timestamps) // 500)
    t_sub = d.timestamps[::step]
    bd = d.bid_depth_top5[::step]
    ad = d.ask_depth_top5[::step]
    bp = d.bid_prices_top5[::step]
    ap = d.ask_prices_top5[::step]

    # Build combined depth matrix: bids (negative) | asks (positive)
    n = len(t_sub)
    depth = np.zeros((n, 10))
    for i in range(5):
        depth[:, 4-i] = -bd[:, i]  # bids, reversed (best bid closest to centre)
        depth[:, 5+i] = ad[:, i]

    if BACKEND == "plotly":
        fig = go.Figure(data=go.Heatmap(
            z=depth.T, x=t_sub,
            y=[f'Bid L{5-i}' for i in range(5)] + [f'Ask L{i+1}' for i in range(5)],
            colorscale=[[0, '#ff4444'], [0.5, '#1a1a2e'], [1, '#44ff44']],
            zmid=0, colorbar=dict(title="Volume<br>(neg=bid)")
        ))
        fig.update_layout(title="<b>Limit Order Book Depth Heatmap</b>",
                          xaxis_title="Time (s)", yaxis_title="Price Level",
                          height=500, template="plotly_dark")
    else:
        fig, ax = plt.subplots(figsize=(16, 5))
        im = ax.imshow(depth.T, aspect='auto', cmap='RdYlGn',
                       extent=[t_sub[0], t_sub[-1], -0.5, 9.5],
                       interpolation='nearest')
        ax.set_yticks(range(10))
        ax.set_yticklabels([f'Bid L{5-i}' for i in range(5)] + [f'Ask L{i+1}' for i in range(5)])
        ax.set_xlabel("Time (s)"); ax.set_title("LOB Depth Heatmap")
        plt.colorbar(im, label="Volume (neg=bid)")
        fig.tight_layout()
    save_fig(fig, "02_lob_heatmap")


# ── 3.3 LOB snapshot ──────────────────────────────────────────

def plot_lob_snapshot(d: SimData, t_idx: int = None):
    """Bar chart of LOB at a specific time step."""
    if t_idx is None:
        t_idx = len(d.timestamps) // 2
    bp = d.bid_prices_top5[t_idx]
    bv = d.bid_depth_top5[t_idx]
    ap = d.ask_prices_top5[t_idx]
    av = d.ask_depth_top5[t_idx]

    # Filter out zero prices
    bid_mask = bp > 0
    ask_mask = ap > 0

    if BACKEND == "plotly":
        fig = go.Figure()
        fig.add_trace(go.Bar(x=bp[bid_mask], y=bv[bid_mask], name='Bids',
                             marker_color='rgba(0,200,100,0.7)', orientation='v',
                             width=0.008))
        fig.add_trace(go.Bar(x=ap[ask_mask], y=av[ask_mask], name='Asks',
                             marker_color='rgba(255,80,80,0.7)', orientation='v',
                             width=0.008))
        mid = d.mid_prices[t_idx]
        fig.add_vline(x=mid, line_dash="dash", line_color="yellow",
                      annotation_text=f"Mid: {mid:.2f}")
        fig.update_layout(title=f"<b>LOB Snapshot at t={d.timestamps[t_idx]:.1f}s</b>",
                          xaxis_title="Price", yaxis_title="Volume",
                          height=450, template="plotly_dark", barmode='overlay')
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(bp[bid_mask], bv[bid_mask], width=0.008, color='green', alpha=0.7, label='Bids')
        ax.bar(ap[ask_mask], av[ask_mask], width=0.008, color='red', alpha=0.7, label='Asks')
        ax.axvline(d.mid_prices[t_idx], ls='--', color='yellow', label=f'Mid: {d.mid_prices[t_idx]:.2f}')
        ax.set_xlabel("Price"); ax.set_ylabel("Volume"); ax.legend()
        ax.set_title(f"LOB Snapshot at t={d.timestamps[t_idx]:.1f}s")
        fig.tight_layout()
    save_fig(fig, "03_lob_snapshot")


# ── 3.4 Hawkes process intensities ────────────────────────────

def plot_hawkes_intensities(d: SimData):
    """Time series of Hawkes process intensities for each event type."""
    labels = ['Limit Buy', 'Limit Sell', 'Mkt Buy', 'Mkt Sell', 'Cancel Bid', 'Cancel Ask']
    colors = ['#00d4ff', '#ff6b6b', '#00ff88', '#ff4444', '#888888', '#bbbbbb']
    step = max(1, len(d.timestamps) // 2000)

    if BACKEND == "plotly":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Limit & Market Order Intensities",
                                            "Cancellation Intensities"))
        for i in range(4):
            fig.add_trace(go.Scatter(
                x=d.timestamps[::step], y=d.hawkes_intensities[::step, i],
                name=labels[i], line=dict(width=1, color=colors[i])),
                row=1, col=1)
        for i in range(4, 6):
            fig.add_trace(go.Scatter(
                x=d.timestamps[::step], y=d.hawkes_intensities[::step, i],
                name=labels[i], line=dict(width=1, color=colors[i])),
                row=2, col=1)
        fig.update_layout(height=600, title_text="<b>Hawkes Process Intensities</b>",
                          template="plotly_dark")
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="λ(t)", row=1, col=1)
        fig.update_yaxes(title_text="λ(t)", row=2, col=1)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        for i in range(4):
            axes[0].plot(d.timestamps[::step], d.hawkes_intensities[::step, i],
                         lw=0.8, label=labels[i])
        for i in range(4, 6):
            axes[1].plot(d.timestamps[::step], d.hawkes_intensities[::step, i],
                         lw=0.8, label=labels[i])
        axes[0].legend(); axes[0].set_ylabel("λ(t)"); axes[0].set_title("Order Intensities")
        axes[1].legend(); axes[1].set_ylabel("λ(t)"); axes[1].set_xlabel("Time (s)")
        fig.suptitle("Hawkes Process Intensities", fontsize=14); fig.tight_layout()
    save_fig(fig, "04_hawkes_intensities")


# ── 3.5 Hawkes inter-arrival times ────────────────────────────

def plot_hawkes_interarrivals(d: SimData):
    """Distribution of volumes per step (proxy for inter-arrival clustering)."""
    vols = d.volumes[d.volumes > 0]
    if len(vols) < 10:
        print("  [Skip] Not enough volume data for inter-arrival plot")
        return

    if BACKEND == "plotly":
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Volume per Timestep Distribution",
                                            "Volume ACF (Clustering)"))
        fig.add_trace(go.Histogram(x=vols, nbinsx=50, name='Volume/step',
                                   marker_color='rgba(0,212,255,0.6)'), row=1, col=1)
        vol_acf = acf(d.volumes, max_lag=50)
        fig.add_trace(go.Bar(x=list(range(len(vol_acf))), y=vol_acf,
                             name='ACF', marker_color='rgba(255,217,61,0.7)'), row=1, col=2)
        fig.add_hline(y=1.96/np.sqrt(len(d.volumes)), line_dash="dash",
                      line_color="red", row=1, col=2)
        fig.add_hline(y=-1.96/np.sqrt(len(d.volumes)), line_dash="dash",
                      line_color="red", row=1, col=2)
        fig.update_layout(height=400, title_text="<b>Order Flow Clustering</b>",
                          template="plotly_dark", showlegend=False)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(vols, bins=50, alpha=0.7); axes[0].set_title("Volume/step Distribution")
        vol_acf = acf(d.volumes, max_lag=50)
        axes[1].bar(range(len(vol_acf)), vol_acf, alpha=0.7)
        axes[1].axhline(1.96/np.sqrt(len(d.volumes)), ls='--', color='r')
        axes[1].axhline(-1.96/np.sqrt(len(d.volumes)), ls='--', color='r')
        axes[1].set_title("Volume ACF"); fig.tight_layout()
    save_fig(fig, "05_hawkes_clustering")


# ════════════════════════════════════════════════════════════════
# §4  STYLISED FACTS VISUALISATIONS
# ════════════════════════════════════════════════════════════════

def plot_stylised_facts(d: SimData):
    """
    Comprehensive 2x3 grid of stylised fact diagnostics:
    [1] Fat tails — QQ plot + log-density
    [2] Volatility clustering — ACF of |r| and r²
    [3] Negative tick autocorrelation — ACF of r at short lags
    [4] Concave market impact — size vs impact scatter
    [5] Mean-reverting spread — spread + ADF
    [6] Volume clustering — ACF of volume
    """
    returns = compute_returns(d.mid_prices)
    returns = returns[np.isfinite(returns)]
    abs_r = np.abs(returns)
    sq_r = returns ** 2

    from scipy import stats as sp_stats

    # ── Compute all statistics ────────────────────────────────
    kurt = sp_stats.kurtosis(returns, fisher=True)
    acf_r = acf(returns, max_lag=30)
    acf_abs = acf(abs_r, max_lag=100)
    acf_sq = acf(sq_r, max_lag=100)
    acf_vol = acf(d.volumes, max_lag=100)

    # QQ data
    qq_theoretical = np.sort(np.random.standard_normal(len(returns)))
    qq_empirical = np.sort(returns / (np.std(returns) + 1e-10))

    # Spread stationarity
    clean_spreads = d.spreads[np.isfinite(d.spreads)]

    # Market impact (bin trades by size, compute avg price change)
    impact_sizes, impact_changes = [], []
    if len(d.trade_sizes) > 100:
        # For each trade, match to nearest timestep and get return
        tp = d.trade_prices
        tsz = d.trade_sizes
        # Simple: bin by size quantile
        for q_lo, q_hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
            lo = np.quantile(tsz, q_lo)
            hi = np.quantile(tsz, q_hi)
            mask = (tsz >= lo) & (tsz < hi + 0.01)
            if mask.sum() > 5:
                avg_size = tsz[mask].mean()
                # Price change proxy: use spread as impact
                avg_impact = np.abs(np.diff(tp[mask])).mean() if mask.sum() > 1 else 0
                impact_sizes.append(avg_size)
                impact_changes.append(avg_impact)

    print(f"\n{'='*50}")
    print(f"  STYLISED FACTS SUMMARY")
    print(f"{'='*50}")
    print(f"  Excess kurtosis:    {kurt:.2f}  {'✓' if kurt > 0 else '✗'} (>0 for fat tails)")
    print(f"  ACF(|r|, lag=10):   {acf_abs[10] if len(acf_abs) > 10 else 'N/A':.4f}  {'✓' if len(acf_abs) > 10 and acf_abs[10] > 0.02 else '✗'}")
    print(f"  ACF(r, lag=1):      {acf_r[1] if len(acf_r) > 1 else 'N/A':.4f}  {'✓' if len(acf_r) > 1 and acf_r[1] < 0 else '✗'} (<0 for bid-ask bounce)")
    print(f"  ACF(vol, lag=5):    {acf_vol[5] if len(acf_vol) > 5 else 'N/A':.4f}  {'✓' if len(acf_vol) > 5 and acf_vol[5] > 0.01 else '✗'}")
    print(f"{'='*50}\n")

    if BACKEND == "plotly":
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                f"<b>①</b> Fat Tails (QQ Plot) — κ={kurt:.1f}",
                f"<b>②</b> Volatility Clustering — ACF(|r|)",
                f"<b>③</b> Tick Autocorrelation — ACF(r)",
                f"<b>④</b> Market Impact — Size vs ΔP",
                f"<b>⑤</b> Mean-Reverting Spread",
                f"<b>⑥</b> Volume Clustering — ACF(Vol)",
            ),
            vertical_spacing=0.12, horizontal_spacing=0.08,
        )

        # ① Fat tails: QQ plot
        n_qq = min(len(qq_theoretical), len(qq_empirical))
        fig.add_trace(go.Scatter(x=qq_theoretical[:n_qq], y=qq_empirical[:n_qq],
                                 mode='markers', marker=dict(size=2, color='#00d4ff'),
                                 name='QQ', showlegend=False), row=1, col=1)
        rng_qq = [min(qq_theoretical.min(), qq_empirical.min()),
                  max(qq_theoretical.max(), qq_empirical.max())]
        fig.add_trace(go.Scatter(x=rng_qq, y=rng_qq,
                                 line=dict(color='red', dash='dash', width=1),
                                 name='Normal ref', showlegend=False), row=1, col=1)

        # ② Volatility clustering: ACF of |r| and r²
        lags = list(range(len(acf_abs)))
        fig.add_trace(go.Scatter(x=lags, y=acf_abs, name='ACF(|r|)',
                                 line=dict(color='#00ff88', width=1.5)), row=1, col=2)
        fig.add_trace(go.Scatter(x=list(range(len(acf_sq))), y=acf_sq, name='ACF(r²)',
                                 line=dict(color='#ffd93d', width=1.5)), row=1, col=2)
        ci = 1.96 / np.sqrt(len(returns))
        fig.add_hline(y=ci, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-ci, line_dash="dash", line_color="red", row=1, col=2)

        # ③ Negative autocorrelation of returns
        fig.add_trace(go.Bar(x=list(range(len(acf_r))), y=acf_r,
                             marker_color=['#ff4444' if v < 0 else '#00ff88' for v in acf_r],
                             name='ACF(r)', showlegend=False), row=1, col=3)
        fig.add_hline(y=ci, line_dash="dash", line_color="red", row=1, col=3)
        fig.add_hline(y=-ci, line_dash="dash", line_color="red", row=1, col=3)

        # ④ Market impact
        if impact_sizes:
            sqrt_x = np.linspace(min(impact_sizes), max(impact_sizes), 50)
            # Fit: impact ∝ sqrt(size)
            coef = np.mean(impact_changes) / (np.mean(np.sqrt(impact_sizes)) + 1e-10)
            fig.add_trace(go.Scatter(x=impact_sizes, y=impact_changes,
                                     mode='markers', marker=dict(size=10, color='#00d4ff'),
                                     name='Empirical', showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=sqrt_x.tolist(), y=(coef * np.sqrt(sqrt_x)).tolist(),
                                     line=dict(color='red', dash='dash'),
                                     name='√size fit', showlegend=False), row=2, col=1)
        else:
            fig.add_annotation(text="Insufficient trade data", xref="x4", yref="y4",
                               x=0.5, y=0.5, showarrow=False, row=2, col=1)

        # ⑤ Mean-reverting spread
        step_s = max(1, len(clean_spreads) // 2000)
        fig.add_trace(go.Scatter(
            x=d.timestamps[:len(clean_spreads)][::step_s],
            y=clean_spreads[::step_s],
            line=dict(width=1, color='#ffd93d'),
            name='Spread', showlegend=False), row=2, col=2)
        if len(clean_spreads) > 10:
            mean_s = np.nanmean(clean_spreads)
            fig.add_hline(y=mean_s, line_dash="dash", line_color="cyan",
                          annotation_text=f"μ={mean_s:.4f}", row=2, col=2)

        # ⑥ Volume clustering
        fig.add_trace(go.Bar(x=list(range(len(acf_vol))), y=acf_vol,
                             marker_color='rgba(0,212,255,0.6)',
                             name='ACF(Vol)', showlegend=False), row=2, col=3)
        ci_v = 1.96 / np.sqrt(len(d.volumes))
        fig.add_hline(y=ci_v, line_dash="dash", line_color="red", row=2, col=3)
        fig.add_hline(y=-ci_v, line_dash="dash", line_color="red", row=2, col=3)

        fig.update_layout(height=750, width=1400,
                          title_text="<b>Stylised Facts Diagnostic Panel</b>",
                          template="plotly_dark", showlegend=True,
                          legend=dict(x=0.35, y=1.02, orientation='h'))
        fig.update_xaxes(title_text="Normal Quantiles", row=1, col=1)
        fig.update_yaxes(title_text="Empirical Quantiles", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)
        fig.update_xaxes(title_text="Lag", row=1, col=3)
        fig.update_xaxes(title_text="Trade Size", row=2, col=1)
        fig.update_yaxes(title_text="Price Impact", row=2, col=1)
        fig.update_xaxes(title_text="Time (s)", row=2, col=2)
        fig.update_xaxes(title_text="Lag", row=2, col=3)

    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        # ① QQ
        n_qq = min(len(qq_theoretical), len(qq_empirical))
        axes[0,0].scatter(qq_theoretical[:n_qq], qq_empirical[:n_qq], s=1, alpha=0.5)
        rng_qq = [min(qq_theoretical.min(), qq_empirical.min()), max(qq_theoretical.max(), qq_empirical.max())]
        axes[0,0].plot(rng_qq, rng_qq, 'r--', lw=1)
        axes[0,0].set_title(f'① Fat Tails QQ (κ={kurt:.1f})'); axes[0,0].set_xlabel("Normal"); axes[0,0].set_ylabel("Empirical")
        # ② Vol clustering
        axes[0,1].plot(acf_abs, label='ACF(|r|)'); axes[0,1].plot(acf_sq, label='ACF(r²)')
        axes[0,1].axhline(ci, ls='--', color='r'); axes[0,1].axhline(-ci, ls='--', color='r')
        axes[0,1].legend(); axes[0,1].set_title('② Volatility Clustering'); axes[0,1].set_xlabel("Lag")
        # ③ Return ACF
        colors_bar = ['red' if v < 0 else 'green' for v in acf_r]
        axes[0,2].bar(range(len(acf_r)), acf_r, color=colors_bar, alpha=0.7)
        axes[0,2].axhline(ci, ls='--', color='r'); axes[0,2].axhline(-ci, ls='--', color='r')
        axes[0,2].set_title('③ Return ACF (bid-ask bounce)'); axes[0,2].set_xlabel("Lag")
        # ④ Market impact
        if impact_sizes:
            axes[1,0].scatter(impact_sizes, impact_changes, s=50, zorder=5)
            sx = np.linspace(min(impact_sizes), max(impact_sizes), 50)
            coef = np.mean(impact_changes) / (np.mean(np.sqrt(impact_sizes)) + 1e-10)
            axes[1,0].plot(sx, coef*np.sqrt(sx), 'r--', label='√size fit')
            axes[1,0].legend()
        axes[1,0].set_title('④ Market Impact'); axes[1,0].set_xlabel("Size"); axes[1,0].set_ylabel("ΔP")
        # ⑤ Spread
        step_s = max(1, len(clean_spreads) // 2000)
        axes[1,1].plot(d.timestamps[:len(clean_spreads)][::step_s], clean_spreads[::step_s], lw=0.5)
        axes[1,1].axhline(np.nanmean(clean_spreads), ls='--', color='r')
        axes[1,1].set_title('⑤ Mean-Reverting Spread'); axes[1,1].set_xlabel("Time (s)")
        # ⑥ Volume ACF
        axes[1,2].bar(range(len(acf_vol)), acf_vol, alpha=0.6)
        axes[1,2].axhline(1.96/np.sqrt(len(d.volumes)), ls='--', color='r')
        axes[1,2].set_title('⑥ Volume Clustering ACF'); axes[1,2].set_xlabel("Lag")
        fig.suptitle("Stylised Facts Diagnostic Panel", fontsize=14)
        fig.tight_layout()

    save_fig(fig, "06_stylised_facts")


# ── 4.2 Fat tails deep dive ───────────────────────────────────

def plot_fat_tails_detail(d: SimData):
    """Log-density plot comparing returns to normal + fitted t-distribution."""
    returns = compute_returns(d.mid_prices)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 100:
        print("  [Skip] Not enough returns for fat tails detail")
        return

    from scipy import stats as sp_stats

    # Standardise
    r_std = (returns - returns.mean()) / (returns.std() + 1e-10)

    # Histogram bins
    bins = np.linspace(-6, 6, 120)
    hist, edges = np.histogram(r_std, bins=bins, density=True)
    centres = (edges[:-1] + edges[1:]) / 2

    # Normal PDF
    norm_pdf = sp_stats.norm.pdf(centres)

    # Fit Student-t
    df_fit, loc_fit, scale_fit = sp_stats.t.fit(r_std)
    t_pdf = sp_stats.t.pdf(centres, df_fit, loc_fit, scale_fit)

    if BACKEND == "plotly":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=centres.tolist(), y=np.log10(hist + 1e-10).tolist(),
                                 mode='markers', marker=dict(size=4, color='#00d4ff'),
                                 name='Empirical'))
        fig.add_trace(go.Scatter(x=centres.tolist(), y=np.log10(norm_pdf + 1e-10).tolist(),
                                 line=dict(color='red', dash='dash', width=2),
                                 name='Normal'))
        fig.add_trace(go.Scatter(x=centres.tolist(), y=np.log10(t_pdf + 1e-10).tolist(),
                                 line=dict(color='#00ff88', width=2),
                                 name=f'Student-t (ν={df_fit:.1f})'))
        fig.update_layout(title="<b>Return Distribution — Fat Tails Detail</b>",
                          xaxis_title="Standardised Returns",
                          yaxis_title="log₁₀(Density)",
                          template="plotly_dark", height=500)
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(centres, np.log10(hist + 1e-10), s=8, label='Empirical', alpha=0.7)
        ax.plot(centres, np.log10(norm_pdf + 1e-10), 'r--', lw=2, label='Normal')
        ax.plot(centres, np.log10(t_pdf + 1e-10), 'g-', lw=2, label=f't (ν={df_fit:.1f})')
        ax.set_xlabel("Standardised Returns"); ax.set_ylabel("log₁₀(Density)")
        ax.legend(); ax.set_title("Return Distribution — Fat Tails"); fig.tight_layout()
    save_fig(fig, "07_fat_tails_detail")


# ════════════════════════════════════════════════════════════════
# §5  MARKET MAKER DIAGNOSTICS
# ════════════════════════════════════════════════════════════════

def plot_market_maker_diagnostics(d: SimData):
    """
    Key MM behavioural diagnostics:
    - Inventory over time
    - Equity over time
    - Spread vs |inventory| (does MM widen when inventory is large?)
    - Spread vs recent volatility (does MM widen in volatile markets?)
    """
    inv = d.mm_inventories
    eq = d.mm_equities
    spr = d.mm_spreads
    ts = d.timestamps

    # Compute rolling volatility (20-step window)
    returns = compute_returns(d.mid_prices)
    rvol = np.zeros(len(d.timestamps))
    w = 20
    for i in range(w, len(returns)):
        rvol[i+1] = np.std(returns[i-w:i])

    # Clean nans
    clean_mask = np.isfinite(spr) & np.isfinite(inv) & (spr > 0)

    if BACKEND == "plotly":
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "MM Inventory Over Time",
                "MM Equity Over Time",
                "Spread vs |Inventory| — Adverse Selection Response",
                "Spread vs Realised Volatility — Volatility Response",
            ),
            vertical_spacing=0.12, horizontal_spacing=0.08,
        )

        # Inventory
        step = max(1, len(ts) // 2000)
        colors_inv = ['rgba(255,80,80,0.8)' if v > 0 else 'rgba(80,255,80,0.8)' for v in inv[::step]]
        fig.add_trace(go.Scatter(x=ts[::step], y=inv[::step],
                                 mode='lines', line=dict(width=1, color='#00d4ff'),
                                 name='Inventory', fill='tozeroy',
                                 fillcolor='rgba(0,212,255,0.15)'),
                      row=1, col=1)

        # Equity
        fig.add_trace(go.Scatter(x=ts[::step], y=eq[::step],
                                 line=dict(width=1.5, color='#00ff88'),
                                 name='Equity'), row=1, col=2)

        # Spread vs |Inventory|
        if clean_mask.sum() > 50:
            abs_inv = np.abs(inv[clean_mask])
            spr_clean = spr[clean_mask]
            # Bin by inventory
            n_bins = 15
            bins = np.linspace(abs_inv.min(), abs_inv.max(), n_bins + 1)
            bin_means_x, bin_means_y, bin_stds = [], [], []
            for b in range(n_bins):
                m = (abs_inv >= bins[b]) & (abs_inv < bins[b+1])
                if m.sum() > 3:
                    bin_means_x.append((bins[b] + bins[b+1]) / 2)
                    bin_means_y.append(np.mean(spr_clean[m]))
                    bin_stds.append(np.std(spr_clean[m]))

            fig.add_trace(go.Scatter(x=abs_inv[::5].tolist(), y=spr_clean[::5].tolist(),
                                     mode='markers', marker=dict(size=2, color='rgba(0,212,255,0.2)'),
                                     name='Raw', showlegend=False), row=2, col=1)
            fig.add_trace(go.Scatter(x=bin_means_x, y=bin_means_y,
                                     mode='lines+markers', marker=dict(size=8, color='#ff6b6b'),
                                     line=dict(width=2.5, color='#ff6b6b'),
                                     name='Binned Mean'), row=2, col=1)

        # Spread vs Volatility
        if clean_mask.sum() > 50:
            rvol_clean = rvol[:len(clean_mask)][clean_mask]
            spr_clean2 = spr[clean_mask]
            valid = (rvol_clean > 0) & np.isfinite(rvol_clean)
            if valid.sum() > 50:
                rv = rvol_clean[valid]
                sp = spr_clean2[valid]
                n_bins = 15
                bins = np.linspace(np.percentile(rv, 1), np.percentile(rv, 99), n_bins+1)
                bx, by = [], []
                for b in range(n_bins):
                    m = (rv >= bins[b]) & (rv < bins[b+1])
                    if m.sum() > 3:
                        bx.append((bins[b]+bins[b+1])/2)
                        by.append(np.mean(sp[m]))
                fig.add_trace(go.Scatter(x=rv[::5].tolist(), y=sp[::5].tolist(),
                                         mode='markers', marker=dict(size=2, color='rgba(255,217,61,0.15)'),
                                         showlegend=False), row=2, col=2)
                fig.add_trace(go.Scatter(x=bx, y=by,
                                         mode='lines+markers', marker=dict(size=8, color='#ffd93d'),
                                         line=dict(width=2.5, color='#ffd93d'),
                                         name='Binned Mean Vol'), row=2, col=2)

        fig.update_layout(height=800, width=1200,
                          title_text="<b>Market Maker Behavioural Diagnostics</b>",
                          template="plotly_dark")
        fig.update_xaxes(title_text="Time (s)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_xaxes(title_text="|Inventory|", row=2, col=1)
        fig.update_yaxes(title_text="Quoted Spread", row=2, col=1)
        fig.update_xaxes(title_text="Realised Volatility", row=2, col=2)
        fig.update_yaxes(title_text="Quoted Spread", row=2, col=2)

    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        step = max(1, len(ts) // 2000)
        axes[0,0].fill_between(ts[::step], 0, inv[::step], alpha=0.3)
        axes[0,0].plot(ts[::step], inv[::step], lw=0.8)
        axes[0,0].set_title('Inventory Over Time'); axes[0,0].set_xlabel("Time (s)")
        axes[0,1].plot(ts[::step], eq[::step], lw=1)
        axes[0,1].set_title('Equity Over Time'); axes[0,1].set_xlabel("Time (s)")
        if clean_mask.sum() > 50:
            axes[1,0].scatter(np.abs(inv[clean_mask])[::5], spr[clean_mask][::5], s=1, alpha=0.2)
            axes[1,0].set_title('Spread vs |Inventory|'); axes[1,0].set_xlabel("|Inventory|"); axes[1,0].set_ylabel("Spread")
        if clean_mask.sum() > 50:
            valid = (rvol[:len(clean_mask)][clean_mask] > 0)
            axes[1,1].scatter(rvol[:len(clean_mask)][clean_mask][valid][::5],
                              spr[clean_mask][valid][::5], s=1, alpha=0.2)
            axes[1,1].set_title('Spread vs Volatility'); axes[1,1].set_xlabel("RVol"); axes[1,1].set_ylabel("Spread")
        fig.suptitle("Market Maker Diagnostics", fontsize=14); fig.tight_layout()

    save_fig(fig, "08_mm_diagnostics")


# ════════════════════════════════════════════════════════════════
# §6  RETURN DISTRIBUTION DEEP DIVE
# ════════════════════════════════════════════════════════════════

def plot_returns_analysis(d: SimData):
    """Multi-scale return analysis: tick, 10-tick, 100-tick."""
    prices = d.mid_prices[d.mid_prices > 0]

    if BACKEND == "plotly":
        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=("1-Tick Returns", "10-Tick Returns", "100-Tick Returns"))
        for i, scale in enumerate([1, 10, 100]):
            if len(prices) > scale * 2:
                r = np.diff(np.log(prices[::scale]))
                r = r[np.isfinite(r)]
                from scipy import stats as sp_stats
                k = sp_stats.kurtosis(r, fisher=True)
                fig.add_trace(go.Histogram(x=r, nbinsx=80, name=f'{scale}-tick (κ={k:.1f})',
                                           marker_color=['#00d4ff','#00ff88','#ffd93d'][i],
                                           opacity=0.7),
                              row=1, col=i+1)
        fig.update_layout(height=400, title_text="<b>Multi-Scale Return Distributions</b>",
                          template="plotly_dark", showlegend=True)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for i, scale in enumerate([1, 10, 100]):
            if len(prices) > scale * 2:
                r = np.diff(np.log(prices[::scale]))
                r = r[np.isfinite(r)]
                from scipy import stats as sp_stats
                k = sp_stats.kurtosis(r, fisher=True)
                axes[i].hist(r, bins=80, alpha=0.7, density=True)
                axes[i].set_title(f'{scale}-tick returns (κ={k:.1f})')
        fig.suptitle("Multi-Scale Returns", fontsize=14); fig.tight_layout()
    save_fig(fig, "09_returns_multiscale")


# ════════════════════════════════════════════════════════════════
# §7  VOLATILITY CLUSTERING DEEP DIVE
# ════════════════════════════════════════════════════════════════

def plot_volatility_clustering(d: SimData):
    """
    Absolute returns time series + ACF showing long-memory clustering.
    """
    returns = compute_returns(d.mid_prices)
    returns = returns[np.isfinite(returns)]
    abs_r = np.abs(returns)

    step = max(1, len(abs_r) // 3000)
    acf_abs = acf(abs_r, max_lag=200)

    if BACKEND == "plotly":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                            subplot_titles=("Absolute Returns — Volatility Clustering Visible",
                                            "ACF of |Returns| — Slow Decay = Long Memory"),
                            vertical_spacing=0.12)
        fig.add_trace(go.Scatter(
            x=list(range(0, len(abs_r), step)), y=abs_r[::step].tolist(),
            mode='lines', line=dict(width=0.5, color='#ff6b6b'),
            name='|r_t|'), row=1, col=1)
        # Rolling volatility
        w = 50
        if len(abs_r) > w:
            rvol = np.convolve(abs_r, np.ones(w)/w, mode='valid')
            fig.add_trace(go.Scatter(
                x=list(range(w, w + len(rvol), max(1, len(rvol)//2000))),
                y=rvol[::max(1, len(rvol)//2000)].tolist(),
                line=dict(width=2, color='#ffd93d'),
                name=f'MA({w})'), row=1, col=1)

        ci = 1.96 / np.sqrt(len(returns))
        fig.add_trace(go.Bar(x=list(range(len(acf_abs))), y=acf_abs.tolist(),
                             marker_color='rgba(0,255,136,0.6)',
                             name='ACF(|r|)'), row=2, col=1)
        fig.add_hline(y=ci, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-ci, line_dash="dash", line_color="red", row=2, col=1)
        fig.update_layout(height=700, title_text="<b>Volatility Clustering Analysis</b>",
                          template="plotly_dark")
        fig.update_xaxes(title_text="Timestep", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=2, col=1)
        fig.update_yaxes(title_text="|Return|", row=1, col=1)
        fig.update_yaxes(title_text="Autocorrelation", row=2, col=1)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        axes[0].plot(abs_r[::step], lw=0.5, color='red', alpha=0.7)
        if len(abs_r) > 50:
            rvol = np.convolve(abs_r, np.ones(50)/50, mode='valid')
            axes[0].plot(range(50, 50+len(rvol)), rvol, lw=2, color='gold')
        axes[0].set_title("Absolute Returns"); axes[0].set_ylabel("|r|")
        axes[1].bar(range(len(acf_abs)), acf_abs, alpha=0.6, color='green')
        ci = 1.96/np.sqrt(len(returns))
        axes[1].axhline(ci, ls='--', color='r'); axes[1].axhline(-ci, ls='--', color='r')
        axes[1].set_title("ACF of |Returns|"); axes[1].set_xlabel("Lag")
        fig.suptitle("Volatility Clustering", fontsize=14); fig.tight_layout()
    save_fig(fig, "10_volatility_clustering")


# ════════════════════════════════════════════════════════════════
# §8  HAWKES CROSS-EXCITATION MATRIX
# ════════════════════════════════════════════════════════════════

def plot_hawkes_excitation_matrix(cfg: SimConfig = None):
    """Visualise the α (excitation) matrix as a heatmap."""
    if cfg is None:
        cfg = SimConfig()
    labels = ['Lim Buy', 'Lim Sell', 'Mkt Buy', 'Mkt Sell', 'Can Bid', 'Can Ask']
    alpha = cfg.hawkes.alpha

    if BACKEND == "plotly":
        fig = go.Figure(data=go.Heatmap(
            z=alpha, x=labels, y=labels,
            colorscale='Viridis', text=np.round(alpha, 3).tolist(),
            texttemplate="%{text}", textfont={"size": 11},
            colorbar=dict(title="α<sub>ij</sub>"),
        ))
        fig.update_layout(title="<b>Hawkes Excitation Matrix α<sub>ij</sub></b><br>"
                                "<sub>Row i is excited by column j events</sub>",
                          height=500, width=600, template="plotly_dark",
                          xaxis_title="Trigger Event j", yaxis_title="Excited Type i")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(alpha, cmap='viridis')
        ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(range(6)); ax.set_yticklabels(labels)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f'{alpha[i,j]:.3f}', ha='center', va='center', fontsize=9)
        plt.colorbar(im); ax.set_title("Hawkes Excitation Matrix αᵢⱼ")
        ax.set_xlabel("Trigger j"); ax.set_ylabel("Excited i"); fig.tight_layout()
    save_fig(fig, "11_hawkes_excitation_matrix")


# ════════════════════════════════════════════════════════════════
# §9  RL TRAINING COMPARISON (if model exists)
# ════════════════════════════════════════════════════════════════

def plot_rl_vs_as_comparison():
    """
    Run both an RL agent (random policy baseline) and A-S agent
    and compare PnL trajectories. Replace random with trained model.
    """
    cfg = SimConfig(episode_steps=2000, seed=42)
    env = MarketMakerEnv(cfg)

    # A-S equity comes from the environment's built-in A-S MMs
    obs, _ = env.reset()
    rl_equities, as_equities = [], []
    rl_inventories = []

    for step in range(2000):
        # Random policy (replace with trained model)
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        rl_equities.append(info['equity'])
        rl_inventories.append(info['inventory'])
        # A-S equity from first baseline MM
        if env.as_mms:
            as_equities.append(env.as_mms[0].portfolio.equity(info['mid_price']))
        if done:
            break

    ts = np.arange(len(rl_equities)) * cfg.dt

    if BACKEND == "plotly":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Equity: Random RL vs A-S Baseline",
                                            "RL Agent Inventory"))
        fig.add_trace(go.Scatter(x=ts.tolist(), y=rl_equities, name='RL (random)',
                                 line=dict(color='#00d4ff', width=1.5)), row=1, col=1)
        if as_equities:
            fig.add_trace(go.Scatter(x=ts.tolist(), y=as_equities, name='A-S Baseline',
                                     line=dict(color='#ff6b6b', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=ts.tolist(), y=rl_inventories, name='Inventory',
                                 line=dict(color='#ffd93d', width=1),
                                 fill='tozeroy', fillcolor='rgba(255,217,61,0.1)'),
                      row=2, col=1)
        fig.update_layout(height=600,
                          title_text="<b>RL Agent vs Avellaneda-Stoikov Benchmark</b>",
                          template="plotly_dark")
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Inventory", row=2, col=1)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(ts, rl_equities, label='RL (random)')
        if as_equities:
            axes[0].plot(ts, as_equities, label='A-S Baseline')
        axes[0].legend(); axes[0].set_title("Equity Comparison"); axes[0].set_ylabel("Equity")
        axes[1].fill_between(ts, 0, rl_inventories, alpha=0.3)
        axes[1].plot(ts, rl_inventories, lw=0.8)
        axes[1].set_title("RL Inventory"); axes[1].set_xlabel("Time (s)")
        fig.suptitle("RL vs A-S Benchmark", fontsize=14); fig.tight_layout()
    save_fig(fig, "12_rl_vs_as_benchmark")


# ════════════════════════════════════════════════════════════════
# §10  MASTER RUNNER
# ════════════════════════════════════════════════════════════════

def main():
    os.makedirs("plots", exist_ok=True)
    print("=" * 60)
    print("  LOB Environment — Diagnostic Visualisation Suite")
    print("=" * 60)

    print("\n[1/2] Collecting simulation data (20,000 steps)...")
    data = collect_simulation_data(n_steps=20000, seed=42)

    print(f"\n[2/2] Generating visualisations ({BACKEND})...\n")

    print("  → Price dynamics & overview...")
    plot_price_dynamics(data)

    print("  → LOB depth heatmap...")
    plot_lob_heatmap(data)

    print("  → LOB snapshot...")
    plot_lob_snapshot(data)

    print("  → Hawkes intensities...")
    plot_hawkes_intensities(data)

    print("  → Order flow clustering...")
    plot_hawkes_interarrivals(data)

    print("  → Hawkes excitation matrix...")
    plot_hawkes_excitation_matrix()

    print("  → Stylised facts panel...")
    plot_stylised_facts(data)

    print("  → Fat tails detail (log-density)...")
    plot_fat_tails_detail(data)

    print("  → Multi-scale returns...")
    plot_returns_analysis(data)

    print("  → Volatility clustering deep dive...")
    plot_volatility_clustering(data)

    print("  → Market maker diagnostics...")
    plot_market_maker_diagnostics(data)

    print("  → RL vs A-S benchmark...")
    plot_rl_vs_as_comparison()

    print(f"\n{'='*60}")
    print(f"  All plots saved to ./plots/")
    print(f"  12 visualisations generated ({BACKEND} backend)")
    print(f"{'='*60}")

    if BACKEND == "plotly":
        print("\n  Open any .html file in a browser for interactive plots.")
        print("  Use --static flag for PNG fallback with matplotlib.")


if __name__ == "__main__":
    main()

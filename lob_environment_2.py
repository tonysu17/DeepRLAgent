"""
Multi-Agent Limit Order Book Environment — Bug-Fixed
=====================================================
Phase 1: Core LOB engine, Hawkes noise, RL MM/IT interfaces.

Key fixes vs v1:
  - A-S MM parameters (kappa, sigma, T) now produce realistic spreads
  - MM fill tracking: noise-trader market orders hitting MM quotes are
    routed back to MM portfolios after each step
  - Noise trader stale-OID cleanup: filled/cancelled OIDs purged each step
  - Cancellation rates increased to prevent unbounded book growth
  - Fundamental volatility increased for realistic price dynamics
  - Order expiry: noise limit orders older than max_age get force-cancelled

Dependencies: numpy, gymnasium, sortedcontainers
"""

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
from typing import Optional, List, Dict, Tuple, NamedTuple
from sortedcontainers import SortedDict
import gymnasium as gym
from gymnasium import spaces


# ════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════

@dataclass
class LOBConfig:
    tick_size: float = 0.01
    initial_mid: float = 100.0
    max_depth_levels: int = 20
    max_order_qty: int = 50
    lot_size: int = 1

@dataclass
class HawkesConfig:
    """
    6 event types: limit_buy, limit_sell, mkt_buy, mkt_sell, cancel_bid, cancel_ask
    """
    n_types: int = 6
    mu: np.ndarray = field(default_factory=lambda: np.array([
        1.2,   # limit buy
        1.2,   # limit sell
        0.8,   # market buy  — FIX v4: raised from 0.5 for more frequent trades
        0.8,   # market sell — FIX v4: raised from 0.5
        2.0,   # cancel bid
        2.0,   # cancel ask
    ]))
    alpha: np.ndarray = field(default_factory=lambda: np.array([
        # lb    ls    mb    ms    cb    ca
        [0.15, 0.05, 0.20, 0.05, 0.02, 0.02],  # limit buy
        [0.05, 0.15, 0.05, 0.20, 0.02, 0.02],  # limit sell
        [0.05, 0.02, 0.30, 0.15, 0.01, 0.01],  # market buy  — raised self
        [0.02, 0.05, 0.15, 0.30, 0.01, 0.01],  # market sell — raised self
        [0.02, 0.02, 0.15, 0.40, 0.08, 0.02],  # cancel bid (ms→cb raised)
        [0.02, 0.02, 0.40, 0.15, 0.02, 0.08],  # cancel ask (mb→ca raised)
    ]))
    beta: np.ndarray = field(default_factory=lambda: np.array([
        [1.5, 1.5, 2.0, 2.0, 2.0, 2.0],
        [1.5, 1.5, 2.0, 2.0, 2.0, 2.0],
        [2.0, 2.0, 3.0, 2.5, 2.0, 2.0],
        [2.0, 2.0, 2.5, 3.0, 2.0, 2.0],
        [2.0, 2.0, 2.5, 2.5, 1.5, 1.5],
        [2.0, 2.0, 2.5, 2.5, 1.5, 1.5],
    ]))
    price_placement_exp: float = 1.5
    max_ticks_from_mid: int = 10
    size_mu: float = 1.0
    size_sigma: float = 0.5
    max_order_age: float = 30.0
    fundamental_anchor_frac: float = 0.15  # FIX v4: reduced from 0.3 — less dampening

@dataclass
class FundamentalConfig:
    kappa: float = 0.05            # mean-reversion speed
    sigma: float = 0.12            # FIX v4: raised from 0.05 — need visible price dynamics
    v_bar: float = 100.0
    signal_noise_std: float = 0.3

@dataclass
class AgentConfig:
    max_inventory: int = 100
    max_quote_offset: float = 0.50
    max_order_size: int = 20
    inventory_penalty: float = 0.001
    tc_penalty: float = 1.0

@dataclass
class SimConfig:
    lob: LOBConfig = field(default_factory=LOBConfig)
    hawkes: HawkesConfig = field(default_factory=HawkesConfig)
    fundamental: FundamentalConfig = field(default_factory=FundamentalConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    dt: float = 0.1
    episode_steps: int = 5000
    n_market_makers: int = 2
    n_informed: int = 1
    obs_window: int = 50
    seed: Optional[int] = None


# ════════════════════════════════════════════════════════════════
# §2  DATA TYPES
# ════════════════════════════════════════════════════════════════

class Side(IntEnum):
    BID = 0
    ASK = 1

class EventType(IntEnum):
    LIMIT_BUY = 0
    LIMIT_SELL = 1
    MKT_BUY = 2
    MKT_SELL = 3
    CANCEL_BID = 4
    CANCEL_ASK = 5

class Fill(NamedTuple):
    price: float
    quantity: int
    aggressor_id: int
    passive_id: int
    aggressor_side: int
    timestamp: float

@dataclass
class Order:
    oid: int
    agent_id: int
    side: Side
    price: float
    qty: int
    ts: float


# ════════════════════════════════════════════════════════════════
# §3  LIMIT ORDER BOOK ENGINE
# ════════════════════════════════════════════════════════════════

class LimitOrderBook:
    def __init__(self, cfg: LOBConfig):
        self.cfg = cfg
        self.bids: SortedDict = SortedDict(lambda p: -p)
        self.asks: SortedDict = SortedDict()
        self.order_map: Dict[int, Tuple[Order, Side]] = {}
        self._oid = 0
        self.trade_tape: List[Fill] = []
        self._mid = cfg.initial_mid

    def _snap_price(self, p: float) -> float:
        return round(p / self.cfg.tick_size) * self.cfg.tick_size

    def _next_oid(self) -> int:
        self._oid += 1
        return self._oid

    def _update_mid(self):
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            self._mid = (bb + ba) / 2.0
        elif bb is not None:
            self._mid = bb
        elif ba is not None:
            self._mid = ba

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids.keys()[0] if len(self.bids) > 0 else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks.keys()[0] if len(self.asks) > 0 else None

    @property
    def mid_price(self) -> float:
        return self._mid

    @property
    def spread(self) -> float:
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None:
            return ba - bb
        return float('inf')

    def order_exists(self, oid: int) -> bool:
        """FIX: check if an order is still live in the book."""
        return oid in self.order_map

    def get_agent_oids(self, agent_id: int) -> List[int]:
        """FIX: get all live OIDs for a given agent."""
        return [oid for oid, (o, _) in self.order_map.items()
                if o.agent_id == agent_id]

    def total_depth(self) -> Tuple[int, int]:
        """Total bid/ask volume for diagnostics."""
        bv = sum(sum(o.qty for o in q) for q in self.bids.values())
        av = sum(sum(o.qty for o in q) for q in self.asks.values())
        return bv, av

    def submit_limit(self, agent_id: int, side: Side, price: float,
                     qty: int, ts: float) -> Tuple[int, List[Fill]]:
        price = self._snap_price(price)
        o = Order(self._next_oid(), agent_id, side, price, qty, ts)
        fills = self._match(o, ts)
        if o.qty > 0:
            book = self.bids if side == Side.BID else self.asks
            if price not in book:
                book[price] = deque()
            book[price].append(o)
            self.order_map[o.oid] = (o, side)
        self._update_mid()
        return o.oid, fills

    def submit_market(self, agent_id: int, side: Side,
                      qty: int, ts: float) -> Tuple[int, List[Fill]]:
        price = 1e9 if side == Side.BID else -1e9
        o = Order(self._next_oid(), agent_id, side, price, qty, ts)
        fills = self._match(o, ts)
        self._update_mid()
        return o.oid, fills

    def cancel(self, oid: int) -> bool:
        if oid not in self.order_map:
            return False
        o, side = self.order_map[oid]
        book = self.bids if side == Side.BID else self.asks
        if o.price in book:
            q = book[o.price]
            try:
                q.remove(o)
            except ValueError:
                del self.order_map[oid]
                return False
            if len(q) == 0:
                del book[o.price]
        del self.order_map[oid]
        self._update_mid()
        return True

    def cancel_agent_orders(self, agent_id: int) -> int:
        to_cancel = [oid for oid, (o, _) in self.order_map.items()
                     if o.agent_id == agent_id]
        for oid in to_cancel:
            self.cancel(oid)
        return len(to_cancel)

    def cancel_orders_older_than(self, agent_id: int, max_age: float,
                                  current_time: float) -> int:
        """FIX: expire old orders to prevent book accumulation."""
        to_cancel = [
            oid for oid, (o, _) in self.order_map.items()
            if o.agent_id == agent_id and (current_time - o.ts) > max_age
        ]
        for oid in to_cancel:
            self.cancel(oid)
        return len(to_cancel)

    def _match(self, incoming: Order, ts: float) -> List[Fill]:
        fills = []
        if incoming.side == Side.BID:
            contra = self.asks
            def acceptable(p): return p <= incoming.price
        else:
            contra = self.bids
            def acceptable(p): return p >= incoming.price

        while incoming.qty > 0 and len(contra) > 0:
            bp = contra.keys()[0]
            if not acceptable(bp):
                break
            q = contra[bp]
            while incoming.qty > 0 and len(q) > 0:
                resting = q[0]
                fq = min(incoming.qty, resting.qty)
                f = Fill(price=bp, quantity=fq,
                         aggressor_id=incoming.agent_id,
                         passive_id=resting.agent_id,
                         aggressor_side=int(incoming.side),
                         timestamp=ts)
                fills.append(f)
                self.trade_tape.append(f)
                incoming.qty -= fq
                resting.qty -= fq
                if resting.qty == 0:
                    q.popleft()
                    self.order_map.pop(resting.oid, None)
            if len(q) == 0:
                del contra[bp]
        return fills

    def depth_snapshot(self, n: int) -> Dict[str, np.ndarray]:
        bid_p, bid_v = np.zeros(n), np.zeros(n)
        ask_p, ask_v = np.zeros(n), np.zeros(n)
        for i, (p, q) in enumerate(self.bids.items()):
            if i >= n: break
            bid_p[i] = p
            bid_v[i] = sum(o.qty for o in q)
        for i, (p, q) in enumerate(self.asks.items()):
            if i >= n: break
            ask_p[i] = p
            ask_v[i] = sum(o.qty for o in q)
        return {'bid_prices': bid_p, 'bid_volumes': bid_v,
                'ask_prices': ask_p, 'ask_volumes': ask_v}

    def get_state_vector(self, n_levels: int = 10) -> np.ndarray:
        snap = self.depth_snapshot(n_levels)
        mid = self.mid_price
        spr = self.spread if self.spread < 1e6 else 0.0
        bv, av = snap['bid_volumes'], snap['ask_volumes']
        total = bv.sum() + av.sum()
        imbalance = (bv.sum() - av.sum()) / (total + 1e-8)
        return np.concatenate([[mid, spr], bv, av, [imbalance]])

    def seed_book(self, mid: float, n_levels: int = 15,
                  base_qty: int = 8, rng: np.random.Generator = None):
        """FIX v3: increased depth to prevent thin-book drift artifacts."""
        if rng is None:
            rng = np.random.default_rng()
        ts = self.cfg.tick_size
        for i in range(1, n_levels + 1):
            qty = max(1, int(base_qty * np.exp(-0.1 * i) + rng.poisson(3)))
            bp = self._snap_price(mid - i * ts)
            ap = self._snap_price(mid + i * ts)
            self.submit_limit(-1, Side.BID, bp, qty, 0.0)
            self.submit_limit(-1, Side.ASK, ap, qty, 0.0)


# ════════════════════════════════════════════════════════════════
# §4  FUNDAMENTAL VALUE PROCESS
# ════════════════════════════════════════════════════════════════

class FundamentalValue:
    def __init__(self, cfg: FundamentalConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.v = cfg.v_bar

    def step(self, dt: float) -> float:
        c = self.cfg
        dv = c.kappa * (c.v_bar - self.v) * dt + c.sigma * np.sqrt(dt) * self.rng.standard_normal()
        self.v += dv
        return self.v

    def noisy_signal(self) -> float:
        return self.v + self.rng.normal(0, self.cfg.signal_noise_std)

    def reset(self):
        self.v = self.cfg.v_bar


# ════════════════════════════════════════════════════════════════
# §5  MULTIVARIATE HAWKES PROCESS
# ════════════════════════════════════════════════════════════════

class MultivariateHawkes:
    def __init__(self, cfg: HawkesConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.n = cfg.n_types
        self.G = np.zeros((self.n, self.n))
        self.t = 0.0

    def reset(self):
        self.G[:] = 0.0
        self.t = 0.0

    def intensities(self) -> np.ndarray:
        return self.cfg.mu + np.sum(self.cfg.alpha * self.G, axis=1)

    def _decay(self, dt: float):
        self.G *= np.exp(-self.cfg.beta * dt)

    def sample_window(self, dt: float) -> List[Tuple[float, int]]:
        events = []
        t_end = self.t + dt
        t_cur = self.t
        while t_cur < t_end:
            lam = self.intensities()
            lam_bar = lam.sum() * 1.3 + 1e-6
            u = self.rng.exponential(1.0 / lam_bar)
            t_cur += u
            if t_cur >= t_end:
                break
            self._decay(u)
            lam_new = self.intensities()
            total = lam_new.sum()
            if self.rng.uniform() < total / lam_bar:
                probs = lam_new / total
                etype = self.rng.choice(self.n, p=probs)
                events.append((t_cur, int(etype)))
                self.G[:, etype] += 1.0
        remaining = t_end - min(t_cur, t_end)
        if remaining > 0:
            self._decay(remaining)
        self.t = t_end
        return events


# ════════════════════════════════════════════════════════════════
# §6  NOISE TRADER (Hawkes → Orders)
# ════════════════════════════════════════════════════════════════

class HawkesNoiseTrader:
    AGENT_ID = -1

    def __init__(self, cfg: HawkesConfig, lob_cfg: LOBConfig,
                 rng: np.random.Generator):
        self.hawkes = MultivariateHawkes(cfg, rng)
        self.hcfg = cfg
        self.lcfg = lob_cfg
        self.rng = rng
        # FIX: track OIDs with timestamps for expiry
        self.active_oids: Dict[Side, List[Tuple[int, float]]] = {
            Side.BID: [], Side.ASK: []
        }

    def reset(self):
        self.hawkes.reset()
        self.active_oids = {Side.BID: [], Side.ASK: []}

    def _sample_size(self) -> int:
        s = self.rng.lognormal(self.hcfg.size_mu, self.hcfg.size_sigma)
        return max(1, min(int(s), self.lcfg.max_order_qty))

    def _sample_limit_offset(self) -> int:
        mx = self.hcfg.max_ticks_from_mid
        k = np.arange(1, mx + 1, dtype=float)
        w = k ** (-self.hcfg.price_placement_exp)
        w /= w.sum()
        return int(self.rng.choice(k, p=w))

    def _purge_stale_oids(self, lob: LimitOrderBook):
        """FIX: remove OIDs that are no longer live in the book."""
        for side in [Side.BID, Side.ASK]:
            self.active_oids[side] = [
                (oid, ts) for (oid, ts) in self.active_oids[side]
                if lob.order_exists(oid)
            ]

    def _expire_old_orders(self, lob: LimitOrderBook, current_time: float):
        """FIX: cancel orders older than max_age."""
        max_age = self.hcfg.max_order_age
        for side in [Side.BID, Side.ASK]:
            surviving = []
            for (oid, ts) in self.active_oids[side]:
                if (current_time - ts) > max_age:
                    lob.cancel(oid)
                else:
                    surviving.append((oid, ts))
            self.active_oids[side] = surviving

    def generate_orders(self, lob: LimitOrderBook, dt: float,
                        current_time: float = 0.0,
                        fundamental_price: float = None) -> List[Fill]:
        """
        FIX v3: accepts fundamental_price. A fraction of limit orders are
        anchored to the fundamental rather than mid, providing mean-reversion.
        """
        self._purge_stale_oids(lob)
        self._expire_old_orders(lob, current_time)

        events = self.hawkes.sample_window(dt)
        all_fills = []
        mid = lob.mid_price
        tick = self.lcfg.tick_size
        frac = self.hcfg.fundamental_anchor_frac

        for t_ev, etype in events:
            etype = EventType(etype)
            qty = self._sample_size()

            if etype == EventType.LIMIT_BUY:
                offset = self._sample_limit_offset()
                # FIX v3: some orders anchored to fundamental
                if fundamental_price is not None and self.rng.uniform() < frac:
                    ref = fundamental_price
                else:
                    ref = mid
                price = ref - offset * tick
                oid, fills = lob.submit_limit(self.AGENT_ID, Side.BID, price, qty, t_ev)
                if lob.order_exists(oid):
                    self.active_oids[Side.BID].append((oid, t_ev))
                all_fills.extend(fills)

            elif etype == EventType.LIMIT_SELL:
                offset = self._sample_limit_offset()
                if fundamental_price is not None and self.rng.uniform() < frac:
                    ref = fundamental_price
                else:
                    ref = mid
                price = ref + offset * tick
                oid, fills = lob.submit_limit(self.AGENT_ID, Side.ASK, price, qty, t_ev)
                if lob.order_exists(oid):
                    self.active_oids[Side.ASK].append((oid, t_ev))
                all_fills.extend(fills)

            elif etype == EventType.MKT_BUY:
                _, fills = lob.submit_market(self.AGENT_ID, Side.BID, qty, t_ev)
                all_fills.extend(fills)

            elif etype == EventType.MKT_SELL:
                _, fills = lob.submit_market(self.AGENT_ID, Side.ASK, qty, t_ev)
                all_fills.extend(fills)

            elif etype == EventType.CANCEL_BID:
                if self.active_oids[Side.BID]:
                    idx = self.rng.integers(len(self.active_oids[Side.BID]))
                    oid, _ = self.active_oids[Side.BID].pop(idx)
                    lob.cancel(oid)

            elif etype == EventType.CANCEL_ASK:
                if self.active_oids[Side.ASK]:
                    idx = self.rng.integers(len(self.active_oids[Side.ASK]))
                    oid, _ = self.active_oids[Side.ASK].pop(idx)
                    lob.cancel(oid)

        return all_fills


# ════════════════════════════════════════════════════════════════
# §7  AGENT PORTFOLIO TRACKER
# ════════════════════════════════════════════════════════════════

class Portfolio:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id
        self.inventory: int = 0
        self.cash: float = 0.0
        self.total_traded: int = 0
        self.total_fees: float = 0.0
        self._peak_equity: float = 0.0
        self.pnl_history: List[float] = []

    def process_fills(self, fills: List[Fill], mid: float):
        for f in fills:
            if f.aggressor_id == self.agent_id:
                sign = 1 if f.aggressor_side == Side.BID else -1
                self.inventory += sign * f.quantity
                self.cash -= sign * f.quantity * f.price
                self.total_traded += f.quantity
            elif f.passive_id == self.agent_id:
                # FIX: passive side is opposite of aggressor
                sign = -1 if f.aggressor_side == Side.BID else 1
                self.inventory += sign * f.quantity
                self.cash -= sign * f.quantity * f.price
                self.total_traded += f.quantity

    def equity(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def drawdown(self, mid: float) -> float:
        eq = self.equity(mid)
        self._peak_equity = max(self._peak_equity, eq)
        return max(0.0, self._peak_equity - eq)

    def reset(self):
        self.inventory = 0
        self.cash = 0.0
        self.total_traded = 0
        self.total_fees = 0.0
        self._peak_equity = 0.0
        self.pnl_history = []


# ════════════════════════════════════════════════════════════════
# §7b  FUNDAMENTAL TRADER (price anchoring)
# ════════════════════════════════════════════════════════════════

class FundamentalTrader:
    """
    FIX v3: Simple mean-reversion trader that acts when mid deviates
    from fundamental. Sends market orders to push price back.
    This is the Glosten-Milgrom informed-trader analogue that
    prevents unbounded price drift in the validation run.
    """
    AGENT_ID = -2

    def __init__(self, threshold: float = 0.5, intensity: float = 0.15,
                 max_qty: int = 2, rng: np.random.Generator = None):
        self.threshold = threshold   # min |mispricing| in % to trigger (raised from 0.15)
        self.intensity = intensity   # probability scaling (reduced from 0.3)
        self.max_qty = max_qty       # reduced from 3
        self.rng = rng or np.random.default_rng()
        self.portfolio = Portfolio(self.AGENT_ID)

    def act(self, lob: LimitOrderBook, fundamental: float,
            ts: float) -> List[Fill]:
        mid = lob.mid_price
        mispricing = fundamental - mid
        rel_mispricing = abs(mispricing) / (mid + 1e-10)

        # FIX v4: higher threshold so small deviations persist (creates stylised facts)
        # Only intervene when price is >0.5% away from fundamental
        if rel_mispricing < self.threshold / 100.0:
            return []

        # Probability scales quadratically with mispricing — gentle for small, aggressive for large
        prob = min(1.0, self.intensity * (rel_mispricing * 100) ** 2)
        if self.rng.uniform() > prob:
            return []

        qty = max(1, min(self.max_qty, int(1 + rel_mispricing * 30)))
        if mispricing > 0:
            _, fills = lob.submit_market(self.AGENT_ID, Side.BID, qty, ts)
        else:
            _, fills = lob.submit_market(self.AGENT_ID, Side.ASK, qty, ts)

        self.portfolio.process_fills(fills, lob.mid_price)
        return fills

    def reset(self):
        self.portfolio.reset()


# ════════════════════════════════════════════════════════════════
# §8  STATE BUILDER
# ════════════════════════════════════════════════════════════════

class StateBuilder:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.n_levels = min(cfg.lob.max_depth_levels, 10)
        self.return_buf: deque = deque(maxlen=cfg.obs_window)
        self.volume_buf: deque = deque(maxlen=cfg.obs_window)
        self.spread_buf: deque = deque(maxlen=cfg.obs_window)
        self.prev_mid: Optional[float] = None

    def update(self, lob: LimitOrderBook, step_volume: int):
        mid = lob.mid_price
        if self.prev_mid is not None and self.prev_mid > 0:
            ret = (mid - self.prev_mid) / self.prev_mid
        else:
            ret = 0.0
        self.return_buf.append(ret)
        self.volume_buf.append(step_volume)
        self.spread_buf.append(lob.spread if lob.spread < 1e6 else 0.0)
        self.prev_mid = mid

    def _common_features(self, lob: LimitOrderBook) -> np.ndarray:
        snap = lob.depth_snapshot(self.n_levels)
        mid = lob.mid_price
        ref = self.cfg.lob.initial_mid
        norm_mid = (mid - ref) / ref
        norm_spread = (lob.spread if lob.spread < 1e6 else 0.0) / ref
        bid_v = snap['bid_volumes'] / 100.0
        ask_v = snap['ask_volumes'] / 100.0
        tb, ta = bid_v.sum(), ask_v.sum()
        imb = (tb - ta) / (tb + ta + 1e-8)
        rets = np.zeros(self.cfg.obs_window)
        buf = list(self.return_buf)
        if buf:
            rets[-len(buf):] = buf
        vols = np.zeros(min(10, self.cfg.obs_window))
        vbuf = list(self.volume_buf)[-10:]
        if vbuf:
            vols[-len(vbuf):] = np.array(vbuf) / 50.0
        recent_r = list(self.return_buf)[-20:]
        rvol = np.std(recent_r) if len(recent_r) > 2 else 0.0
        return np.concatenate([
            [norm_mid, norm_spread, imb, rvol * 100],
            bid_v, ask_v, rets[-20:], vols])

    def mm_observation(self, lob: LimitOrderBook, portfolio: Portfolio) -> np.ndarray:
        common = self._common_features(lob)
        inv_norm = portfolio.inventory / self.cfg.agent.max_inventory
        eq = portfolio.equity(lob.mid_price) / self.cfg.lob.initial_mid
        dd = portfolio.drawdown(lob.mid_price) / self.cfg.lob.initial_mid
        return np.concatenate([common, [inv_norm, eq, dd]]).astype(np.float32)

    def it_observation(self, lob: LimitOrderBook, portfolio: Portfolio,
                       signal: float) -> np.ndarray:
        mm_obs = self.mm_observation(lob, portfolio)
        mid = lob.mid_price
        norm_signal = (signal - mid) / self.cfg.lob.initial_mid
        return np.concatenate([mm_obs, [norm_signal]]).astype(np.float32)

    @property
    def mm_obs_dim(self) -> int:
        return 4 + 2 * self.n_levels + 20 + 10 + 3

    @property
    def it_obs_dim(self) -> int:
        return self.mm_obs_dim + 1

    def reset(self):
        self.return_buf.clear()
        self.volume_buf.clear()
        self.spread_buf.clear()
        self.prev_mid = None


# ════════════════════════════════════════════════════════════════
# §9  REWARD FUNCTIONS
# ════════════════════════════════════════════════════════════════

class RewardComputer:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.prev_equity: Dict[int, float] = {}

    def compute(self, agent_id: int, portfolio: Portfolio,
                mid: float, step_fills: List[Fill],
                dd_penalty: float = 0.0005,
                dd_threshold: float = 50.0) -> float:
        eq = portfolio.equity(mid)
        prev = self.prev_equity.get(agent_id, eq)
        delta_pnl = eq - prev
        self.prev_equity[agent_id] = eq
        tc = sum(f.quantity * 0.005 for f in step_fills
                 if f.aggressor_id == agent_id or f.passive_id == agent_id)
        inv_pen = self.cfg.inventory_penalty * (portfolio.inventory ** 2)
        dd = portfolio.drawdown(mid)
        dd_pen = dd_penalty * max(0.0, dd - dd_threshold)
        return delta_pnl - inv_pen - self.cfg.tc_penalty * tc - dd_pen

    def reset(self):
        self.prev_equity.clear()


# ════════════════════════════════════════════════════════════════
# §10  AVELLANEDA-STOIKOV BASELINE MARKET MAKER
# ════════════════════════════════════════════════════════════════

class AvellanedaStoikovMM:
    """
    FIX: Parameters completely recalibrated.
    - kappa raised to 100 (dense book → tighter spreads)
    - sigma set to price-scale volatility (~0.5/sqrt(T))
    - T set to 100 (reasonable horizon)
    - gamma set to 0.3 (moderate risk aversion)
    - max_qty set to 3 (small quote sizes so they get filled)

    With these: spread ≈ γσ²T + (2/γ)ln(1+γ/κ) ≈ 0.3*0.25*100 + 6.67*ln(1.003)
                       ≈ 7.5 + 0.02 ≈ 7.52 — still too wide!

    Better approach: use the INFINITE HORIZON approximation:
        δ* = (2/γ) * ln(1 + γ/κ)   +   spread_floor
    and adjust reservation price by inventory:
        r = mid - q * γ * σ² * τ

    With γ=0.1, κ=100: δ* = 20*ln(1.001) = 0.02 per side → total spread ~0.04
    This is 4 ticks, which is realistic.
    """

    def __init__(self, agent_id: int, gamma: float = 0.1,
                 kappa: float = 100.0, sigma: float = 0.05,
                 max_qty: int = 3, T: float = 100.0):
        self.agent_id = agent_id
        self.gamma = gamma
        self.kappa = kappa
        self.sigma = sigma
        self.max_qty = max_qty
        self.T = T
        self.portfolio = Portfolio(agent_id)
        self.active_oids: List[int] = []
        # FIX v3: track recent mid-prices for adaptive σ
        self._recent_mids: deque = deque(maxlen=100)

    def act(self, lob: LimitOrderBook, t: float) -> List[Fill]:
        """Cancel old quotes, compute A-S quotes, submit. Returns only crossing fills."""
        for oid in self.active_oids:
            lob.cancel(oid)
        self.active_oids.clear()

        mid = lob.mid_price
        self._recent_mids.append(mid)

        q = self.portfolio.inventory
        tau = max(self.T - (t % self.T), 1.0)
        g, k = self.gamma, self.kappa

        # FIX v4: adaptive σ from realised volatility with meaningful floor
        if len(self._recent_mids) > 10:
            mids_arr = np.array(self._recent_mids)
            rets = np.diff(np.log(mids_arr + 1e-10))
            rets = rets[rets != 0]  # exclude zero returns for cleaner estimate
            if len(rets) > 3:
                s = max(np.std(rets) * np.sqrt(10), 0.005)
            else:
                s = self.sigma
        else:
            s = self.sigma
        # Floor: never let σ drop below initial estimate
        s = max(s, 0.005)

        # Reservation price (inventory-adjusted)
        r = mid - q * g * (s ** 2) * tau

        # Optimal half-spread
        half_spread = (1.0 / g) * np.log(1.0 + g / k) + 0.5 * g * (s ** 2) * tau

        # Floor: at least 1 tick on each side
        half_spread = max(half_spread, lob.cfg.tick_size)

        bid_price = r - half_spread
        ask_price = r + half_spread

        all_fills = []
        oid_b, fills_b = lob.submit_limit(
            self.agent_id, Side.BID, bid_price, self.max_qty, t)
        oid_a, fills_a = lob.submit_limit(
            self.agent_id, Side.ASK, ask_price, self.max_qty, t)
        self.active_oids = [oid_b, oid_a]
        all_fills.extend(fills_b + fills_a)
        return all_fills

    def process_external_fills(self, all_fills: List[Fill], mid: float):
        """FIX: process fills where this MM was the passive party."""
        my_fills = [f for f in all_fills
                    if f.aggressor_id == self.agent_id
                    or f.passive_id == self.agent_id]
        self.portfolio.process_fills(my_fills, mid)

    def reset(self):
        self.portfolio.reset()
        self.active_oids.clear()
        self._recent_mids.clear()


# ════════════════════════════════════════════════════════════════
# §11  GYMNASIUM ENVIRONMENT — MARKET MAKER
# ════════════════════════════════════════════════════════════════

class MarketMakerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig = None):
        super().__init__()
        self.cfg = cfg or SimConfig()
        c = self.cfg
        sb = StateBuilder(c)
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(sb.mm_obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            0.0, 1.0, shape=(4,), dtype=np.float32)
        self.rng: Optional[np.random.Generator] = None
        self._setup_done = False

    def _setup(self):
        c = self.cfg
        self.rng = np.random.default_rng(c.seed)
        self.lob = LimitOrderBook(c.lob)
        self.fundamental = FundamentalValue(c.fundamental, self.rng)
        self.noise_trader = HawkesNoiseTrader(c.hawkes, c.lob, self.rng)
        self.state_builder = StateBuilder(c)
        self.reward_computer = RewardComputer(c.agent)
        self.rl_agent_id = 0
        self.portfolio = Portfolio(self.rl_agent_id)
        self.rl_active_oids: List[int] = []
        self.as_mms = []
        for i in range(max(0, c.n_market_makers - 1)):
            self.as_mms.append(AvellanedaStoikovMM(
                agent_id=100 + i,
                gamma=0.1 + self.rng.uniform(-0.02, 0.02),
                kappa=100.0 + self.rng.uniform(-20, 20),
                sigma=0.05,
            ))
        # FIX v3: fundamental trader for price anchoring
        self.fund_trader = FundamentalTrader(
            threshold=0.5, intensity=0.15, max_qty=2, rng=self.rng)
        self._setup_done = True

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.cfg.seed = seed
        self._setup()
        c = self.cfg
        self.lob = LimitOrderBook(c.lob)
        self.lob.seed_book(c.lob.initial_mid, rng=self.rng)
        self.fundamental.reset()
        self.noise_trader.reset()
        self.state_builder.reset()
        self.reward_computer.reset()
        self.portfolio.reset()
        self.rl_active_oids.clear()
        for mm in self.as_mms:
            mm.reset()
        self.fund_trader.reset()
        self.step_count = 0
        self.t = 0.0
        for _ in range(20):
            self.noise_trader.generate_orders(self.lob, c.dt, self.t,
                                               fundamental_price=self.fundamental.v)
            self.fundamental.step(c.dt)
            self.state_builder.update(self.lob, 0)
            self.t += c.dt
        obs = self.state_builder.mm_observation(self.lob, self.portfolio)
        return obs, {}

    def step(self, action: np.ndarray):
        c = self.cfg
        ac = c.agent
        self.step_count += 1
        self.t += c.dt

        bid_offset = action[0] * ac.max_quote_offset
        ask_offset = action[1] * ac.max_quote_offset
        bid_qty = max(1, int(action[2] * ac.max_order_size))
        ask_qty = max(1, int(action[3] * ac.max_order_size))
        mid = self.lob.mid_price

        # Cancel RL agent's previous quotes
        for oid in self.rl_active_oids:
            self.lob.cancel(oid)
        self.rl_active_oids.clear()

        # Submit RL agent's new quotes
        all_fills = []
        oid_b, fb = self.lob.submit_limit(
            self.rl_agent_id, Side.BID, mid - bid_offset, bid_qty, self.t)
        oid_a, fa = self.lob.submit_limit(
            self.rl_agent_id, Side.ASK, mid + ask_offset, ask_qty, self.t)
        self.rl_active_oids = [oid_b, oid_a]
        all_fills.extend(fb + fa)

        # A-S market makers act (submit quotes)
        for mm in self.as_mms:
            f = mm.act(self.lob, self.t)
            all_fills.extend(f)

        # Noise traders (may hit MM resting orders!)
        noise_fills = self.noise_trader.generate_orders(
            self.lob, c.dt, self.t, fundamental_price=self.fundamental.v)
        all_fills.extend(noise_fills)

        # FIX v3: fundamental trader provides price anchoring
        ft_fills = self.fund_trader.act(self.lob, self.fundamental.v, self.t)
        all_fills.extend(ft_fills)

        # Fundamental value step
        self.fundamental.step(c.dt)

        # FIX: Process ALL fills for A-S market makers (including noise hits)
        for mm in self.as_mms:
            mm.process_external_fills(all_fills, self.lob.mid_price)

        # Process RL agent's fills
        rl_fills = [f for f in all_fills
                    if f.aggressor_id == self.rl_agent_id
                    or f.passive_id == self.rl_agent_id]
        self.portfolio.process_fills(rl_fills, self.lob.mid_price)

        # Clip inventory
        if abs(self.portfolio.inventory) > ac.max_inventory:
            excess = abs(self.portfolio.inventory) - ac.max_inventory
            side = Side.ASK if self.portfolio.inventory > 0 else Side.BID
            self.lob.submit_market(self.rl_agent_id, side, excess, self.t)

        # Update state
        step_vol = sum(f.quantity for f in all_fills)
        self.state_builder.update(self.lob, step_vol)

        reward = self.reward_computer.compute(
            self.rl_agent_id, self.portfolio, self.lob.mid_price, rl_fills)

        done = self.step_count >= c.episode_steps
        obs = self.state_builder.mm_observation(self.lob, self.portfolio)

        info = {
            'inventory': self.portfolio.inventory,
            'equity': self.portfolio.equity(self.lob.mid_price),
            'mid_price': self.lob.mid_price,
            'spread': self.lob.spread,
            'n_fills': len(rl_fills),
            'fundamental': self.fundamental.v,
        }
        return obs, reward, done, False, info


# ════════════════════════════════════════════════════════════════
# §12  GYMNASIUM ENVIRONMENT — INFORMED TRADER
# ════════════════════════════════════════════════════════════════

class InformedTraderEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg: SimConfig = None):
        super().__init__()
        self.cfg = cfg or SimConfig()
        c = self.cfg
        sb = StateBuilder(c)
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(sb.it_obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            np.array([-1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]), dtype=np.float32)
        self.rng: Optional[np.random.Generator] = None

    def reset(self, seed=None, options=None):
        c = self.cfg
        self.rng = np.random.default_rng(seed or c.seed)
        self.lob = LimitOrderBook(c.lob)
        self.lob.seed_book(c.lob.initial_mid, rng=self.rng)
        self.fundamental = FundamentalValue(c.fundamental, self.rng)
        self.noise_trader = HawkesNoiseTrader(c.hawkes, c.lob, self.rng)
        self.state_builder = StateBuilder(c)
        self.reward_computer = RewardComputer(c.agent)
        self.rl_agent_id = 1
        self.portfolio = Portfolio(self.rl_agent_id)
        self.as_mms = [
            AvellanedaStoikovMM(
                agent_id=100+i,
                gamma=0.1 + self.rng.uniform(-0.02, 0.02),
                kappa=100.0 + self.rng.uniform(-20, 20),
                sigma=0.05,
            ) for i in range(c.n_market_makers)]
        self.fund_trader = FundamentalTrader(
            threshold=0.5, intensity=0.15, max_qty=2, rng=self.rng)
        self.step_count = 0
        self.t = 0.0
        for _ in range(20):
            self.noise_trader.generate_orders(self.lob, c.dt, self.t,
                                               fundamental_price=self.fundamental.v)
            self.fundamental.step(c.dt)
            self.state_builder.update(self.lob, 0)
            self.t += c.dt
        signal = self.fundamental.noisy_signal()
        obs = self.state_builder.it_observation(self.lob, self.portfolio, signal)
        return obs, {}

    def step(self, action: np.ndarray):
        c = self.cfg
        ac = c.agent
        self.step_count += 1
        self.t += c.dt

        direction = action[0]
        aggression = action[1]
        size_frac = action[2]
        qty = max(1, int(size_frac * ac.max_order_size))
        side = Side.BID if direction > 0 else Side.ASK

        all_fills = []
        if abs(direction) > 0.1:
            if aggression > 0.7:
                _, fills = self.lob.submit_market(self.rl_agent_id, side, qty, self.t)
                all_fills.extend(fills)
            else:
                mid = self.lob.mid_price
                offset = (1.0 - aggression) * ac.max_quote_offset
                price = mid - offset if side == Side.BID else mid + offset
                _, fills = self.lob.submit_limit(self.rl_agent_id, side, price, qty, self.t)
                all_fills.extend(fills)

        for mm in self.as_mms:
            f = mm.act(self.lob, self.t)
            all_fills.extend(f)

        noise_fills = self.noise_trader.generate_orders(
            self.lob, c.dt, self.t, fundamental_price=self.fundamental.v)
        all_fills.extend(noise_fills)

        ft_fills = self.fund_trader.act(self.lob, self.fundamental.v, self.t)
        all_fills.extend(ft_fills)

        self.fundamental.step(c.dt)

        # FIX: process fills for MMs
        for mm in self.as_mms:
            mm.process_external_fills(all_fills, self.lob.mid_price)

        rl_fills = [f for f in all_fills
                    if f.aggressor_id == self.rl_agent_id
                    or f.passive_id == self.rl_agent_id]
        self.portfolio.process_fills(rl_fills, self.lob.mid_price)

        if abs(self.portfolio.inventory) > ac.max_inventory:
            excess = abs(self.portfolio.inventory) - ac.max_inventory
            clip_side = Side.ASK if self.portfolio.inventory > 0 else Side.BID
            self.lob.submit_market(self.rl_agent_id, clip_side, excess, self.t)

        step_vol = sum(f.quantity for f in all_fills)
        self.state_builder.update(self.lob, step_vol)

        reward = self.reward_computer.compute(
            self.rl_agent_id, self.portfolio, self.lob.mid_price, rl_fills)

        done = self.step_count >= c.episode_steps
        signal = self.fundamental.noisy_signal()
        obs = self.state_builder.it_observation(self.lob, self.portfolio, signal)

        info = {
            'inventory': self.portfolio.inventory,
            'equity': self.portfolio.equity(self.lob.mid_price),
            'mid_price': self.lob.mid_price,
            'fundamental': self.fundamental.v,
            'signal': signal,
            'mispricing': self.fundamental.v - self.lob.mid_price,
            'n_fills': len(rl_fills),
        }
        return obs, reward, done, False, info


# ════════════════════════════════════════════════════════════════
# §13  ENVIRONMENT VALIDATION
# ════════════════════════════════════════════════════════════════

class EnvironmentValidator:
    def __init__(self, cfg: SimConfig = None, n_steps: int = 20000):
        self.cfg = cfg or SimConfig()
        self.n_steps = n_steps

    def collect_data(self) -> Dict[str, np.ndarray]:
        c = self.cfg
        rng = np.random.default_rng(c.seed)
        lob = LimitOrderBook(c.lob)
        lob.seed_book(c.lob.initial_mid, rng=rng)
        fundamental = FundamentalValue(c.fundamental, rng)
        noise = HawkesNoiseTrader(c.hawkes, c.lob, rng)
        mms = [AvellanedaStoikovMM(agent_id=100+i, sigma=0.05,
                                    kappa=100.0, gamma=0.1)
               for i in range(c.n_market_makers)]
        # FIX v3: add fundamental trader for price anchoring
        fund_trader = FundamentalTrader(threshold=0.5, intensity=0.15,
                                         max_qty=2, rng=rng)

        mids, spreads, volumes, fundamentals = [], [], [], []
        t = 0.0
        for step in range(self.n_steps):
            t += c.dt
            for mm in mms:
                mm.act(lob, t)
            # FIX v3: pass fundamental_price for anchored limit orders
            fills = noise.generate_orders(lob, c.dt, t,
                                          fundamental_price=fundamental.v)
            # FIX v3: fundamental trader acts
            ft_fills = fund_trader.act(lob, fundamental.v, t)
            fills.extend(ft_fills)

            fundamental.step(c.dt)
            for mm in mms:
                mm.process_external_fills(fills, lob.mid_price)

            mids.append(lob.mid_price)
            s = lob.spread
            spreads.append(s if s < 1e6 else np.nan)
            volumes.append(sum(f.quantity for f in fills))
            fundamentals.append(fundamental.v)

        return {
            'mid_prices': np.array(mids),
            'spreads': np.array(spreads),
            'volumes': np.array(volumes),
            'fundamentals': np.array(fundamentals),
        }

    def validate(self, verbose: bool = True) -> Dict[str, bool]:
        data = self.collect_data()
        prices = data['mid_prices']
        vols = data['volumes']
        spreads = data['spreads']
        returns = np.diff(np.log(prices + 1e-10))
        returns = returns[np.isfinite(returns)]
        results = {}

        from scipy import stats as sp_stats

        kurt = sp_stats.kurtosis(returns, fisher=True)
        results['fat_tails'] = kurt > 0
        if verbose:
            print(f"[Fat tails]  Excess kurtosis = {kurt:.2f}  "
                  f"{'✓' if results['fat_tails'] else '✗'}")

        abs_r = np.abs(returns)
        if len(abs_r) > 50:
            acf_10 = np.corrcoef(abs_r[10:], abs_r[:-10])[0, 1]
        else:
            acf_10 = 0
        results['vol_clustering'] = acf_10 > 0.02
        if verbose:
            print(f"[Vol cluster] ACF(|r|, lag=10) = {acf_10:.4f}  "
                  f"{'✓' if results['vol_clustering'] else '✗'}")

        if len(returns) > 2:
            acf_1 = np.corrcoef(returns[1:], returns[:-1])[0, 1]
        else:
            acf_1 = 0
        results['neg_autocorr'] = acf_1 < 0
        if verbose:
            print(f"[Neg ACF(1)] ACF(r, lag=1) = {acf_1:.4f}  "
                  f"{'✓' if results['neg_autocorr'] else '✗'} (<0 = bid-ask bounce)")

        clean_spreads = spreads[np.isfinite(spreads)]
        adf_stat = 0.99
        if len(clean_spreads) > 100:
            adf_stat = np.corrcoef(clean_spreads[1:], clean_spreads[:-1])[0, 1]
            results['mean_rev_spread'] = adf_stat < 0.99
        else:
            results['mean_rev_spread'] = False
        if verbose:
            print(f"[Spread MR]  Spread AR(1) = {adf_stat:.4f}  "
                  f"{'✓' if results['mean_rev_spread'] else '✗'}")

        if len(vols) > 20:
            vol_acf = np.corrcoef(vols[5:], vols[:-5])[0, 1]
        else:
            vol_acf = 0
        results['vol_cluster_volume'] = vol_acf > 0.01
        if verbose:
            print(f"[Vol ACF]    ACF(volume, lag=5) = {vol_acf:.4f}  "
                  f"{'✓' if results['vol_cluster_volume'] else '✗'}")

        passed = sum(results.values())
        total = len(results)
        if verbose:
            print(f"\n{'='*40}")
            print(f"Passed {passed}/{total} stylised fact checks")
        return results


# ════════════════════════════════════════════════════════════════
# §14  TRAINING SCRIPTS
# ════════════════════════════════════════════════════════════════

def train_market_maker(total_timesteps: int = 200_000):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import EvalCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("pip install stable-baselines3"); return
    cfg = SimConfig(episode_steps=2000, seed=42)
    env = Monitor(MarketMakerEnv(cfg))
    eval_env = Monitor(MarketMakerEnv(SimConfig(episode_steps=2000, seed=99)))
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, verbose=1,
                tensorboard_log="./tb_logs/mm_ppo")
    eval_cb = EvalCallback(eval_env, n_eval_episodes=5, eval_freq=10000,
                           best_model_save_path="./models/mm_best")
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)
    model.save("./models/mm_final")
    print("Training complete."); return model


def train_informed_trader(total_timesteps: int = 200_000):
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("pip install stable-baselines3"); return
    cfg = SimConfig(episode_steps=2000, seed=42)
    env = Monitor(InformedTraderEnv(cfg))
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01, verbose=1,
                tensorboard_log="./tb_logs/it_ppo")
    model.learn(total_timesteps=total_timesteps)
    model.save("./models/it_final")
    print("Training complete."); return model


# ════════════════════════════════════════════════════════════════
# §15  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent LOB Environment — Phase 1 (Bug-Fixed)")
    print("=" * 60)

    print("\n[1] Running environment validation...\n")
    validator = EnvironmentValidator(n_steps=10000)
    results = validator.validate(verbose=True)

    print("\n[2] Smoke-testing MarketMakerEnv...")
    mm_env = MarketMakerEnv(SimConfig(episode_steps=200, seed=0))
    obs, info = mm_env.reset()
    print(f"    Obs shape: {obs.shape}")
    total_r = 0.0
    for i in range(200):
        a = mm_env.action_space.sample()
        obs, r, done, trunc, info = mm_env.step(a)
        total_r += r
        if done: break
    print(f"    Steps: {i+1}, Reward: {total_r:.4f}, "
          f"Inv: {info['inventory']}, Spread: {info['spread']:.4f}")

    # Diagnostic: check A-S MM got fills
    for j, mm in enumerate(mm_env.as_mms):
        print(f"    A-S MM {j}: inv={mm.portfolio.inventory}, "
              f"traded={mm.portfolio.total_traded}, "
              f"equity={mm.portfolio.equity(info['mid_price']):.4f}")

    # Diagnostic: check book depth isn't exploding
    bv, av = mm_env.lob.total_depth()
    print(f"    Book depth: bid_vol={bv}, ask_vol={av}")

    print("\n[3] Smoke-testing InformedTraderEnv...")
    it_env = InformedTraderEnv(SimConfig(episode_steps=200, seed=0))
    obs, info = it_env.reset()
    total_r = 0.0
    for i in range(200):
        a = it_env.action_space.sample()
        obs, r, done, trunc, info = it_env.step(a)
        total_r += r
        if done: break
    print(f"    Steps: {i+1}, Reward: {total_r:.4f}, "
          f"Mispricing: {info['mispricing']:.4f}")

    print("\n[4] To train: model = train_market_maker(200000)")

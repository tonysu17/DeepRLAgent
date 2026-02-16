"""
Phase 2: Deep RL Trading Agent
==============================
Transformer-PPO with DeepStack-inspired value network for
position sizing in a multi-agent limit order book.

Architecture:
  [State Sequence] → Transformer Encoder → Latent z_t
       z_t → Policy Head (Actor): Gaussian π(a|s)
       z_t → Value Head (Critic): V(s) — "intuition" à la DeepStack

Dependencies: torch, numpy, gymnasium
    pip install torch numpy gymnasium

Usage:
    from phase2_agent import TransformerPPOAgent, train_agent
    agent = TransformerPPOAgent(obs_dim=57, act_dim=4)
    train_agent(total_steps=500_000)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from collections import deque
import time, os

# ════════════════════════════════════════════════════════════════
# §1  CONFIGURATION
# ════════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    """Hyperparameters for the Transformer-PPO agent."""
    # Architecture
    obs_dim: int = 57             # from StateBuilder.mm_obs_dim
    act_dim: int = 4              # [bid_offset, ask_offset, bid_qty, ask_qty]
    seq_len: int = 64             # lookback window for transformer
    d_model: int = 128            # transformer hidden dimension
    n_heads: int = 4              # attention heads
    n_layers: int = 3             # transformer layers
    d_ff: int = 256               # feedforward dimension
    dropout: float = 0.1
    # Policy
    log_std_init: float = -0.5    # initial log std for Gaussian policy
    log_std_min: float = -2.0
    log_std_max: float = 0.5
    # PPO
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # value function clipping
    ent_coef: float = 0.01        # entropy bonus
    vf_coef: float = 0.5          # value loss weight
    max_grad_norm: float = 0.5
    # Training
    n_steps: int = 2048           # steps per rollout
    batch_size: int = 64
    n_epochs: int = 10            # PPO epochs per rollout
    n_envs: int = 1               # parallel environments
    target_kl: Optional[float] = 0.03  # early stopping on KL
    # Reward shaping
    use_differential_sharpe: bool = True
    sharpe_eta: float = 0.01      # EMA decay for differential Sharpe
    # Device
    device: str = "auto"

    def get_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


# ════════════════════════════════════════════════════════════════
# §2  POSITIONAL ENCODING
# ════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """Learned positional embeddings for the sequence dimension."""
    def __init__(self, seq_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


# ════════════════════════════════════════════════════════════════
# §3  TRANSFORMER STATE ENCODER
# ════════════════════════════════════════════════════════════════

class TransformerStateEncoder(nn.Module):
    """
    Processes a sequence of observation vectors through a causal
    Transformer to produce a latent embedding z_t.

    This is the perceptual backbone: it learns which historical
    events (large trades, spread changes, inventory shifts) are
    most informative for current decisions — analogous to how a
    human trader's "intuition" implicitly weighs past events.
    """
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg

        # Project raw obs to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(cfg.obs_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
            nn.GELU(),
        )

        # Positional encoding
        self.pos_enc = LearnedPositionalEncoding(cfg.seq_len, cfg.d_model)

        # Causal Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-LN for stability (Parisotto et al., 2020)
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=cfg.n_layers,
            enable_nested_tensor=False,
        )

        # Causal mask (upper triangular)
        self.register_buffer(
            'causal_mask',
            torch.triu(torch.ones(cfg.seq_len, cfg.seq_len), diagonal=1).bool()
        )

        # Output projection with gating (GTrXL-inspired)
        self.gate = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_seq: (batch, seq_len, obs_dim)
        Returns:
            z_t: (batch, d_model) — latent embedding at final timestep
        """
        B, T, _ = obs_seq.shape

        # Project and add positional encoding
        h = self.input_proj(obs_seq)      # (B, T, d_model)
        h = self.pos_enc(h)

        # Causal mask for current sequence length
        mask = self.causal_mask[:T, :T]

        # Transformer forward
        h = self.transformer(h, mask=mask)  # (B, T, d_model)

        # Extract final timestep with gating
        h_last = h[:, -1, :]               # (B, d_model)
        gate = self.gate(h_last)
        z = gate * self.out_proj(h_last)    # (B, d_model)

        return z


# ════════════════════════════════════════════════════════════════
# §4  POLICY HEAD (ACTOR)
# ════════════════════════════════════════════════════════════════

class PolicyHead(nn.Module):
    """
    Maps latent embedding to a squashed Gaussian policy.
    Outputs continuous actions in [0, 1] for the market maker:
      [bid_offset, ask_offset, bid_qty_frac, ask_qty_frac]
    """
    def __init__(self, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg

        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.act_dim),
        )

        # Learnable log standard deviation
        self.log_std = nn.Parameter(
            torch.ones(cfg.act_dim) * cfg.log_std_init
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std) for the Gaussian policy."""
        mu = self.net(z)  # (B, act_dim)
        log_std = self.log_std.expand_as(mu)
        log_std = torch.clamp(log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        return mu, log_std

    def get_distribution(self, z: torch.Tensor) -> Normal:
        mu, log_std = self.forward(z)
        return Normal(mu, log_std.exp())

    def sample(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability."""
        dist = self.get_distribution(z)
        raw = dist.rsample()
        # Squash to [0, 1] via sigmoid
        action = torch.sigmoid(raw)
        # Log prob with Jacobian correction for sigmoid squashing
        log_prob = dist.log_prob(raw) - torch.log(action * (1 - action) + 1e-8)
        log_prob = log_prob.sum(dim=-1)  # (B,)
        return action, log_prob

    def log_prob(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of a given action."""
        dist = self.get_distribution(z)
        # Inverse sigmoid to recover raw action
        raw = torch.log(action / (1 - action + 1e-8) + 1e-8)
        lp = dist.log_prob(raw) - torch.log(action * (1 - action) + 1e-8)
        return lp.sum(dim=-1)

    def entropy(self, z: torch.Tensor) -> torch.Tensor:
        dist = self.get_distribution(z)
        return dist.entropy().sum(dim=-1)


# ════════════════════════════════════════════════════════════════
# §5  VALUE HEAD (CRITIC) — DeepStack-Inspired "Intuition"
# ════════════════════════════════════════════════════════════════

class ValueHead(nn.Module):
    """
    DeepStack-inspired value network.

    In DeepStack, the value network estimates counterfactual values
    at information set boundaries without full game-tree traversal.
    Here, V(s) estimates expected risk-adjusted future payoffs from
    the current (partially observed) market state.

    The value head captures "intuition" about:
      - Adverse selection risk embedded in the current LOB state
      - Likely future price trajectory given recent order flow
      - Cost of current inventory position
      - Opportunity cost of inaction

    This is the architectural analogue of DeepStack's deep
    counterfactual value network: it compresses complex game-theoretic
    reasoning into a single scalar estimate, enabling real-time
    decision-making under imperfect information.
    """
    def __init__(self, cfg: AgentConfig):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model // 4),
            nn.GELU(),
            nn.Linear(cfg.d_model // 4, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns V(s) scalar estimate. Shape: (B,)"""
        return self.net(z).squeeze(-1)


# ════════════════════════════════════════════════════════════════
# §6  FULL AGENT: TRANSFORMER-PPO
# ════════════════════════════════════════════════════════════════

class TransformerPPOAgent(nn.Module):
    """
    Complete Transformer-PPO agent combining:
      - TransformerStateEncoder: sequential state processing
      - PolicyHead: continuous Gaussian actor
      - ValueHead: DeepStack-inspired critic

    Architecture diagram:
        [obs_t-T, ..., obs_t] → Transformer → z_t
                                                ├→ PolicyHead → π(a|s)
                                                └→ ValueHead  → V(s)
    """
    def __init__(self, cfg: AgentConfig = None):
        super().__init__()
        self.cfg = cfg or AgentConfig()
        self.encoder = TransformerStateEncoder(self.cfg)
        self.actor = PolicyHead(self.cfg)
        self.critic = ValueHead(self.cfg)
        self.device = self.cfg.get_device()
        self.to(self.device)

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[TransformerPPO] {trainable:,} trainable params "
              f"({total:,} total) on {self.device}")

    def encode(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Encode observation sequence to latent."""
        return self.encoder(obs_seq)

    def get_action_and_value(self, obs_seq: torch.Tensor,
                              deterministic: bool = False):
        """
        Forward pass for rollout collection.
        Returns: action, log_prob, entropy, value
        """
        z = self.encode(obs_seq)
        if deterministic:
            mu, _ = self.actor(z)
            action = torch.sigmoid(mu)
            log_prob = self.actor.log_prob(z, action)
        else:
            action, log_prob = self.actor.sample(z)
        entropy = self.actor.entropy(z)
        value = self.critic(z)
        return action, log_prob, entropy, value

    def evaluate_actions(self, obs_seq: torch.Tensor,
                          actions: torch.Tensor):
        """
        Forward pass for PPO update.
        Returns: log_prob, entropy, value
        """
        z = self.encode(obs_seq)
        log_prob = self.actor.log_prob(z, actions)
        entropy = self.actor.entropy(z)
        value = self.critic(z)
        return log_prob, entropy, value

    def get_value(self, obs_seq: torch.Tensor) -> torch.Tensor:
        z = self.encode(obs_seq)
        return self.critic(z)


# ════════════════════════════════════════════════════════════════
# §7  OBSERVATION SEQUENCE BUFFER
# ════════════════════════════════════════════════════════════════

class ObsSequenceBuffer:
    """
    Maintains a rolling window of observations for each environment,
    producing the (batch, seq_len, obs_dim) tensor the Transformer needs.
    Zero-padded for the initial steps before the window is full.
    """
    def __init__(self, n_envs: int, seq_len: int, obs_dim: int):
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.buffers = [deque(maxlen=seq_len) for _ in range(n_envs)]

    def reset(self, env_idx: int = None):
        if env_idx is not None:
            self.buffers[env_idx].clear()
        else:
            for b in self.buffers:
                b.clear()

    def push(self, obs: np.ndarray, env_idx: int = 0):
        self.buffers[env_idx].append(obs.copy())

    def get_sequence(self, env_idx: int = 0) -> np.ndarray:
        """Returns (seq_len, obs_dim) array, zero-padded if needed."""
        buf = list(self.buffers[env_idx])
        n = len(buf)
        seq = np.zeros((self.seq_len, self.obs_dim), dtype=np.float32)
        if n > 0:
            arr = np.array(buf, dtype=np.float32)
            seq[-n:] = arr
        return seq

    def get_batch(self) -> np.ndarray:
        """Returns (n_envs, seq_len, obs_dim)."""
        return np.stack([self.get_sequence(i) for i in range(len(self.buffers))])


# ════════════════════════════════════════════════════════════════
# §8  REWARD ENGINEERING
# ════════════════════════════════════════════════════════════════

class DifferentialSharpeReward:
    """
    Moody & Saffell (2001) differential Sharpe ratio.

    D_t = (B_{t-1} ΔA_t - 0.5 A_{t-1} ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}

    where A_t, B_t are EMAs of returns and squared returns.
    This directly optimises the Sharpe ratio online.
    """
    def __init__(self, eta: float = 0.01):
        self.eta = eta
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns

    def update(self, r_t: float) -> float:
        """Compute differential Sharpe and update EMAs."""
        dA = r_t - self.A
        dB = r_t ** 2 - self.B

        denom = (self.B - self.A ** 2) ** 1.5
        if abs(denom) < 1e-12:
            # Early steps: use raw return
            D = r_t
        else:
            D = (self.B * dA - 0.5 * self.A * dB) / denom

        self.A += self.eta * dA
        self.B += self.eta * dB
        return D

    def reset(self):
        self.A = 0.0
        self.B = 0.0


class RewardShaper:
    """
    Composite reward combining multiple objectives.

    R_t = w_pnl · ΔPnL
          - w_inv · I_t²              (inventory penalty)
          - w_tc  · TC_t              (transaction costs)
          - w_dd  · max(0, DD-DD*)    (drawdown penalty)
          + w_sharpe · D_t            (differential Sharpe bonus)
    """
    def __init__(self, w_pnl: float = 1.0, w_inv: float = 0.001,
                 w_tc: float = 0.5, w_dd: float = 0.0005,
                 dd_threshold: float = 50.0,
                 w_sharpe: float = 0.5, sharpe_eta: float = 0.01):
        self.w_pnl = w_pnl
        self.w_inv = w_inv
        self.w_tc = w_tc
        self.w_dd = w_dd
        self.dd_threshold = dd_threshold
        self.w_sharpe = w_sharpe
        self.diff_sharpe = DifferentialSharpeReward(eta=sharpe_eta)
        self.prev_equity = None

    def compute(self, equity: float, inventory: int,
                n_fills: int, spread: float) -> float:
        if self.prev_equity is None:
            self.prev_equity = equity
            return 0.0

        delta_pnl = equity - self.prev_equity
        self.prev_equity = equity

        # Components
        inv_pen = self.w_inv * (inventory ** 2)
        tc = self.w_tc * n_fills * 0.005
        dd_pen = 0.0  # simplified for now

        # Differential Sharpe
        d_sharpe = self.diff_sharpe.update(delta_pnl)
        sharpe_bonus = self.w_sharpe * d_sharpe

        return self.w_pnl * delta_pnl - inv_pen - tc - dd_pen + sharpe_bonus

    def reset(self):
        self.prev_equity = None
        self.diff_sharpe.reset()


# ════════════════════════════════════════════════════════════════
# §9  ROLLOUT BUFFER
# ════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Stores rollout data for PPO updates."""
    def __init__(self, n_steps: int, seq_len: int, obs_dim: int,
                 act_dim: int, gamma: float, gae_lambda: float,
                 device: torch.device):
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.ptr = 0

        # Storage
        self.obs_seqs = np.zeros((n_steps, seq_len, obs_dim), dtype=np.float32)
        self.actions = np.zeros((n_steps, act_dim), dtype=np.float32)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

    def add(self, obs_seq, action, reward, value, log_prob, done):
        i = self.ptr
        self.obs_seqs[i] = obs_seq
        self.actions[i] = action
        self.rewards[i] = reward
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_gae(self, last_value: float):
        """Generalised Advantage Estimation (Schulman et al., 2015)."""
        gae = 0.0
        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_val = last_value
                next_done = 0.0
            else:
                next_val = self.values[t + 1]
                next_done = self.dones[t + 1]
            delta = (self.rewards[t] + self.gamma * next_val * (1 - next_done)
                     - self.values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            self.advantages[t] = gae
        self.returns[:self.ptr] = self.advantages[:self.ptr] + self.values[:self.ptr]

    def get_batches(self, batch_size: int):
        """Yield random mini-batches for PPO update."""
        n = self.ptr
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = indices[start:end]
            yield {
                'obs_seqs': torch.FloatTensor(self.obs_seqs[idx]).to(self.device),
                'actions': torch.FloatTensor(self.actions[idx]).to(self.device),
                'old_log_probs': torch.FloatTensor(self.log_probs[idx]).to(self.device),
                'advantages': torch.FloatTensor(self.advantages[idx]).to(self.device),
                'returns': torch.FloatTensor(self.returns[idx]).to(self.device),
            }

    def reset(self):
        self.ptr = 0


# ════════════════════════════════════════════════════════════════
# §10  PPO TRAINER
# ════════════════════════════════════════════════════════════════

class PPOTrainer:
    """
    Proximal Policy Optimisation (Schulman et al., 2017) trainer
    for the Transformer-PPO agent.
    """
    def __init__(self, agent: TransformerPPOAgent, cfg: AgentConfig = None):
        self.agent = agent
        self.cfg = cfg or agent.cfg
        self.device = self.cfg.get_device()

        self.optimizer = torch.optim.Adam(
            agent.parameters(), lr=self.cfg.lr, eps=1e-5)

        # Learning rate scheduler
        self.scheduler = None  # set in train()

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run PPO update on collected rollout data."""
        cfg = self.cfg

        # Normalise advantages
        adv = buffer.advantages[:buffer.ptr]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buffer.advantages[:buffer.ptr] = adv

        metrics = {'policy_loss': 0, 'value_loss': 0, 'entropy': 0,
                   'approx_kl': 0, 'clip_fraction': 0}
        n_updates = 0

        for epoch in range(cfg.n_epochs):
            for batch in buffer.get_batches(cfg.batch_size):
                obs_seqs = batch['obs_seqs']
                actions = batch['actions']
                old_lp = batch['old_log_probs']
                advantages = batch['advantages']
                returns = batch['returns']

                # Forward pass
                new_lp, entropy, values = self.agent.evaluate_actions(
                    obs_seqs, actions)

                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_range,
                                    1 + cfg.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if cfg.clip_range_vf is not None:
                    # Clipped value loss
                    old_v = batch.get('old_values', values.detach())
                    v_clipped = old_v + torch.clamp(
                        values - old_v, -cfg.clip_range_vf, cfg.clip_range_vf)
                    vf_loss1 = (values - returns) ** 2
                    vf_loss2 = (v_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (policy_loss
                        + cfg.vf_coef * value_loss
                        + cfg.ent_coef * entropy_loss)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(),
                                         cfg.max_grad_norm)
                self.optimizer.step()

                # Metrics
                with torch.no_grad():
                    approx_kl = (old_lp - new_lp).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.mean().item()
                metrics['approx_kl'] += approx_kl
                metrics['clip_fraction'] += clip_frac
                n_updates += 1

            # Early stopping on KL divergence
            if cfg.target_kl and metrics['approx_kl'] / max(n_updates, 1) > cfg.target_kl:
                break

        # Average metrics
        for k in metrics:
            metrics[k] /= max(n_updates, 1)
        return metrics


# ════════════════════════════════════════════════════════════════
# §11  TRAINING LOOP
# ════════════════════════════════════════════════════════════════

def train_agent(
    total_steps: int = 500_000,
    agent_cfg: AgentConfig = None,
    env_cfg=None,
    save_dir: str = "./models/transformer_ppo",
    log_interval: int = 5,
):
    """
    Main training loop implementing the staged training protocol.

    Stage 1: Train against fixed A-S market makers + Hawkes noise
    (Stages 2-3 for multi-agent co-training to be added later)
    """
    # Lazy import to avoid circular dependency
    from lob_environment import SimConfig, MarketMakerEnv

    # Config
    if env_cfg is None:
        env_cfg = SimConfig(episode_steps=2000, seed=42)
    if agent_cfg is None:
        agent_cfg = AgentConfig(obs_dim=57, act_dim=4)

    os.makedirs(save_dir, exist_ok=True)
    device = agent_cfg.get_device()

    # Environment
    env = MarketMakerEnv(env_cfg)
    obs, _ = env.reset()

    # Agent
    agent = TransformerPPOAgent(agent_cfg)
    trainer = PPOTrainer(agent, agent_cfg)

    # Observation buffer
    obs_buffer = ObsSequenceBuffer(
        n_envs=1, seq_len=agent_cfg.seq_len, obs_dim=agent_cfg.obs_dim)
    obs_buffer.push(obs, 0)

    # Rollout buffer
    rollout = RolloutBuffer(
        n_steps=agent_cfg.n_steps,
        seq_len=agent_cfg.seq_len,
        obs_dim=agent_cfg.obs_dim,
        act_dim=agent_cfg.act_dim,
        gamma=agent_cfg.gamma,
        gae_lambda=agent_cfg.gae_lambda,
        device=device,
    )

    # Reward shaper
    reward_shaper = RewardShaper(
        w_sharpe=0.5 if agent_cfg.use_differential_sharpe else 0.0,
        sharpe_eta=agent_cfg.sharpe_eta,
    )

    # Tracking
    global_step = 0
    n_rollouts = 0
    episode_rewards = []
    ep_reward = 0.0
    ep_len = 0
    best_mean_reward = -float('inf')

    print(f"\n{'='*60}")
    print(f"  Phase 2: Transformer-PPO Training")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Device: {device}")
    print(f"  Seq length: {agent_cfg.seq_len}")
    print(f"  Architecture: {agent_cfg.n_layers}L, {agent_cfg.n_heads}H, "
          f"d={agent_cfg.d_model}")
    print(f"{'='*60}\n")

    t_start = time.time()

    while global_step < total_steps:
        rollout.reset()

        # ── Collect rollout ──────────────────────────────────
        agent.eval()
        for step in range(agent_cfg.n_steps):
            global_step += 1

            # Get observation sequence
            obs_seq = obs_buffer.get_sequence(0)
            obs_seq_t = torch.FloatTensor(obs_seq).unsqueeze(0).to(device)

            # Get action from policy
            with torch.no_grad():
                action, log_prob, entropy, value = agent.get_action_and_value(obs_seq_t)

            action_np = action.cpu().numpy().squeeze(0)
            log_prob_np = log_prob.cpu().item()
            value_np = value.cpu().item()

            # Step environment
            next_obs, raw_reward, done, truncated, info = env.step(action_np)

            # Shape reward
            shaped_reward = reward_shaper.compute(
                equity=info['equity'],
                inventory=info['inventory'],
                n_fills=info['n_fills'],
                spread=info['spread'],
            )

            # Store transition
            rollout.add(obs_seq, action_np, shaped_reward,
                        value_np, log_prob_np, done)

            ep_reward += shaped_reward
            ep_len += 1

            # Update observation buffer
            obs_buffer.push(next_obs, 0)

            if done or truncated:
                episode_rewards.append(ep_reward)
                ep_reward = 0.0
                ep_len = 0
                obs, _ = env.reset()
                obs_buffer.reset(0)
                obs_buffer.push(obs, 0)
                reward_shaper.reset()

        # ── Compute GAE ──────────────────────────────────────
        with torch.no_grad():
            last_obs_seq = torch.FloatTensor(
                obs_buffer.get_sequence(0)).unsqueeze(0).to(device)
            last_value = agent.get_value(last_obs_seq).cpu().item()
        rollout.compute_gae(last_value)

        # ── PPO Update ───────────────────────────────────────
        agent.train()
        metrics = trainer.update(rollout)
        n_rollouts += 1

        # ── Logging ──────────────────────────────────────────
        if n_rollouts % log_interval == 0:
            elapsed = time.time() - t_start
            fps = global_step / (elapsed + 1e-8)
            recent = episode_rewards[-20:] if episode_rewards else [0]
            mean_r = np.mean(recent)
            std_r = np.std(recent)

            print(f"[Step {global_step:>8,}/{total_steps:,}] "
                  f"R={mean_r:>8.2f}±{std_r:.2f} | "
                  f"π_loss={metrics['policy_loss']:>7.4f} "
                  f"v_loss={metrics['value_loss']:>7.4f} "
                  f"ent={metrics['entropy']:>6.3f} "
                  f"kl={metrics['approx_kl']:>6.4f} "
                  f"clip={metrics['clip_fraction']:>5.3f} | "
                  f"{fps:.0f} fps")

            # Save best
            if mean_r > best_mean_reward and len(episode_rewards) >= 10:
                best_mean_reward = mean_r
                torch.save(agent.state_dict(),
                           os.path.join(save_dir, "best.pt"))

        # Periodic checkpoint
        if n_rollouts % (log_interval * 10) == 0:
            torch.save({
                'agent': agent.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
                'global_step': global_step,
                'episode_rewards': episode_rewards,
            }, os.path.join(save_dir, f"checkpoint_{global_step}.pt"))

    # Final save
    torch.save(agent.state_dict(), os.path.join(save_dir, "final.pt"))
    print(f"\nTraining complete. {global_step:,} steps in {time.time()-t_start:.0f}s")
    print(f"Best mean reward: {best_mean_reward:.4f}")

    return agent, episode_rewards


# ════════════════════════════════════════════════════════════════
# §12  MLP BASELINE (for ablation comparison)
# ════════════════════════════════════════════════════════════════

class MLPAgent(nn.Module):
    """
    MLP baseline agent for ablation: replaces the Transformer encoder
    with a simple feedforward network over the single latest observation.
    Same policy and value heads for fair comparison.
    """
    def __init__(self, cfg: AgentConfig = None):
        super().__init__()
        self.cfg = cfg or AgentConfig()
        c = self.cfg

        # MLP encoder (no sequence, just latest obs)
        self.encoder = nn.Sequential(
            nn.Linear(c.obs_dim, c.d_model),
            nn.GELU(),
            nn.Linear(c.d_model, c.d_model),
            nn.GELU(),
        )
        self.actor = PolicyHead(c)
        self.critic = ValueHead(c)
        self.device = c.get_device()
        self.to(self.device)

    def encode(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # Only use last timestep
        if obs_seq.dim() == 3:
            obs = obs_seq[:, -1, :]
        else:
            obs = obs_seq
        return self.encoder(obs)

    def get_action_and_value(self, obs_seq, deterministic=False):
        z = self.encode(obs_seq)
        if deterministic:
            mu, _ = self.actor(z)
            action = torch.sigmoid(mu)
            lp = self.actor.log_prob(z, action)
        else:
            action, lp = self.actor.sample(z)
        ent = self.actor.entropy(z)
        val = self.critic(z)
        return action, lp, ent, val

    def evaluate_actions(self, obs_seq, actions):
        z = self.encode(obs_seq)
        lp = self.actor.log_prob(z, actions)
        ent = self.actor.entropy(z)
        val = self.critic(z)
        return lp, ent, val

    def get_value(self, obs_seq):
        z = self.encode(obs_seq)
        return self.critic(z)


# ════════════════════════════════════════════════════════════════
# §13  EVALUATION & INFERENCE
# ════════════════════════════════════════════════════════════════

def evaluate_agent(agent, env, n_episodes: int = 10,
                   seq_len: int = 64, obs_dim: int = 57,
                   deterministic: bool = True) -> Dict[str, float]:
    """Evaluate a trained agent over multiple episodes."""
    device = next(agent.parameters()).device
    agent.eval()

    obs_buffer = ObsSequenceBuffer(n_envs=1, seq_len=seq_len, obs_dim=obs_dim)
    all_rewards, all_equities, all_inventories = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        obs_buffer.reset(0)
        obs_buffer.push(obs, 0)

        ep_reward = 0.0
        ep_equities = []
        ep_inventories = []

        while True:
            obs_seq = torch.FloatTensor(
                obs_buffer.get_sequence(0)).unsqueeze(0).to(device)

            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    obs_seq, deterministic=deterministic)

            action_np = action.cpu().numpy().squeeze(0)
            obs, reward, done, trunc, info = env.step(action_np)
            obs_buffer.push(obs, 0)

            ep_reward += reward
            ep_equities.append(info['equity'])
            ep_inventories.append(info['inventory'])

            if done or trunc:
                break

        all_rewards.append(ep_reward)
        all_equities.append(ep_equities)
        all_inventories.append(ep_inventories)

    # Compute metrics
    rewards = np.array(all_rewards)
    final_eq = [eq[-1] for eq in all_equities]
    max_inv = [max(abs(i) for i in inv) for inv in all_inventories]

    return {
        'mean_reward': rewards.mean(),
        'std_reward': rewards.std(),
        'mean_final_equity': np.mean(final_eq),
        'mean_max_inventory': np.mean(max_inv),
        'sharpe': rewards.mean() / (rewards.std() + 1e-8),
    }


# ════════════════════════════════════════════════════════════════
# §14  MAIN
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 2: Transformer-PPO Agent")
    print("=" * 60)

    # ── Smoke test architecture ──
    print("\n[1] Architecture smoke test...")
    cfg = AgentConfig(obs_dim=57, act_dim=4, seq_len=64,
                      d_model=128, n_heads=4, n_layers=3)
    agent = TransformerPPOAgent(cfg)

    # Test forward pass
    dummy_seq = torch.randn(2, 64, 57).to(cfg.get_device())
    with torch.no_grad():
        action, lp, ent, val = agent.get_action_and_value(dummy_seq)
    print(f"    Action shape: {action.shape}")
    print(f"    Log prob: {lp.mean().item():.4f}")
    print(f"    Entropy: {ent.mean().item():.4f}")
    print(f"    Value: {val.mean().item():.4f}")
    print(f"    Action range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    # ── Test reward shaper ──
    print("\n[2] Reward shaper test...")
    rs = RewardShaper()
    for i in range(10):
        r = rs.compute(equity=100 + i * 0.5, inventory=i - 5,
                       n_fills=2, spread=0.02)
        print(f"    Step {i}: equity={100+i*0.5:.1f}, inv={i-5}, "
              f"reward={r:.6f}")

    # ── MLP baseline comparison ──
    print("\n[3] MLP baseline smoke test...")
    mlp = MLPAgent(cfg)
    with torch.no_grad():
        a, lp, ent, v = mlp.get_action_and_value(dummy_seq)
    mlp_params = sum(p.numel() for p in mlp.parameters())
    tf_params = sum(p.numel() for p in agent.parameters())
    print(f"    MLP params: {mlp_params:,}")
    print(f"    Transformer params: {tf_params:,}")
    print(f"    Ratio: {tf_params/mlp_params:.1f}x")

    # ── Training ──
    print("\n[4] To train:")
    print("    agent, rewards = train_agent(total_steps=500_000)")
    print("\n    For quick test:")
    print("    agent, rewards = train_agent(total_steps=20_000)")

    from phase2_agent import TransformerPPOAgent, train_agent
    agent, rewards = train_agent(total_steps=20_000)  
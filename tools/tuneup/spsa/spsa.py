"""
SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

Pure math module -- no I/O, no network, no dependencies beyond stdlib.

Algorithm (range-scaled):
  r_i = upper_i - lower_i  (parameter range)
  delta_k ~ Bernoulli(+1, -1) per parameter

  theta_plus_i  = clamp(theta_i + c_k * delta_i * r_i)
  theta_minus_i = clamp(theta_i - c_k * delta_i * r_i)

  pert_i = c_k * delta_i * r_i  (clamped to |pert| >= 1 in engine space)
  g_hat[i] = (score_plus - score_minus) / (2 * pert_i / r_i)
  theta_{k+1}_i = clamp(theta_i + a_k * g_hat_i * r_i)

  a_k = a / (A + k + 1)^alpha
  c_k = c / (k + 1)^gamma

  c is a fraction of parameter range (e.g., 0.05 = 5%).
  a controls learning rate; the step is a_k * gradient * range.
"""

import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from config import Parameter, SPSAConfig


@dataclass
class SPSAState:
    """Serializable SPSA state for checkpointing."""
    iteration: int = 0
    theta: Dict[str, float] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)
    current_delta: Dict[str, int] = field(default_factory=dict)
    games_completed: int = 0
    total_wins: int = 0       # games won by theta_plus
    total_draws: int = 0      # drawn games
    total_losses: int = 0     # games lost by theta_plus (= won by theta_minus)
    created_at: float = 0.0   # session start timestamp (epoch)
    # Reference mode accumulators (theta+ vs ref, theta- vs ref separately)
    total_plus_wins: int = 0
    total_plus_draws: int = 0
    total_plus_losses: int = 0
    total_minus_wins: int = 0
    total_minus_draws: int = 0
    total_minus_losses: int = 0

    def to_dict(self) -> dict:
        d = {
            "iteration": self.iteration,
            "theta": dict(self.theta),
            "history": list(self.history),
            "current_delta": dict(self.current_delta),
            "games_completed": self.games_completed,
            "total_wins": self.total_wins,
            "total_draws": self.total_draws,
            "total_losses": self.total_losses,
            "created_at": self.created_at,
        }
        # Only persist reference accumulators when non-zero
        for key in ("total_plus_wins", "total_plus_draws", "total_plus_losses",
                     "total_minus_wins", "total_minus_draws", "total_minus_losses"):
            val = getattr(self, key)
            if val:
                d[key] = val
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SPSAState":
        return cls(
            iteration=d["iteration"],
            theta=dict(d["theta"]),
            history=list(d.get("history", [])),
            current_delta={k: int(v) for k, v in d.get("current_delta", {}).items()},
            games_completed=d.get("games_completed", 0),
            total_wins=d.get("total_wins", 0),
            total_draws=d.get("total_draws", 0),
            total_losses=d.get("total_losses", 0),
            created_at=d.get("created_at", 0.0),
            total_plus_wins=d.get("total_plus_wins", 0),
            total_plus_draws=d.get("total_plus_draws", 0),
            total_plus_losses=d.get("total_plus_losses", 0),
            total_minus_wins=d.get("total_minus_wins", 0),
            total_minus_draws=d.get("total_minus_draws", 0),
            total_minus_losses=d.get("total_minus_losses", 0),
        )


class SPSAOptimizer:
    """SPSA optimizer for chess engine parameter tuning."""

    def __init__(self, params: Dict[str, Parameter], spsa_config: SPSAConfig, max_iterations: int, state: SPSAState = None):
        self.params = params
        self.config = spsa_config
        self.max_iterations = max_iterations

        # A = fraction of total iterations for initial stabilization
        self.A = spsa_config.A_ratio * max_iterations

        # Effective perturbation per param (set by compute_candidates, used by update).
        self._effective_perts = {}

        if state is not None:
            self.state = state
        else:
            self.state = SPSAState(
                iteration=0,
                theta={name: p.init for name, p in params.items()},
            )

    @property
    def iteration(self) -> int:
        return self.state.iteration

    @property
    def theta(self) -> Dict[str, float]:
        return self.state.theta

    def a_k(self, k: int = None) -> float:
        """Learning rate at iteration k."""
        if k is None:
            k = self.state.iteration
        return self.config.a / (self.A + k + 1) ** self.config.alpha

    def c_k(self, k: int = None) -> float:
        """Perturbation magnitude at iteration k."""
        if k is None:
            k = self.state.iteration
        return self.config.c / (k + 1) ** self.config.gamma

    def generate_perturbation(self) -> Dict[str, int]:
        """Generate Bernoulli +/-1 perturbation vector."""
        return {name: random.choice([-1, 1]) for name in self.params}

    def compute_candidates(self, delta: Dict[str, int]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute perturbed parameter vectors.

        Enforces a minimum perturbation of +/-1 in engine space so the engine
        always sees distinct values (avoids pure-noise gradient estimates).
        The actual perturbation per parameter is stored in _effective_perts
        for use by update().

        Returns:
            (theta_plus, theta_minus) as dicts of param_name -> engine value.
        """
        ck = self.c_k()
        theta_plus = {}
        theta_minus = {}

        for name, param in self.params.items():
            t = self.state.theta[name]
            d = delta[name]
            r = param.upper - param.lower
            pert = ck * d * r

            # Minimum +/-1 engine-space step.
            if param.is_normalized:
                orig_range = param.original_upper - param.original_lower
                min_pert = 2.0 / orig_range
            else:
                min_pert = 1.0
            if abs(pert) < min_pert:
                pert = d * min_pert

            tp = param.clamp(t + pert)
            tm = param.clamp(t - pert)

            # Verify engine values actually differ; bump if rounding collapsed them.
            if param.denormalize(tp) == param.denormalize(tm):
                pert = d * min_pert * 2
                tp = param.clamp(t + pert)
                tm = param.clamp(t - pert)

            theta_plus[name] = param.to_engine_value(tp)
            theta_minus[name] = param.to_engine_value(tm)
            self._effective_perts[name] = pert / r

        return theta_plus, theta_minus

    def update(self, delta: Dict[str, int], score_plus: float, score_minus: float, games: int) -> Dict[str, float]:
        """
        Compute gradient estimate and update theta.

        Args:
            delta: perturbation vector from generate_perturbation()
            score_plus: aggregated win rate for theta_plus
            score_minus: aggregated win rate for theta_minus
            games: number of games played this iteration (for accurate totals
                when games_per_iteration changes mid-session)

        Returns:
            Updated theta dict.
        """
        ak = self.a_k()
        ck = self.c_k()
        k = self.state.iteration

        new_theta = {}
        for name, param in self.params.items():
            d = delta[name]
            r = param.upper - param.lower
            # Use actual perturbation (accounts for minimum step on int params).
            eff = self._effective_perts.get(name, ck * d)
            g_hat = (score_plus - score_minus) / (2.0 * eff)
            t = self.state.theta[name] + ak * g_hat * r
            t = param.clamp(t)
            new_theta[name] = t

        # Record history
        score_diff = score_plus - score_minus
        self.state.history.append({
            "iteration": k,
            "theta": dict(self.state.theta),
            "score_plus": score_plus,
            "score_minus": score_minus,
            "score_diff": score_diff,
            "elo_diff": self.elo_estimate(score_plus) - self.elo_estimate(score_minus),
            "a_k": ak,
            "c_k": ck,
            "games": games,
        })

        self.state.theta = new_theta
        self.state.iteration = k + 1

        return new_theta

    def advance(self, score_plus: float, score_minus: float, games: int):
        """Advance iteration counter without updating theta (skipped update)."""
        k = self.state.iteration
        self.state.history.append({
            "iteration": k,
            "theta": dict(self.state.theta),
            "score_plus": score_plus,
            "score_minus": score_minus,
            "score_diff": score_plus - score_minus,
            "elo_diff": self.elo_estimate(score_plus) - self.elo_estimate(score_minus),
            "a_k": self.a_k(),
            "c_k": self.c_k(),
            "skipped": True,
            "games": games,
        })
        self.state.iteration = k + 1

    def is_done(self) -> bool:
        return self.state.iteration >= self.max_iterations

    def get_engine_values(self) -> Dict[str, any]:
        """Current theta as engine-facing values (rounded ints, etc.)."""
        return {
            name: self.params[name].to_engine_value(val)
            for name, val in self.state.theta.items()
        }

    def elo_estimate(self, score: float) -> float:
        """Convert win rate to approximate ELO difference."""
        if score <= 0.0 or score >= 1.0:
            return float("inf") if score >= 1.0 else float("-inf")
        from math import log10
        return -400.0 * log10(1.0 / score - 1.0)

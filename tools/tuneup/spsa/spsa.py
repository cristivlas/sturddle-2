"""
SPSA (Simultaneous Perturbation Stochastic Approximation) optimizer.

Pure math module — no I/O, no network, no dependencies beyond stdlib.

Algorithm (range-scaled):
  r_i = upper_i - lower_i  (parameter range)
  delta_k ~ Bernoulli(+1, -1) per parameter

  theta_plus_i  = clamp(theta_i + c_k * delta_i * r_i)
  theta_minus_i = clamp(theta_i - c_k * delta_i * r_i)

  g_hat[i] = (score_plus - score_minus) / (2 * c_k * delta_i)
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

    def to_dict(self) -> dict:
        return {
            "iteration": self.iteration,
            "theta": dict(self.theta),
            "history": list(self.history),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SPSAState":
        return cls(
            iteration=d["iteration"],
            theta=dict(d["theta"]),
            history=list(d.get("history", [])),
        )


class SPSAOptimizer:
    """SPSA optimizer for chess engine parameter tuning."""

    def __init__(self, params: Dict[str, Parameter], spsa_config: SPSAConfig,
                 max_iterations: int, state: SPSAState = None):
        self.params = params
        self.config = spsa_config
        self.max_iterations = max_iterations

        # A = fraction of total iterations for initial stabilization
        self.A = spsa_config.A_ratio * max_iterations

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

    def compute_candidates(self, delta: Dict[str, int]
                           ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Compute perturbed parameter vectors.

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
            tp = param.clamp(t + ck * d * r)
            tm = param.clamp(t - ck * d * r)
            theta_plus[name] = param.to_engine_value(tp)
            theta_minus[name] = param.to_engine_value(tm)

        return theta_plus, theta_minus

    def update(self, delta: Dict[str, int],
               score_plus: float, score_minus: float) -> Dict[str, float]:
        """
        Compute gradient estimate and update theta.

        Args:
            delta: perturbation vector from generate_perturbation()
            score_plus: aggregated win rate for theta_plus
            score_minus: aggregated win rate for theta_minus

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
            g_hat = (score_plus - score_minus) / (2.0 * ck * d)
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
        })

        self.state.theta = new_theta
        self.state.iteration = k + 1

        return new_theta

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

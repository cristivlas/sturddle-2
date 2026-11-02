"""Config dataclasses: tuning.json (shared session) and worker.json (per machine)."""

import json
from dataclasses import dataclass, field, asdict, MISSING
from typing import Any, Dict, Optional


@dataclass
class Parameter:
    """Lakas-compatible tunable parameter."""
    name: str
    init: float
    lower: float
    upper: float
    type: str = "int"  # "int" or "float"
    # engine range when lower/upper are normalized to [-1, 1]
    original_lower: Optional[float] = None
    original_upper: Optional[float] = None

    @property
    def is_normalized(self) -> bool:
        return self.original_lower is not None

    def clamp(self, value: float) -> float:
        return max(self.lower, min(self.upper, value))

    def discretize(self, value: float) -> float:
        if self.type == "int":
            return round(value)
        return value

    def to_engine_value(self, value: float):
        v = self.clamp(value)
        if self.type == "int":
            return int(round(v))
        return v

    def denormalize(self, theta_val: float) -> int:
        if self.is_normalized:
            lo, hi = self.original_lower, self.original_upper
            return int(round((theta_val + 1) * (hi - lo) / 2 + lo))
        return int(round(theta_val))


@dataclass
class SPSAConfig:
    """c is a fraction of each parameter's range; the step is a_k * gradient * range."""
    budget: int = 10000
    a: float = 0.5
    c: float = 0.05
    A_ratio: float = 0.1
    alpha: float = 0.602
    gamma: float = 0.101
    draw_weight: float = 0.5

    def validate(self):
        errors = []
        if self.budget < 1:
            errors.append("budget must be >= 1")
        if self.a <= 0:
            errors.append("a (learning rate) must be > 0")
        if self.c <= 0:
            errors.append("c (perturbation) must be > 0")
        if not (0 < self.A_ratio < 1):
            errors.append("A_ratio must be in (0, 1)")
        if self.alpha <= 0:
            errors.append("alpha must be > 0")
        if self.gamma <= 0:
            errors.append("gamma must be > 0")
        if not (0 <= self.draw_weight <= 0.5):
            errors.append("draw_weight must be in [0, 0.5]")
        if errors:
            raise ValueError("SPSA config: " + "; ".join(errors))


@dataclass
class EngineConfig:
    protocol: str = "uci"
    fixed_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningConfig:
    """Session config shared by all workers. depth, if set, overrides time_control."""
    engine: EngineConfig = field(default_factory=EngineConfig)
    time_control: str = "1+0.1"
    depth: Optional[int] = None
    games_per_iteration: int = 200
    output_dir: str = "./spsa_output"
    retry_after: int = 5            # seconds
    dashboard_refresh: int = 60     # SSE heartbeat, seconds
    dashboard_history: int = 100    # 0 = unlimited
    overdue_factor: float = 1.35
    worker_idle_timeout: float = 60.0
    chunk_timeout_factor: float = 2.0
    min_chunk_timeout: float = 60.0
    min_chunk_expected_duration: float = 60.0
    overflow_factor: float = 1.15   # max games dispatched per iteration / games_per_iteration
    ewma_warmup_chunks: int = 3     # chunks before a worker gets speed-proportional work
    static_dir: str = ""            # favicon etc., empty = disabled
    log_rotation: bool = True
    validate_interval: int = 5      # seconds, 0 = disabled
    max_retries: int = 3            # reconnects within max_retries * retry_after seconds, 0 = unlimited
    auto_resign: bool = True
    resign_movecount: int = 3
    resign_score: int = 700
    draw_movenumber: int = 40
    draw_movecount: int = 8
    draw_score: int = 10
    spsa: SPSAConfig = field(default_factory=SPSAConfig)
    parameters: Dict[str, Parameter] = field(default_factory=dict)

    def max_iterations(self) -> int:
        return self.spsa.budget // self.games_per_iteration

    def validate(self):
        self.spsa.validate()
        errors = []
        if self.games_per_iteration < 2:
            errors.append("games_per_iteration must be >= 2")
        if self.spsa.budget < self.games_per_iteration:
            errors.append("budget must be >= games_per_iteration")
        if self.worker_idle_timeout <= 0:
            errors.append("worker_idle_timeout must be > 0")
        if self.overdue_factor < 1:
            errors.append("overdue_factor must be >= 1")
        if self.chunk_timeout_factor < 1:
            errors.append("chunk_timeout_factor must be >= 1")
        if self.overflow_factor < 1:
            errors.append("overflow_factor must be >= 1")
        if self.ewma_warmup_chunks < 0:
            errors.append("ewma_warmup_chunks must be >= 0")
        if self.min_chunk_timeout < self.min_chunk_expected_duration:
            self.min_chunk_timeout = self.min_chunk_expected_duration
        for name, p in self.parameters.items():
            if p.lower >= p.upper:
                errors.append(f"parameter {name}: lower must be < upper")
            if not (p.lower <= p.init <= p.upper):
                errors.append(f"parameter {name}: init must be in [lower, upper]")
        if errors:
            raise ValueError("Tuning config: " + "; ".join(errors))

    def to_json(self) -> str:
        def _param_dict(p: Parameter) -> dict:
            d = {"init": p.init, "lower": p.lower, "upper": p.upper, "type": p.type}
            if p.original_lower is not None:
                d["original_lower"] = p.original_lower
            if p.original_upper is not None:
                d["original_upper"] = p.original_upper
            return d

        d = {
            "engine": asdict(self.engine),
            "time_control": self.time_control,
            "depth": self.depth,
            "games_per_iteration": self.games_per_iteration,
            "retry_after": self.retry_after,
            "dashboard_refresh": self.dashboard_refresh,
            "dashboard_history": self.dashboard_history,
            "output_dir": self.output_dir,
            "overdue_factor": self.overdue_factor,
            "worker_idle_timeout": self.worker_idle_timeout,
            "chunk_timeout_factor": self.chunk_timeout_factor,
            "min_chunk_timeout": self.min_chunk_timeout,
            "min_chunk_expected_duration": self.min_chunk_expected_duration,
            "overflow_factor": self.overflow_factor,
            "ewma_warmup_chunks": self.ewma_warmup_chunks,
            "static_dir": self.static_dir,
            "validate_interval": self.validate_interval,
            "max_retries": self.max_retries,
            "auto_resign": self.auto_resign,
            "resign_movecount": self.resign_movecount,
            "resign_score": self.resign_score,
            "draw_movenumber": self.draw_movenumber,
            "draw_movecount": self.draw_movecount,
            "draw_score": self.draw_score,
            "spsa": asdict(self.spsa),
            "parameters": { name: _param_dict(p) for name, p in self.parameters.items() },
        }
        d = {k: v for k, v in d.items() if v is not None}
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TuningConfig":
        with open(path) as f:
            d = json.load(f)

        engine = EngineConfig(**d.get("engine", {}))
        spsa = SPSAConfig(**d.get("spsa", {}))

        parameters = {}
        for name, pdict in d.get("parameters", {}).items():
            parameters[name] = Parameter(name=name, **pdict)

        fields = cls.__dataclass_fields__
        skip = {"engine", "spsa", "parameters"}
        kwargs = {}
        for name, f in fields.items():
            if name in skip:
                continue
            default = None if f.default is MISSING else f.default
            kwargs[name] = d.get(name, default)

        cfg = cls(engine=engine, spsa=spsa, parameters=parameters, **kwargs)
        cfg.validate()
        return cfg


@dataclass
class WorkerConfig:
    """Per-machine worker config; all paths are local."""
    name: str = ""  # defaults to hostname
    coordinator: str = "http://localhost:8080"
    engine: str = "./sturddle"
    cutechess_cli: str = "cutechess-cli"
    concurrency: int = 1
    opening_book: str = ""
    book_format: str = ""
    book_depth: int = 8
    games_dir: str = "./games"
    log_file: str = "worker.log"
    max_chunk_size: int = 0  # 0 = unlimited
    max_rounds_per_chunk: int = 10  # cap = concurrency * this * 2, 0 = unlimited
    http_retry_timeout: int = 300  # seconds
    parameter_overrides: dict = field(default_factory=dict)
    cutechess_overrides: dict = field(default_factory=dict)  # tc, depth
    log_rotation: bool = True
    reference_engine: str = ""  # theta+/- play this instead of each other
    ramdisk: bool = False  # for PyInstaller --onefile engines
    ramdisk_drive: str = ""  # empty = auto-select
    auto_install_imdisk: bool = True
    ramdisk_size: int = 0  # MB, 0 = estimate
    ramdisk_decompression: float = 2.6
    max_forfeit_pct: float = 0.05  # 0 = disabled
    max_retries: int = 3  # consecutive errors, 0 = unlimited

    @classmethod
    def from_json(cls, path: str) -> "WorkerConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkItem:
    """A batch of games assigned to a worker."""
    iteration: int
    theta_plus: Dict[str, Any]   # engine values
    theta_minus: Dict[str, Any]
    num_games: int
    chunk_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkItem":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WorkResult:
    """Game counts for a batch; PGNs stay on the worker."""
    iteration: int
    num_games: int
    wins: int = 0       # theta_plus, standard mode
    draws: int = 0
    losses: int = 0
    chunk_id: str = ""
    worker: str = ""
    shutting_down: bool = False
    # reference mode
    reference_mode: bool = False
    plus_wins: int = 0
    plus_draws: int = 0
    plus_losses: int = 0
    minus_wins: int = 0
    minus_draws: int = 0
    minus_losses: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

from dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class EnvConfig:
    map_size: int = MISSING
    frozen_prob: float = MISSING
    seed: int = MISSING
    is_slippery: bool = MISSING
    render_mode: str = MISSING

@dataclass
class PolicyIterationConfig:
    gamma: float = MISSING
    tol: float = MISSING

@dataclass
class QLearningConfig:
    num_episodes: int = MISSING
    gamma: float = MISSING
    learning_rate: float = MISSING
    epsilon: float = MISSING
    epsilon_decay: float = MISSING

@dataclass
class RenderConfig:
    max_steps: int = MISSING

@dataclass
class Config:
    env: EnvConfig = MISSING
    policy_iteration: PolicyIterationConfig = MISSING
    qlearning: QLearningConfig = MISSING
    render: RenderConfig = MISSING
    algorithm: str = MISSING

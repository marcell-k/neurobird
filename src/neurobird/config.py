"""Central configuration for the simulation, agent and training."""

import math
from dataclasses import dataclass
from pathlib import Path

# UI / assets
IMAGES_DIR = Path("images")
WHITE = (255, 255, 255)

# Simulation / world
FPS = 60
STEP_MSEC = 1000.0 / FPS
WORLD_WIDTH = 284 * 2
WORLD_HEIGHT = 512

BIRD_X = 50
BIRD_SIZE = 32
FLAP_VELOCITY = -0.45  # px/ms, negative is up
GRAVITY = 0.0018  # px/ms^2
MAX_FALL_SPEED = 0.6  # px/ms

PIPE_WIDTH = 80
PIECE_HEIGHT = 32
GAP_HEIGHT = 128
PIPE_SPEED = 0.18  # px/ms
PIPE_SPACING = 540.0  # px between consecutive pipes (3000 ms at the base speed)
MAX_GAP_TOP_PIECES = 10
MOVE_AMPLITUDE = 48  # px a moving gap swings above and below its resting position
MOVE_PERIOD_FRAMES = 120  # frames per full up-and-down swing
MAX_GAP_SPEED = MOVE_AMPLITUDE * 2 * math.pi / MOVE_PERIOD_FRAMES  # px/frame, peak


def _lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Tunable rules. Each value starts at its base and ramps linearly to its end value."""

    pipe_speed: float = PIPE_SPEED  # px/ms
    gap_height: int = GAP_HEIGHT  # px
    pipe_spacing: float = PIPE_SPACING  # px between consecutive pipes
    ramp_frames: int = 0  # frames until the end values are reached; 0 disables the ramp
    end_pipe_speed: float = PIPE_SPEED
    end_gap_height: int = GAP_HEIGHT
    end_pipe_spacing: float = PIPE_SPACING
    moving_pipes: bool = False

    def progress(self, frames: int) -> float:
        """Fraction of the ramp completed, from 0.0 to 1.0."""
        if self.ramp_frames <= 0:
            return 0.0
        return min(frames / self.ramp_frames, 1.0)

    def speed_at(self, frames: int) -> float:
        """Pipe speed in px/ms after the given number of frames."""
        return _lerp(self.pipe_speed, self.end_pipe_speed, self.progress(frames))

    def gap_at(self, frames: int) -> int:
        """Gap height in px for a pipe spawned after the given number of frames."""
        return round(_lerp(self.gap_height, self.end_gap_height, self.progress(frames)))

    def spacing_at(self, frames: int) -> float:
        """Distance in px between consecutive pipes after the given number of frames."""
        return _lerp(self.pipe_spacing, self.end_pipe_spacing, self.progress(frames))


EASY = GameConfig()
HARD = GameConfig(
    ramp_frames=60 * FPS,
    end_pipe_speed=0.27,
    end_gap_height=96,
    end_pipe_spacing=320,
    moving_pipes=True,
)


CONFIGS = {"easy": EASY, "hard": HARD}

# Agent / fitness
MAX_FRAMES = 10_000  # cap so a perfect bird cannot run forever
SCORE_BONUS = 100  # fitness reward per pipe passed
HALF_HEIGHT = WORLD_HEIGHT / 2
DX_SCALE = WORLD_WIDTH - BIRD_X
SPEED_SCALE = max(config.end_pipe_speed for config in CONFIGS.values()) * 1.1


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters and defaults for the evolutionary training loop."""

    population: int = 200
    elites: int = 10  # copied unchanged into the next generation
    tournament_size: int = 5  # parents are the best of this many random picks
    crossover_prob: float = 0.5  # chance a child mixes two parents instead of cloning one
    episodes_start: int = 4  # games per network in the first generation, on shared seeds
    episodes_end: int = 12  # games per network in the last generation
    hall_size: int = 20  # per-generation champions re-evaluated on held-out seeds
    holdout_episodes: int = 50  # fixed seeds used only to pick the network that gets saved
    mutation_start: float = 0.2  # noise std in the first generation
    mutation_end: float = 0.02  # noise std in the last generation
    mutation_rate: float = 0.5  # fraction of weights that get noise in each child
    curriculum_fraction: float = 0.3  # share of generations spent on easy first
    workers: int = 4  # parallel evaluation processes
    curriculum: bool = True


DEFAULT_TRAINING = TrainingConfig()

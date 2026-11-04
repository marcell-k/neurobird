"""Glue between the neural network and the headless game."""

import random
from typing import TYPE_CHECKING

import numpy as np

from neurobird import sim
from neurobird.config import (
    BIRD_SIZE,
    BIRD_X,
    DX_SCALE,
    EASY,
    GAP_HEIGHT,
    HALF_HEIGHT,
    MAX_FALL_SPEED,
    MAX_FRAMES,
    MAX_GAP_SPEED,
    SCORE_BONUS,
    SPEED_SCALE,
    STEP_MSEC,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    GameConfig,
)

if TYPE_CHECKING:
    from neurobird.nn import NeuralNetwork, Vector


def _norm_y(y: float) -> float:
    """Map a vertical pixel position to [-1,1]."""

    return 2.0 * y / WORLD_HEIGHT - 1.0


def _placeholder_pipe() -> sim.Pipe:
    """Far away pipe with a centered gap, used before the first pipe exists."""
    top = (WORLD_HEIGHT - GAP_HEIGHT) // 2
    return sim.Pipe(WORLD_WIDTH - 1, top, top + GAP_HEIGHT, base_top=top)


def _frames_to_reach(game: sim.Game, pipe: sim.Pipe) -> int:
    """Frames until the pipe's leading edge meets the bird's right edge (0 if it already has)."""
    distance = pipe.x - (BIRD_X + BIRD_SIZE)
    per_frame = game.config.speed_at(game.frames) * STEP_MSEC
    return max(0, round(distance / per_frame))


def _gap_features(game: sim.Game, pipe: sim.Pipe) -> list[float]:
    """Seven inputs describing the bird relative to one pipe's gap."""
    bird_center = game.y + BIRD_SIZE / 2
    gap_height = pipe.gap_bottom - pipe.gap_top
    arrival_top = game.predicted_gap_top(pipe, _frames_to_reach(game, pipe))
    return [
        (pipe.x - BIRD_X) / DX_SCALE,
        (bird_center - (pipe.gap_top + gap_height / 2)) / HALF_HEIGHT,  # + means below center
        (game.y - pipe.gap_top) / HALF_HEIGHT,  # clearance to the top pipe, < 0 means collision
        (pipe.gap_bottom - (game.y + BIRD_SIZE)) / HALF_HEIGHT,  # clearance to the bottom pipe
        pipe.gap_velocity / MAX_GAP_SPEED,
        (arrival_top + gap_height / 2 - bird_center) / HALF_HEIGHT,  # gap center on arrival
        gap_height / WORLD_HEIGHT,
    ]


def _lookahead_features(first: sim.Pipe, second: sim.Pipe | None) -> list[float]:
    """Two inputs about the pipe after the next one (a neutral value if it hasn't spawned)."""
    if second is None:
        return [1.0, 0.0]
    center_shift = (second.gap_top + second.gap_bottom - first.gap_top - first.gap_bottom) / 2
    return [(second.x - BIRD_X) / DX_SCALE, center_shift / HALF_HEIGHT]


# Inputs:
# 1.  Bird vertical position
# 2.  Bird vertical velocity
# 3.  Current pipe speed
# 4.  Horizontal distance to the next pipe
# 5.  Bird center offset from the gap center
# 6.  Clearance to the top pipe (bird_top - gap_top)
# 7.  Clearance to the bottom pipe (gap_bottom - bird_bottom)
# 8.  Gap vertical velocity
# 9.  Predicted gap center offset at arrival, relative to the bird
# 10. Gap height
# 11. Horizontal distance to the second upcoming pipe
# 12. Gap center shift between the next two pipes
OBSERVATION_SIZE = 12


def observe(game: sim.Game) -> Vector:
    """Build the network's normalized input vector, of length OBSERVATION_SIZE."""
    upcoming = game.upcoming_pipes(2)
    first = upcoming[0] if upcoming else _placeholder_pipe()
    second = upcoming[1] if len(upcoming) > 1 else None
    return np.array(
        [
            _norm_y(game.y + BIRD_SIZE / 2),
            game.velocity / MAX_FALL_SPEED,
            game.config.speed_at(game.frames) / SPEED_SCALE,
            *_gap_features(game, first),
            *_lookahead_features(first, second),
        ],
        dtype=np.float64,
    )


def should_flap(nn: NeuralNetwork, game: sim.Game) -> bool:
    """Flap when the network's raw output is positive."""
    return nn.forward(observe(game)) > 0.0


def run_episode(
    nn: NeuralNetwork, seed: int, config: GameConfig = EASY, max_frames: int = MAX_FRAMES
) -> float:
    """Play one game with the network; fitness is frames survived plus a score bonus."""
    game = sim.Game(random.Random(seed), config)
    while game.alive and game.frames < max_frames:
        game.step(flap=should_flap(nn, game))
    return game.frames + SCORE_BONUS * game.score

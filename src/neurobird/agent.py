"""Glue between the neural network and the headless game."""

import random
from typing import TYPE_CHECKING

import numpy as np

from neurobird import sim

if TYPE_CHECKING:
    from neurobird.nn import NeuralNetwork, Vector

MAX_FRAMES = 10_000  # cap so a perfect bird cannot run forever
SCORE_BONUS = 100  # fitness reward per pipe passed
HALF_HEIGHT = sim.WORLD_HEIGHT / 2
DX_SCALE = sim.WORLD_WIDTH - sim.BIRD_X
SPEED_SCALE = 0.3  # px/ms, a bit above the fastest pipe speed in any config


def _norm_y(y: float) -> float:
    """Map a vertical pixel position to [-1,1]."""

    return 2.0 * y / sim.WORLD_HEIGHT - 1.0


def _placeholder_pipe() -> sim.Pipe:
    """Far away pipe with a centered gap, used before the first pipe exists."""
    top = (sim.WORLD_HEIGHT - sim.GAP_HEIGHT) // 2
    return sim.Pipe(sim.WORLD_WIDTH - 1, top, top + sim.GAP_HEIGHT, base_top=top)


def _frames_to_reach(game: sim.Game, pipe: sim.Pipe) -> int:
    """Frames until the pipe's leading edge meets the bird's right edge (0 if it already has)."""
    distance = pipe.x - (sim.BIRD_X + sim.BIRD_SIZE)
    per_frame = game.config.speed_at(game.frames) * sim.STEP_MSEC
    return max(0, round(distance / per_frame))


def _gap_features(game: sim.Game, pipe: sim.Pipe) -> list[float]:
    """Seven inputs describing the bird relative to one pipe's gap."""
    bird_center = game.y + sim.BIRD_SIZE / 2
    gap_height = pipe.gap_bottom - pipe.gap_top
    arrival_top = game.predicted_gap_top(pipe, _frames_to_reach(game, pipe))
    return [
        (pipe.x - sim.BIRD_X) / DX_SCALE,
        (bird_center - (pipe.gap_top + gap_height / 2)) / HALF_HEIGHT,  # + means below center
        (game.y - pipe.gap_top) / HALF_HEIGHT,  # clearance to the top pipe, < 0 means collision
        (pipe.gap_bottom - (game.y + sim.BIRD_SIZE)) / HALF_HEIGHT,  # clearance to the bottom pipe
        pipe.gap_velocity / sim.MAX_GAP_SPEED,
        (arrival_top + gap_height / 2 - bird_center) / HALF_HEIGHT,  # gap center on arrival
        gap_height / sim.WORLD_HEIGHT,
    ]


def _lookahead_features(first: sim.Pipe, second: sim.Pipe | None) -> list[float]:
    """Two inputs about the pipe after the next one (a neutral value if it hasn't spawned)."""
    if second is None:
        return [1.0, 0.0]
    center_shift = (second.gap_top + second.gap_bottom - first.gap_top - first.gap_bottom) / 2
    return [(second.x - sim.BIRD_X) / DX_SCALE, center_shift / HALF_HEIGHT]


def observe(game: sim.Game) -> Vector:
    """Build the network's INPUT_SIZE normalized inputs from the current game state."""
    upcoming = game.upcoming_pipes(2)
    first = upcoming[0] if upcoming else _placeholder_pipe()
    second = upcoming[1] if len(upcoming) > 1 else None
    return np.array(
        [
            _norm_y(game.y + sim.BIRD_SIZE / 2),
            game.velocity / sim.MAX_FALL_SPEED,
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
    nn: NeuralNetwork, seed: int, config: sim.GameConfig = sim.EASY, max_frames: int = MAX_FRAMES
) -> float:
    """Play one game with the network; fitness is frames survived plus a score bonus."""
    game = sim.Game(random.Random(seed), config)
    while game.alive and game.frames < max_frames:
        game.step(flap=should_flap(nn, game))
    return game.frames + SCORE_BONUS * game.score

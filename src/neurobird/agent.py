"""Glue between the neural network and the headless game."""

import random
from typing import TYPE_CHECKING

import numpy as np

from neurobird import sim

if TYPE_CHECKING:
    from neurobird.nn import NeuralNetwork, Vector

MAX_FRAMES = 10_000  # cap so a perfect bird cannot run forever
SCORE_BONUS = 100  # fitness reward per pipe passed


def _norm_y(y: float) -> float:
    """Map a vertical pixel position to [-1,1]."""

    return 2.0 * y / sim.WORLD_HEIGHT - 1.0


def observe(game: sim.Game) -> Vector:
    """Build the network's five normalized inputs from the current game state."""
    pipe = game.next_pipe()
    if pipe is None:
        # No pipe yet: pretend a far away pipe with a centered gap.
        dx = 1.0
        gap_top = (sim.WORLD_HEIGHT - sim.GAP_HEIGHT) / 2
        gap_bottom = gap_top + sim.GAP_HEIGHT
    else:
        dx = (pipe.x - sim.BIRD_X) / (sim.WORLD_WIDTH - sim.BIRD_X)
        gap_top, gap_bottom = pipe.gap_top, pipe.gap_bottom
    return np.array(
        [
            _norm_y(game.y + sim.BIRD_SIZE / 2),
            game.velocity / sim.MAX_FALL_SPEED,
            dx,
            _norm_y(gap_top),
            _norm_y(gap_bottom),
        ],
        dtype=np.float64,
    )


def should_flap(nn: NeuralNetwork, game: sim.Game) -> bool:
    """Flap when the network's raw output is positive."""
    return nn.forward(observe(game)) > 0.0


def run_episode(nn: NeuralNetwork, seed: int, max_frames: int = MAX_FRAMES) -> float:
    """Play one game with the network; fitness is frames survived plus a score bonus."""
    game = sim.Game(random.Random(seed))
    while game.alive and game.frames < max_frames:
        game.step(flap=should_flap(nn, game))
    return game.frames + SCORE_BONUS * game.score

"""Simulation, needed for neural network headless run."""

import math
from collections import deque
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

from neurobird.config import (
    BIRD_SIZE,
    BIRD_X,
    FLAP_VELOCITY,
    GRAVITY,
    HARD,
    MAX_FALL_SPEED,
    MAX_GAP_SPEED,
    MAX_GAP_TOP_PIECES,
    MOVE_AMPLITUDE,
    MOVE_PERIOD_FRAMES,
    PIECE_HEIGHT,
    PIPE_WIDTH,
    STEP_MSEC,
    WORLD_HEIGHT,
    WORLD_WIDTH,
    GameConfig,
)

if TYPE_CHECKING:
    import random


@dataclass(slots=True)
class Pipe:
    """A pair of pipes described by the vertical gap between them."""

    x: float
    gap_top: int
    gap_bottom: int
    scored: bool = False
    base_top: int = 0  # resting gap_top; a moving gap oscillates around it
    phase: float = 0.0  # radians
    age: int = 0  # frames since spawn
    gap_velocity: float = 0.0  # px/frame, positive is down

    @property
    def gap_height(self) -> int:
        """Vertical size of the gap in px."""
        return self.gap_bottom - self.gap_top

    def gap_top_at(self, age: int) -> int:
        """Gap top of a moving pipe at the given age, kept on screen."""
        low, high = PIECE_HEIGHT, MAX_GAP_TOP_PIECES * PIECE_HEIGHT
        return max(
            low, min(high, round(self.base_top + MOVE_AMPLITUDE * math.sin(self._angle(age))))
        )

    def advance(self) -> None:
        """Oscillate the gap around its resting position, keeping it on screen."""
        height = self.gap_height
        self.gap_top = self.gap_top_at(self.age)
        self.gap_bottom = self.gap_top + height
        self.gap_velocity = MAX_GAP_SPEED * math.cos(self._angle(self.age))
        self.age += 1

    def hits(self, y: float) -> bool:
        """Whether a bird whose top edge is at `y` overlaps this pipe."""
        return (
            self.x < BIRD_X + BIRD_SIZE
            and self.x + PIPE_WIDTH > BIRD_X
            and (y < self.gap_top or y + BIRD_SIZE > self.gap_bottom)
        )

    def _angle(self, age: int) -> float:
        return self.phase + 2 * math.pi * age / MOVE_PERIOD_FRAMES


class Game:
    """One run of the game. Advance it one frame at a time with `step`."""

    def __init__(self, rng: random.Random, config: GameConfig = HARD) -> None:
        """Start a new run using the given random generator and rules."""
        self._rng = rng
        self.config = config
        self.y = WORLD_HEIGHT / 2 - BIRD_SIZE / 2
        self.velocity = 0.0
        self.pipes: deque[Pipe] = deque()
        self.score = 0
        self.frames = 0
        self.alive = True

    def step(self, *, flap: bool) -> None:
        """Advance one fixed time step, optionally flapping. Does nothing once dead."""
        if not self.alive:
            return
        if self._should_spawn():
            self._spawn_pipe()
        if flap:
            self.velocity = FLAP_VELOCITY
        self.velocity = min(self.velocity + GRAVITY * STEP_MSEC, MAX_FALL_SPEED)
        self.y += self.velocity * STEP_MSEC
        self._advance_pipes(self.config.speed_at(self.frames) * STEP_MSEC)
        if self._collides():
            self.alive = False
        self.frames += 1

    def upcoming_pipes(self, count: int) -> list[Pipe]:
        """Get up to `count` pipes the bird has not fully passed, nearest first."""
        return list(islice((p for p in self.pipes if p.x + PIPE_WIDTH >= BIRD_X), count))

    def predicted_gap_top(self, pipe: Pipe, frames_ahead: int) -> int:
        """Where the pipe's gap top will be after `frames_ahead` more steps."""
        if frames_ahead <= 0 or not self.config.moving_pipes:
            return pipe.gap_top
        return pipe.gap_top_at(pipe.age + frames_ahead - 1)

    def _advance_pipes(self, dx: float) -> None:
        """Scroll pipes left by `dx`, move their gaps, score passed pipes, drop off-screen ones."""
        for pipe in self.pipes:
            pipe.x -= dx
            if self.config.moving_pipes:
                pipe.advance()
            if not pipe.scored and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.scored = True
                self.score += 1
        while self.pipes and self.pipes[0].x <= -PIPE_WIDTH:
            self.pipes.popleft()

    def _should_spawn(self) -> bool:
        if not self.pipes:
            return True
        return self.pipes[-1].x <= WORLD_WIDTH - 1 - self.config.spacing_at(self.frames)

    def _spawn_pipe(self) -> None:
        gap_top = self._rng.randint(1, MAX_GAP_TOP_PIECES) * PIECE_HEIGHT
        gap_height = self.config.gap_at(self.frames)
        phase = self._rng.uniform(0.0, 2 * math.pi) if self.config.moving_pipes else 0.0
        pipe = Pipe(WORLD_WIDTH - 1, gap_top, gap_top + gap_height, base_top=gap_top, phase=phase)
        self.pipes.append(pipe)

    def _collides(self) -> bool:
        if self.y <= 0 or self.y >= WORLD_HEIGHT - BIRD_SIZE:
            return True
        return any(pipe.hits(self.y) for pipe in self.pipes)

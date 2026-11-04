"""Simulation, needed for neural network headless run."""

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import random

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


def _move_gap(pipe: Pipe) -> None:
    """Oscillate a pipe's gap around its resting position, keeping it on screen."""
    angle = pipe.phase + 2 * math.pi * pipe.age / MOVE_PERIOD_FRAMES
    low, high = PIECE_HEIGHT, MAX_GAP_TOP_PIECES * PIECE_HEIGHT
    size = pipe.gap_bottom - pipe.gap_top
    pipe.gap_top = max(low, min(high, round(pipe.base_top + MOVE_AMPLITUDE * math.sin(angle))))
    pipe.gap_bottom = pipe.gap_top + size
    pipe.gap_velocity = MAX_GAP_SPEED * math.cos(angle)
    pipe.age += 1


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
        speed = self.config.speed_at(self.frames)
        for pipe in self.pipes:
            pipe.x -= speed * STEP_MSEC
        while self.pipes and self.pipes[0].x <= -PIPE_WIDTH:
            self.pipes.popleft()
        if self.config.moving_pipes:
            for pipe in self.pipes:
                _move_gap(pipe)
        if self._collides():
            self.alive = False
        for pipe in self.pipes:
            if not pipe.scored and pipe.x + PIPE_WIDTH < BIRD_X:
                pipe.scored = True
                self.score += 1
        self.frames += 1

    def next_pipe(self) -> Pipe | None:
        """Get the closest pipe the bird has not fully passed, if any."""
        return next((p for p in self.pipes if p.x + PIPE_WIDTH >= BIRD_X), None)

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
        return any(
            pipe.x < BIRD_X + BIRD_SIZE
            and pipe.x + PIPE_WIDTH > BIRD_X
            and (self.y < pipe.gap_top or self.y + BIRD_SIZE > pipe.gap_bottom)
            for pipe in self.pipes
        )

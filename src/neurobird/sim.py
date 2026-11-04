"""Simulation, needed for neural network headless run."""

from collections import deque
from dataclasses import dataclass
lazy import random

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
SPAWN_INTERVAL_FRAMES = 180  # 3000 ms at 60 FPS
MAX_GAP_TOP_PIECES = 10


@dataclass(slots=True)
class Pipe:
    """A pair of pipes described by the vertical gap between them."""

    x: float
    gap_top: int
    gap_bottom: int
    scored: bool = False


class Game:
    """One run of the game. Advance it one frame at a time with `step`."""

    def __init__(self, rng: random.Random) -> None:
        """Start a new run using the given random generator for pipe placement."""
        self._rng = rng
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
        if self.frames % SPAWN_INTERVAL_FRAMES == 0:
            self._spawn_pipe()
        if flap:
            self.velocity = FLAP_VELOCITY
        self.velocity = min(self.velocity + GRAVITY * STEP_MSEC, MAX_FALL_SPEED)
        self.y += self.velocity * STEP_MSEC
        for pipe in self.pipes:
            pipe.x -= PIPE_SPEED * STEP_MSEC
        while self.pipes and self.pipes[0].x <= -PIPE_WIDTH:
            self.pipes.popleft()
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

    def _spawn_pipe(self) -> None:
        gap_top = self._rng.randint(1, MAX_GAP_TOP_PIECES) * PIECE_HEIGHT
        self.pipes.append(Pipe(WORLD_WIDTH - 1, gap_top, gap_top + GAP_HEIGHT))

    def _collides(self) -> bool:
        if self.y <= 0 or self.y >= WORLD_HEIGHT - BIRD_SIZE:
            return True
        return any(
            pipe.x < BIRD_X + BIRD_SIZE
            and pipe.x + PIPE_WIDTH > BIRD_X
            and (self.y < pipe.gap_top or self.y + BIRD_SIZE > pipe.gap_bottom)
            for pipe in self.pipes
        )

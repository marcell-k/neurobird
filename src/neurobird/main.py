"""Flappy Bird, implemented using Pygame."""

import math
import time
from collections import deque
from pathlib import Path
from random import randint

import pygame
from pygame import SRCALPHA, Rect, Surface

FPS = 60
ANIMATION_SPEED = 0.18  # pixels per millisecond
WIN_WIDTH = 284 * 2  # BG image size: 284x512 px; tiled twice
WIN_HEIGHT = 512
IMAGES_DIR = Path("images")


class Bird(pygame.sprite.Sprite):
    """Represents the bird controlled by the player."""

    WIDTH = HEIGHT = 32
    FLAP_VELOCITY = -0.45  # px/ms, negative is up
    GRAVITY = 0.0018  # px/ms^2
    MAX_FALL_SPEED = 0.6  # px/ms

    def __init__(self, x: float, y: float, images) -> None:
        """Initialise a new Bird instance."""
        self.x, self.y = x, y
        self.velocity = 0.0
        self._img_wingup, self._img_wingdown = images
        self._mask_wingup = pygame.mask.from_surface(self._img_wingup)
        self._mask_wingdown = pygame.mask.from_surface(self._img_wingdown)

    def flap(self) -> None:
        """Set the vertical velocity to a fixed upward value."""
        self.velocity = Bird.FLAP_VELOCITY

    def update(self, delta_frames: int = 1) -> None:
        """Apply gravity to the velocity, then move the bird."""
        dt = _frames_to_msec(delta_frames)
        self.velocity = min(self.velocity + Bird.GRAVITY * dt, Bird.MAX_FALL_SPEED)
        self.y += self.velocity * dt

    @property
    def image(self):
        """Get a Surface containing this bird's image."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._img_wingup
        return self._img_wingdown

    @property
    def mask(self):
        """Get a bitmask for use in collision detection."""
        if pygame.time.get_ticks() % 500 >= 250:
            return self._mask_wingup
        return self._mask_wingdown

    @property
    def rect(self):
        """Get the bird's position, width, and height, as a pygame.Rect."""
        return Rect(self.x, self.y, Bird.WIDTH, Bird.HEIGHT)


class PipePair(pygame.sprite.Sprite):
    """Represents an obstacle."""

    WIDTH = 80
    PIECE_HEIGHT = 32
    ADD_INTERVAL = 3000

    def __init__(self, pipe_end_img: Surface, pipe_body_img: Surface) -> None:
        """Initialize new random PipePairs."""
        self.x = float(WIN_WIDTH - 1)
        self.score_counted = False

        self.image = pygame.Surface((PipePair.WIDTH, WIN_HEIGHT), SRCALPHA)
        self.image.fill((0, 0, 0, 0))
        total_pipe_body_pieces = int(
            (WIN_HEIGHT - 3 * Bird.HEIGHT - 3 * PipePair.PIECE_HEIGHT) / PipePair.PIECE_HEIGHT
        )
        self.bottom_pieces = randint(1, total_pipe_body_pieces)
        self.top_pieces = total_pipe_body_pieces - self.bottom_pieces

        # bottom pipe
        for i in range(1, self.bottom_pieces + 1):
            piece_pos = (0, WIN_HEIGHT - i * PipePair.PIECE_HEIGHT)
            self.image.blit(pipe_body_img, piece_pos)
        bottom_pipe_end_y = WIN_HEIGHT - self.bottom_height_px
        bottom_end_piece_pos = (0, bottom_pipe_end_y - PipePair.PIECE_HEIGHT)
        self.image.blit(pipe_end_img, bottom_end_piece_pos)

        # top pipe
        for i in range(self.top_pieces):
            self.image.blit(pipe_body_img, (0, i * PipePair.PIECE_HEIGHT))
        top_pipe_end_y = self.top_height_px
        self.image.blit(pipe_end_img, (0, top_pipe_end_y))

        # compensate for added end pieces
        self.top_pieces += 1
        self.bottom_pieces += 1

        # for collision detection
        self.mask = pygame.mask.from_surface(self.image)

    @property
    def top_height_px(self) -> int:
        """Get the top pipe's height, in pixels."""
        return self.top_pieces * PipePair.PIECE_HEIGHT

    @property
    def bottom_height_px(self) -> int:
        """Get the bottom pipe's height, in pixels."""
        return self.bottom_pieces * PipePair.PIECE_HEIGHT

    @property
    def visible(self) -> bool:
        """Get whether this PipePair on screen, visible to the player."""
        return -PipePair.WIDTH < self.x < WIN_WIDTH

    @property
    def rect(self) -> Rect:
        """Get the Rect which contains this PipePair."""
        return Rect(self.x, 0, PipePair.WIDTH, PipePair.PIECE_HEIGHT)

    def update(self, delta_frames: int = 1) -> None:
        """Update the PipePair's position."""
        self.x -= ANIMATION_SPEED * _frames_to_msec(delta_frames)

    def collides_with(self, bird: Bird) -> tuple[int, int] | None:
        """Get whether the bird collides with a pipe in this PipePair."""
        return pygame.sprite.collide_mask(self, bird)


def load_images() -> dict[str, Surface]:
    """Load all images required by the game and return a dict of them."""

    def _load_image(img_file_name: str) -> Surface:
        return pygame.image.load(IMAGES_DIR / img_file_name).convert_alpha()

    return {
        "background": _load_image("background.png"),
        "pipe-end": _load_image("pipe_end.png"),
        "pipe-body": _load_image("pipe_body.png"),
        "bird-wingup": _load_image("bird_wing_up.png"),
        "bird-wingdown": _load_image("bird_wing_down.png"),
    }


def _countdown(
    display_surface: Surface,
    background: Surface,
    font: pygame.font.Font,
    bird: Bird,
    seconds: int = 3,
) -> bool:
    """Show a countdown while still handling events. Return False if the window was closed."""
    clock = pygame.time.Clock()
    start = pygame.time.get_ticks()
    while (remaining := seconds * 1000 - (pygame.time.get_ticks() - start)) > 0:
        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            return False
        for x in (0, WIN_WIDTH / 2):
            display_surface.blit(background, (x, 0))
        display_surface.blit(bird.image, bird.rect)
        label = font.render(str(math.ceil(remaining / 1000)), True, (255, 255, 255))
        display_surface.blit(label, (WIN_WIDTH / 2 - label.get_width() / 2, WIN_HEIGHT / 3))
        pygame.display.flip()
        clock.tick(FPS)
    return True


def _frames_to_msec(frames: int, fps: int = FPS) -> float:
    """Convert frames to milliseconds at the specified framerate."""
    return 1000.0 * frames / fps


def _msec_to_frames(milliseconds: int, fps: int = FPS) -> float:
    return fps * milliseconds / 1000.0


def main() -> None:
    """Application's entry point."""

    pygame.init()
    time.sleep(3)

    display_surface = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    pygame.display.set_caption("Pygame Flappy Bird")

    clock = pygame.time.Clock()
    score_font = pygame.font.SysFont(None, 32, bold=True)  # default font
    images = load_images()

    # the bird stays in the same x position, so bird.x is a constant
    # center bird on screen
    bird = Bird(
        50, int(WIN_HEIGHT / 2 - Bird.HEIGHT / 2), (images["bird-wingup"], images["bird-wingdown"])
    )

    pipes = deque()
    if not _countdown(display_surface, images["background"], score_font, bird):
        pygame.quit()
        return

    frame_clock = 0  # this counter is only incremented if the game isn't paused
    score = 0
    done = paused = False
    while not done:
        clock.tick(FPS)

        # Handle this 'manually'.  If we used pygame.time.set_timer(),
        # pipe addition would be messed up when paused.
        if not (paused or frame_clock % _msec_to_frames(PipePair.ADD_INTERVAL)):
            pp = PipePair(images["pipe-end"], images["pipe-body"])
            pipes.append(pp)

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                done = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                bird.flap()

        if paused:
            continue  # don't draw anything

        # check for collisions
        pipe_collision = any(p.collides_with(bird) for p in pipes)
        if pipe_collision or 0 >= bird.y or bird.y >= WIN_HEIGHT - Bird.HEIGHT:
            done = True

        for x in (0, WIN_WIDTH / 2):
            display_surface.blit(images["background"], (x, 0))

        while pipes and not pipes[0].visible:
            pipes.popleft()

        for p in pipes:
            p.update()
            display_surface.blit(p.image, p.rect)

        bird.update()
        display_surface.blit(bird.image, bird.rect)

        # update and display score
        for p in pipes:
            if p.x + PipePair.WIDTH < bird.x and not p.score_counted:
                score += 1
                p.score_counted = True

        score_surface = score_font.render(str(score), True, (255, 255, 255))
        score_x = WIN_WIDTH / 2 - score_surface.get_width() / 2
        display_surface.blit(score_surface, (score_x, PipePair.PIECE_HEIGHT))

        pygame.display.flip()
        frame_clock += 1
    print("Game over! Score: %i" % score)
    pygame.quit()


if __name__ == "__main__":
    main()

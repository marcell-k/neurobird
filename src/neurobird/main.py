"""Pygame renderer and human-play entry point for the headless game."""

import argparse
import math
import random
from pathlib import Path

import pygame
from pygame import Surface

from neurobird import sim
from neurobird.agent import should_flap
from neurobird.nn import NeuralNetwork

IMAGES_DIR = Path("images")
WHITE = (255, 255, 255)


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


def _draw_pipe(surface: Surface, images: dict[str, Surface], pipe: sim.Pipe) -> None:
    """Draw both halves of a pipe from its gap bounds."""
    x = round(pipe.x)
    body, end = images["pipe-body"], images["pipe-end"]
    for y in range(0, pipe.gap_top - sim.PIECE_HEIGHT, sim.PIECE_HEIGHT):
        surface.blit(body, (x, y))
    surface.blit(end, (x, pipe.gap_top - sim.PIECE_HEIGHT))
    surface.blit(end, (x, pipe.gap_bottom))
    for y in range(pipe.gap_bottom + sim.PIECE_HEIGHT, sim.WORLD_HEIGHT, sim.PIECE_HEIGHT):
        surface.blit(body, (x, y))


def _draw(
    display: Surface, images: dict[str, Surface], font: pygame.font.Font, game: sim.Game
) -> None:
    """Draw the current state of the game."""
    for x in (0, sim.WORLD_WIDTH // 2):
        display.blit(images["background"], (x, 0))
    for pipe in game.pipes:
        _draw_pipe(display, images, pipe)
    wings_up = (game.frames * sim.STEP_MSEC) % 500 >= 250  # noqa: PLR2004
    bird_image = images["bird-wingup" if wings_up else "bird-wingdown"]
    display.blit(bird_image, (sim.BIRD_X, round(game.y)))
    label = font.render(str(game.score), antialias=True, color=WHITE)
    display.blit(label, ((sim.WORLD_WIDTH - label.get_width()) // 2, sim.PIECE_HEIGHT))


def _countdown(
    display: Surface,
    images: dict[str, Surface],
    font: pygame.font.Font,
    game: sim.Game,
    seconds: int = 3,
) -> bool:
    """Show a countdown while still handling events. Return False if the window was closed."""
    clock = pygame.time.Clock()
    start = pygame.time.get_ticks()
    while (remaining := seconds * 1000 - (pygame.time.get_ticks() - start)) > 0:
        if any(e.type == pygame.QUIT for e in pygame.event.get()):
            return False
        _draw(display, images, font, game)
        label = font.render(str(math.ceil(remaining / 1000)), antialias=True, color=WHITE)
        display.blit(label, ((sim.WORLD_WIDTH - label.get_width()) // 2, sim.WORLD_HEIGHT // 3))
        pygame.display.flip()
        clock.tick(sim.FPS)
    return True


def main() -> None:
    """Application's entry point: play as a human, or watch a saved network with --ai."""
    parser = argparse.ArgumentParser(description="Pygame Flappy Bird")
    parser.add_argument(
        "--ai", type=Path, default=None, metavar="FILE", help="watch a trained network play"
    )
    parser.add_argument("--difficulty", choices=sorted(sim.CONFIGS), default="easy")
    args = parser.parse_args()
    brain = NeuralNetwork.load(args.ai) if args.ai else None
    pygame.init()
    display = pygame.display.set_mode((sim.WORLD_WIDTH, sim.WORLD_HEIGHT))
    pygame.display.set_caption("Pygame Flappy Bird")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32, bold=True)
    images = load_images()
    game = sim.Game(random.Random(), sim.CONFIGS[args.difficulty])

    if not _countdown(display, images, font, game):
        pygame.quit()
        return

    running = True
    while running and game.alive:
        clock.tick(sim.FPS)
        flap = False
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                flap = True
        if brain is not None:
            flap = should_flap(brain, game)
        game.step(flap=flap)
        _draw(display, images, font, game)
        pygame.display.flip()

    print(f"Game over! Score: {game.score}")  # noqa: T201
    pygame.quit()


if __name__ == "__main__":
    main()

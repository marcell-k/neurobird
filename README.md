# neurobird

Flappy Bird in Pygame, plus a neural network that learns to play it through neuroevolution.

## Setup

```sh
uv sync
```

## Usage

```sh
uv run neurobird                          # play (space to flap)
uv run neurobird --difficulty hard        # harder: speed, gap and spacing ramp up
uv run neurobird-train --generations 100  # train, saves best.npz
uv run neurobird --ai best.npz            # watch the trained network play
```

Both commands accept `--difficulty easy|hard`. Train and watch on the same difficulty.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clt_playground.core import simulate_clt
from src.clt_playground.plots import (
    create_population_figure,
    create_progression_figure,
    create_sample_means_histogram,
)


SCREENSHOT_DIR = ROOT / "assets" / "screenshots"


def save_figure(figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    figure.clf()


def main() -> None:
    simulation = simulate_clt(
        distribution_name="lognormal",
        params={"mu": 0.0, "sigma": 0.9},
        sample_size=30,
        num_simulations=3000,
        seed=42,
    )

    save_figure(create_population_figure(simulation), SCREENSHOT_DIR / "population.png")
    save_figure(create_sample_means_histogram(simulation, show_theoretical_reference=True), SCREENSHOT_DIR / "means_histogram.png")
    save_figure(
        create_progression_figure(
            distribution_name="lognormal",
            params={"mu": 0.0, "sigma": 0.9},
            sample_sizes=[1, 5, 15, 30],
            num_simulations=2500,
            seed=42,
        ),
        SCREENSHOT_DIR / "progression.png",
    )


if __name__ == "__main__":
    main()

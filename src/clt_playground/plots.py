from __future__ import annotations

from math import pi

import matplotlib.pyplot as plt
import numpy as np

from src.clt_playground.core import SimulationResult, simulate_clt


COLORS = {
    "population": "#1d3557",
    "means": "#e76f51",
    "reference": "#2a9d8f",
    "grid": "#d9d9d9",
    "background": "#f8fafc",
}


def _normal_pdf(x_values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if np.isclose(std, 0.0):
        return np.zeros_like(x_values)
    coefficient = 1.0 / (std * np.sqrt(2.0 * pi))
    exponent = -0.5 * ((x_values - mean) / std) ** 2
    return coefficient * np.exp(exponent)


def _apply_style(axis: plt.Axes) -> None:
    axis.set_facecolor(COLORS["background"])
    axis.grid(True, axis="y", alpha=0.25, color=COLORS["grid"])
    for spine in axis.spines.values():
        spine.set_visible(False)


def create_population_figure(simulation: SimulationResult) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    _apply_style(axis)

    if simulation.distribution_name == "bernoulli":
        values = np.array([0.0, 1.0])
        probabilities = np.array(
            [
                np.mean(simulation.population_sample == 0.0),
                np.mean(simulation.population_sample == 1.0),
            ]
        )
        axis.bar(values, probabilities, color=COLORS["population"], width=0.55, alpha=0.9)
        axis.set_xticks(values)
    else:
        axis.hist(simulation.population_sample, bins=45, density=True, color=COLORS["population"], alpha=0.85)

    axis.axvline(simulation.population_mean, color=COLORS["reference"], linestyle="--", linewidth=2)
    axis.set_title("Population distribution")
    axis.set_xlabel("Value")
    axis.set_ylabel("Density" if simulation.distribution_name != "bernoulli" else "Probability")
    return figure


def create_sample_means_figure(simulation: SimulationResult) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    _apply_style(axis)

    simulation_ids = np.arange(1, len(simulation.sample_means) + 1)
    axis.scatter(
        simulation_ids,
        simulation.sample_means,
        s=12,
        alpha=0.5,
        color=COLORS["means"],
        edgecolors="none",
    )
    axis.axhline(simulation.population_mean, color=COLORS["reference"], linestyle="--", linewidth=2)
    axis.set_title("Repeated sample means")
    axis.set_xlabel("Simulation")
    axis.set_ylabel("Sample mean")
    return figure


def create_sample_means_histogram(simulation: SimulationResult, show_theoretical_reference: bool) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    _apply_style(axis)

    axis.hist(simulation.sample_means, bins=40, density=True, color=COLORS["means"], alpha=0.82)
    axis.set_title("Histogram of sample means")
    axis.set_xlabel("Sample mean")
    axis.set_ylabel("Density")

    if show_theoretical_reference:
        x_values = np.linspace(simulation.sample_means.min(), simulation.sample_means.max(), 300)
        density = _normal_pdf(
            x_values,
            mean=simulation.population_mean,
            std=simulation.theoretical_standard_error,
        )
        axis.plot(x_values, density, color=COLORS["reference"], linewidth=2.5, label="CLT normal approximation")
        axis.legend(frameon=False)

    return figure


def create_progression_figure(
    distribution_name: str,
    params: dict[str, float],
    sample_sizes: list[int],
    num_simulations: int,
    seed: int,
) -> plt.Figure:
    figure, axes = plt.subplots(1, len(sample_sizes), figsize=(5 * len(sample_sizes), 4), sharey=True)
    if len(sample_sizes) == 1:
        axes = np.array([axes])

    simulations = [
        simulate_clt(
            distribution_name=distribution_name,
            params=params,
            sample_size=size,
            num_simulations=num_simulations,
            seed=seed + index,
        )
        for index, size in enumerate(sample_sizes)
    ]

    global_min = min(np.min(item.sample_means) for item in simulations)
    global_max = max(np.max(item.sample_means) for item in simulations)

    for axis, simulation in zip(axes, simulations):
        _apply_style(axis)
        axis.hist(
            simulation.sample_means,
            bins=35,
            density=True,
            range=(global_min, global_max),
            color=COLORS["means"],
            alpha=0.82,
        )

        x_values = np.linspace(global_min, global_max, 300)
        density = _normal_pdf(
            x_values,
            mean=simulation.population_mean,
            std=simulation.theoretical_standard_error,
        )
        axis.plot(x_values, density, color=COLORS["reference"], linewidth=2)
        axis.set_title(f"n = {simulation.sample_size}")
        axis.set_xlabel("Sample mean")

    axes[0].set_ylabel("Density")
    figure.suptitle("How the sample-means distribution becomes more Gaussian", fontsize=14, y=1.02)
    figure.tight_layout()
    return figure

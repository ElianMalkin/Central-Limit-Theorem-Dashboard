from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    distribution_name: str
    params: dict[str, float]
    sample_size: int
    num_simulations: int
    seed: int
    population_sample: np.ndarray
    population_mean: float
    population_std: float
    theoretical_standard_error: float
    raw_samples: np.ndarray
    sample_means: np.ndarray
    sample_means_skewness: float


def available_distributions() -> dict[str, dict[str, object]]:
    return {
        "exponential": {"label": "Exponential", "discrete": False},
        "uniform": {"label": "Uniform", "discrete": False},
        "bernoulli": {"label": "Bernoulli", "discrete": True},
        "lognormal": {"label": "Lognormal", "discrete": False},
        "bimodal": {"label": "Bimodal Gaussian mixture", "discrete": False},
    }


def _distribution_moments(distribution_name: str, params: dict[str, float]) -> tuple[float, float]:
    if distribution_name == "exponential":
        rate = params["rate"]
        mean = 1.0 / rate
        variance = 1.0 / (rate**2)
    elif distribution_name == "uniform":
        low = params["low"]
        high = params["high"]
        mean = 0.5 * (low + high)
        variance = ((high - low) ** 2) / 12.0
    elif distribution_name == "bernoulli":
        probability = params["p"]
        mean = probability
        variance = probability * (1.0 - probability)
    elif distribution_name == "lognormal":
        mu = params["mu"]
        sigma = params["sigma"]
        mean = np.exp(mu + 0.5 * sigma**2)
        variance = (np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2)
    elif distribution_name == "bimodal":
        weight_left = params["weight_left"]
        mean_left = params["mean_left"]
        std_left = params["std_left"]
        mean_right = params["mean_right"]
        std_right = params["std_right"]

        mean = weight_left * mean_left + (1.0 - weight_left) * mean_right
        second_moment = weight_left * (std_left**2 + mean_left**2) + (1.0 - weight_left) * (
            std_right**2 + mean_right**2
        )
        variance = second_moment - mean**2
    else:
        raise ValueError(f"Unsupported distribution: {distribution_name}")

    return float(mean), float(np.sqrt(variance))


def _sample_distribution(
    rng: np.random.Generator, distribution_name: str, params: dict[str, float], size: int | tuple[int, ...]
) -> np.ndarray:
    if distribution_name == "exponential":
        return rng.exponential(scale=1.0 / params["rate"], size=size)

    if distribution_name == "uniform":
        return rng.uniform(low=params["low"], high=params["high"], size=size)

    if distribution_name == "bernoulli":
        return rng.binomial(n=1, p=params["p"], size=size).astype(float)

    if distribution_name == "lognormal":
        return rng.lognormal(mean=params["mu"], sigma=params["sigma"], size=size)

    if distribution_name == "bimodal":
        weight_left = params["weight_left"]
        selectors = rng.random(size=size) < weight_left
        samples = np.empty(size, dtype=float)
        samples[selectors] = rng.normal(loc=params["mean_left"], scale=params["std_left"], size=int(selectors.sum()))
        samples[~selectors] = rng.normal(
            loc=params["mean_right"],
            scale=params["std_right"],
            size=int((~selectors).sum()),
        )
        return samples

    raise ValueError(f"Unsupported distribution: {distribution_name}")


def _skewness(values: np.ndarray) -> float:
    centered = values - np.mean(values)
    std = np.std(values)
    if np.isclose(std, 0.0):
        return 0.0
    return float(np.mean(centered**3) / (std**3))


def simulate_clt(
    distribution_name: str,
    params: dict[str, float],
    sample_size: int,
    num_simulations: int,
    seed: int,
) -> SimulationResult:
    rng = np.random.default_rng(seed)
    population_sample = _sample_distribution(rng, distribution_name, params, size=30000)
    raw_samples = _sample_distribution(rng, distribution_name, params, size=(num_simulations, sample_size))
    sample_means = raw_samples.mean(axis=1)
    population_mean, population_std = _distribution_moments(distribution_name, params)

    return SimulationResult(
        distribution_name=distribution_name,
        params=params,
        sample_size=sample_size,
        num_simulations=num_simulations,
        seed=seed,
        population_sample=population_sample,
        population_mean=population_mean,
        population_std=population_std,
        theoretical_standard_error=population_std / np.sqrt(sample_size),
        raw_samples=raw_samples,
        sample_means=sample_means,
        sample_means_skewness=_skewness(sample_means),
    )


def build_progression_sizes(sample_size: int) -> list[int]:
    candidates = [1, max(2, sample_size // 5), max(2, sample_size // 2), sample_size]
    return sorted(set(candidates))

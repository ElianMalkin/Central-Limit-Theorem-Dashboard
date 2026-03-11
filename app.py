from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.clt_playground.core import (
    available_distributions,
    build_progression_sizes,
    build_sample_preview,
    simulate_clt,
)
from src.clt_playground.plots import (
    create_population_figure,
    create_progression_figure,
    create_sample_means_figure,
    create_sample_means_histogram,
)


st.set_page_config(
    page_title="Central Limit Theorem Playground",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _distribution_label_lookup() -> dict[str, str]:
    return {key: config["label"] for key, config in available_distributions().items()}


def _sidebar_distribution_parameters(distribution_name: str) -> dict[str, float]:
    params: dict[str, float] = {}

    if distribution_name == "exponential":
        params["rate"] = st.sidebar.slider("Rate (lambda)", 0.2, 5.0, 1.0, 0.1)
    elif distribution_name == "uniform":
        low = st.sidebar.slider("Lower bound", -10.0, 10.0, 0.0, 0.5)
        high_default = max(low + 0.5, 5.0)
        high = st.sidebar.slider("Upper bound", low + 0.5, 20.0, high_default, 0.5)
        params["low"] = low
        params["high"] = high
    elif distribution_name == "bernoulli":
        params["p"] = st.sidebar.slider("Success probability p", 0.05, 0.95, 0.35, 0.05)
    elif distribution_name == "lognormal":
        params["mu"] = st.sidebar.slider("Log-mean (mu)", -1.0, 2.0, 0.0, 0.1)
        params["sigma"] = st.sidebar.slider("Log-standard deviation (sigma)", 0.2, 1.6, 0.8, 0.1)
    elif distribution_name == "bimodal":
        params["weight_left"] = st.sidebar.slider("Left mode weight", 0.1, 0.9, 0.5, 0.05)
        params["mean_left"] = st.sidebar.slider("Left mean", -6.0, 0.0, -2.5, 0.1)
        params["std_left"] = st.sidebar.slider("Left std", 0.2, 3.0, 1.0, 0.1)
        params["mean_right"] = st.sidebar.slider("Right mean", 0.0, 8.0, 2.5, 0.1)
        params["std_right"] = st.sidebar.slider("Right std", 0.2, 3.0, 1.0, 0.1)

    return params


def _format_distribution_parameters(params: dict[str, float]) -> str:
    if not params:
        return "Default parameters"
    return ", ".join(f"{key}={value:.3g}" for key, value in params.items())


def main() -> None:
    st.title("Central Limit Theorem Playground")
    st.caption(
        "Explore how averages behave when we repeatedly sample from skewed, discrete, and bimodal populations."
    )

    labels = _distribution_label_lookup()
    distribution_name = st.sidebar.selectbox(
        "Distribution type",
        list(labels.keys()),
        format_func=lambda key: labels[key],
    )
    params = _sidebar_distribution_parameters(distribution_name)

    sample_size = st.sidebar.slider("Sample size (n)", 1, 500, 30, 1)
    num_simulations = st.sidebar.slider("Number of simulations", 100, 10000, 3000, 100)
    random_seed = st.sidebar.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)

    st.sidebar.markdown("### Display options")
    show_raw_data = st.sidebar.checkbox("Display raw simulation data", value=False)
    show_means_histogram = st.sidebar.checkbox("Display histogram of sample means", value=True)
    show_theoretical_reference = st.sidebar.checkbox("Display theoretical mean and standard error", value=True)

    simulation = simulate_clt(
        distribution_name=distribution_name,
        params=params,
        sample_size=sample_size,
        num_simulations=num_simulations,
        seed=int(random_seed),
    )

    population_df = pd.DataFrame({"value": simulation.population_sample})
    sample_means_df = pd.DataFrame(
        {
            "simulation": np.arange(1, len(simulation.sample_means) + 1),
            "sample_mean": simulation.sample_means,
        }
    )

    st.info(
        "Even when the original distribution is skewed, sample means become approximately "
        "normal as sample size increases."
    )

    metric_columns = st.columns(4 if show_theoretical_reference else 3)
    metric_columns[0].metric("Population mean", f"{simulation.population_mean:.4f}")
    metric_columns[1].metric("Mean of sample means", f"{simulation.sample_means.mean():.4f}")
    metric_columns[2].metric("Empirical std of sample means", f"{simulation.sample_means.std(ddof=1):.4f}")
    if show_theoretical_reference:
        metric_columns[3].metric("Theoretical standard error", f"{simulation.theoretical_standard_error:.4f}")

    summary_columns = st.columns([1.3, 1])
    with summary_columns[0]:
        st.pyplot(create_population_figure(simulation), clear_figure=True, use_container_width=True)
    with summary_columns[1]:
        st.subheader("Experiment setup")
        st.markdown(
            "\n".join(
                [
                    f"- **Distribution:** {labels[distribution_name]}",
                    f"- **Parameters:** {_format_distribution_parameters(params)}",
                    f"- **Sample size:** {sample_size}",
                    f"- **Repeated samples:** {num_simulations}",
                    f"- **Population std:** {simulation.population_std:.4f}",
                    f"- **Sample-means skewness:** {simulation.sample_means_skewness:.4f}",
                ]
            )
        )
        st.caption(
            "As n grows, the sample-means skewness usually shrinks toward zero, which is one way to see the CLT in action."
        )

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.pyplot(create_sample_means_figure(simulation), clear_figure=True, use_container_width=True)
    with chart_columns[1]:
        if show_means_histogram:
            st.pyplot(
                create_sample_means_histogram(simulation, show_theoretical_reference),
                clear_figure=True,
                use_container_width=True,
            )
        else:
            st.subheader("Histogram hidden")
            st.write("Enable the histogram toggle in the sidebar to inspect the distribution of sample means.")

    progression_sizes = build_progression_sizes(sample_size)
    progression_note = ", ".join(str(size) for size in progression_sizes)
    st.pyplot(
        create_progression_figure(
            distribution_name=distribution_name,
            params=params,
            sample_sizes=progression_sizes,
            num_simulations=min(num_simulations, 4000),
            seed=int(random_seed),
        ),
        clear_figure=True,
        use_container_width=True,
    )
    st.caption(f"Comparison panel for n = {progression_note}.")

    if show_raw_data:
        preview_df = build_sample_preview(simulation.raw_samples, limit=10)
        with st.expander("Raw simulation preview", expanded=False):
            st.write("Population sample preview")
            st.dataframe(population_df.head(15), use_container_width=True)
            st.write("First repeated samples")
            st.dataframe(preview_df, use_container_width=True)
            st.write("Sample means preview")
            st.dataframe(sample_means_df.head(20), use_container_width=True)

    with st.expander("Math behind the app", expanded=False):
        st.markdown(
            r"""
For independent observations \(X_1, X_2, \dots, X_n\) with finite mean \(\mu\) and standard deviation \(\sigma\),
the sample mean is

\[
\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i
\]

and the Central Limit Theorem says that, for large \(n\),

\[
\bar{X}_n \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right).
\]

That is why the sample-means histogram tends to look more Gaussian even when the original population is skewed or discrete.
            """
        )


if __name__ == "__main__":
    main()


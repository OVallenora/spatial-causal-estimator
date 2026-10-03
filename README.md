# Spatial Causal Inference Simulator

Lightweight Python simulator that estimates the average treatment effect (ATE) of geographic interventions using spatial clustering and Monte Carlo bootstrapping. 

This project bridges unsupervised machine learning with causal inference. It utilizes **K-means spatial clustering** to partition geographic zones and evaluates statistical robustness across **2,000 Monte Carlo simulation loops**.

## Overview

In observational geographic data, estimating the true causal impact of a localized intervention (e.g., a regional pricing change, a new algorithmic dispatch protocol) is challenging due to spatial confounding. 

This script:
1. Generates synthetic geographic baseline data.
2. Applies unsupervised **K-means clustering** to partition the coordinates into representative spatial zones.
3. Simulates a localized "treatment" applied only to specific clusters.
4. Uses **Monte Carlo bootstrapping** to robustly estimate the ATE and calculate the 95% Confidence Interval, proving statistical significance against natural variance.

## Key Features

* **Spatial Treatment Assignment:** Uses `scikit-learn`'s K-Means to deterministically assign treatment/control groups based on physical geography rather than random uniform distribution.
* **Causal Impact Estimation:** Isolates the true causal effect of an intervention from noisy baseline metrics.
* **Robust Statistical Evaluation:** Bootstraps the ATE over thousands of iterations to ensure the estimated impact is statistically significant (validated via p-values and confidence intervals).

## Requirements

This project relies on standard scientific Python libraries.

```bash
pip install numpy scipy scikit-learn

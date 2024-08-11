# -*- coding: utf-8 -*-
"""topsis_gpu_analysis.py

Author: Mandeep Kumar
Email: themandeepkumar@gmail.com

**TOPSIS Analysis of GPU Compute Instances for HPC and AI in the Cloud**

This Jupyter notebook contains the implementation of the Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS), a Multi Criteria Decision Making (MCDM) method for evaluating and ranking GPU compute instances for HPC and AI from various cloud providers. The notebook guides you through the process of data preparation, criteria weighting, and application of the TOPSIS algorithm. Additionally, it includes sensitivity analysis to explore the impact of varying criteria weights, bootstrap analysis to assess the stability of the rankings, and non-parametric tests to evaluate the consistency of the results. The notebook is designed to be a comprehensive tool for researchers and practitioners looking to make informed decisions about GPU compute instance selection for HPC and AI in cloud computing environments.
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, friedmanchisquare
import seaborn as sns

# Set display options to show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# Define the TOPSIS function
def topsis(raw_data, weights, benefit_categories):
    m, n = raw_data.shape
    # Normalize the raw data
    divisors = np.sqrt(np.sum(raw_data ** 2, axis=0))
    normalized_data = raw_data / divisors

    # Apply weights
    weighted_data = normalized_data * weights

    # Determine Ideal and Negative Ideal Solutions
    ideal_solution = np.zeros(n)
    negative_ideal_solution = np.zeros(n)
    for j in range(n):
        if j in benefit_categories:
            ideal_solution[j] = np.max(weighted_data[:, j])
            negative_ideal_solution[j] = np.min(weighted_data[:, j])
        else:
            ideal_solution[j] = np.min(weighted_data[:, j])
            negative_ideal_solution[j] = np.max(weighted_data[:, j])

    # Calculate distances
    dist_to_ideal = np.sqrt(np.sum((weighted_data - ideal_solution) ** 2, axis=1))
    dist_to_negative_ideal = np.sqrt(np.sum((weighted_data - negative_ideal_solution) ** 2, axis=1))

    # Calculate TOPSIS scores
    scores = dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)
    return scores

"""**Identification of Criteria and Weights**

The initial phase of the TOPSIS methodology involves determining and defining the criteria.

**Requirements:**

For this analysis, the following information is consistently provided:

The scores for each alternative across various categories.

*   The scores for each alternative across various categories.
*   The importance or weights assigned to each category.

Note: Categories may be classified as either beneficial, where maximizing their contribution is desired, or as cost categories, where minimizing their impact is preferred.
"""

# Identification of Criteria and Weights
categories = np.array(["Number of GPU (G)", "Number of Physical CPU Cores (C)", "GPU Memory (GM)", "CPU Memory (CM)", "GPU FP64 Performance (GP)", "On-Demand Hourly Cost (HC)"])
alternatives = np.array(["AWS p5.48xlarge", "AWS p4de.24xlarge", "AWS p4d.24xlarge", "GCP a3-highgpu-8g", "GCP a2-ultragpu-8g", "GCP a2-ultragpu-4g", "GCP a2-ultragpu-2g", "GCP a2-ultragpu-1g", "GCP a2-megagpu-16g", "GCP a2-highgpu-8g", "GCP a2-highgpu-4g", "GCP a2-highgpu-2g", "GCP a2-highgpu-1g", "Azure ND96isr_H100_v5", "Azure NC80adis_H100_v5", "Azure NC40ads_H100_v5", "Azure ND96amsr_A100_v4", "Azure ND96asr_A100_v4", "Azure NC96ads_A100_v4", "Azure NC48ads_A100_v4", "Azure NC24ads_A100_v4", "OCI BM.GPU.H100.8", "OCI BM.GPU.A100-v2.8", "OCI BM.GPU4.8"])
raw_data = np.array([
    [8, 96, 640, 2048, 272, 98.32],
    [8, 48, 640, 1152, 77.6, 53.09472],
    [8, 48, 320, 1152, 77.6, 32.7726],
    [8, 104, 640, 1872, 272, 88.5057808],
    [8, 48, 640, 1360, 77.6, 40.5503836],
    [4, 24, 320, 680, 38.8, 20.2751918],
    [2, 12, 160, 340, 19.4, 10.137589],
    [1, 6, 80, 170, 9.7, 5.06879452],
    [16, 48, 640, 1360, 155.2, 55.7395068],
    [8, 48, 320, 680, 77.6, 29.3870822],
    [4, 24, 160, 340, 38.8, 14.6935342],
    [2, 12, 80, 170, 19.4, 7.34676712],
    [1, 6, 40, 85, 9.7, 3.67338356],
    [8, 96, 640, 1900, 272, 98.32],
    [2, 80, 188, 640, 60, 13.96],
    [1, 40, 94, 320, 30, 6.98],
    [8, 96, 640, 1900, 77.6, 32.77],
    [8, 96, 320, 900, 77.6, 27.197],
    [4, 96, 320, 880, 38.8, 14.692],
    [2, 48, 160, 440, 19.4, 7.346],
    [1, 24, 80, 220, 9.7, 3.673],
    [8, 112, 640, 2048, 272, 80],
    [8, 128, 640, 2048, 77.6, 32],
    [8, 64, 320, 2048, 77.6, 24.4],
])

initial_weights = np.array([0.15, 0.10, 0.15, 0.10, 0.25, 0.25])
benefit_categories = set([0, 1, 2, 3, 4])

# Display raw data and weights
raw_data_df = pd.DataFrame(data=raw_data, index=alternatives, columns=categories)
weights_df = pd.DataFrame(data=initial_weights, index=categories, columns=["Weights"])

print("Raw Data:")
display(raw_data_df)
print("Initial Weights:")
display(weights_df)

"""**Normalization of Data**

Normalization is essential to bring all criteria to a common scale, ensuring that each criterion contributes proportionally to the decision-making process. This step involves transforming the raw data for each criterion into a dimensionless value between 0 and 1. Various normalization techniques, such as min-max normalization or z-score normalization, can be applied depending on the nature of the data.

"""

# Normalize the raw data
m, n = raw_data.shape
divisors = np.empty(n)
for j in range(n):
    column = raw_data[:, j]
    divisors[j] = np.sqrt(column @ column)
normalized_data = raw_data / divisors

# Normalize the weights to ensure that they sum up to 1
weights = initial_weights / np.sum(initial_weights)

normalized_data_df = pd.DataFrame(data=normalized_data, index=alternatives, columns=categories)

print("Normalized Data:")
display(normalized_data_df)

"""The weights are normalized to ensure that they sum up to 1."""

# Weighted normalized decision matrix
weighted_data = normalized_data * weights

weighted_data_df = pd.DataFrame(data=weighted_data, index=alternatives, columns=categories)

print("Weighted Normalized Data:")
display(weighted_data_df)

"""**Determination of Ideal Solution and Negative Ideal Solution**

Ideal Solution and Negative Ideal Solution are key concepts used to evaluate alternatives based on their distance from these ideal points.
"""

# Determine the Ideal and Negative Ideal Solutions
a_pos = np.zeros(n)
a_neg = np.zeros(n)
for j in range(n):
    column = weighted_data[:, j]
    max_val = np.max(column)
    min_val = np.min(column)

    if j in benefit_categories:
        a_pos[j] = max_val
        a_neg[j] = min_val
    else:
        a_pos[j] = min_val
        a_neg[j] = max_val

ideal_df = pd.DataFrame(data=[a_pos, a_neg], index=["Ideal Solution", "Negative Ideal Solution"], columns=categories)
print("Ideal and Negative Ideal Solutions:")
display(ideal_df)

"""**Calculation of Similarity Scores**

The core of TOPSIS lies in the calculation of similarity scores for each alternative with respect to the ideal and negative ideal solutions. The ideal solution represents the maximum (or minimum, depending on the nature of the criterion) values for each criterion, while the negative ideal solution represents the minimum (or maximum) values.
"""

# Calculate the similarity scores
sp = np.zeros(m)
sn = np.zeros(m)
cs = np.zeros(m)

for i in range(m):
    diff_pos = weighted_data[i] - a_pos
    diff_neg = weighted_data[i] - a_neg
    sp[i] = np.sqrt(diff_pos @ diff_pos)
    sn[i] = np.sqrt(diff_neg @ diff_neg)
    cs[i] = sn[i] / (sp[i] + sn[i])

similarity_scores_df = pd.DataFrame(data=zip(sp, sn), index=alternatives, columns=["S+", "S-"])
print("Similarity Scores:")
display(similarity_scores_df)

"""**Ranking of Alternatives**

The final step involves ranking the alternatives based on their relative closeness to the ideal solution and distance from the anti-ideal solution.
"""

# Ranking of alternatives
initial_ranks = rankdata(-cs)
ranking_df = pd.DataFrame(data=zip(cs, initial_ranks), index=alternatives, columns=["TOPSIS Score", "Initial Rank"]).sort_values(by="Initial Rank")
print("Initial Ranking of Alternatives (Descending Order):")
display(ranking_df)

"""**Sensitivity Analysis**

Sensitivity analysis in the context of TOPSIS is performed to evaluate the robustness of the rankings by examining how variations in the criteria weights affect the results. This analysis ensures that the final rankings are reliable and not overly sensitive to changes in the assigned weights.
"""

# Sensitivity Analysis: Varying weights for each criterion
def sensitivity_analysis(raw_data, initial_weights, benefit_categories, alternatives):
    sensitivities = {}
    # Obtain initial ranking with current weights
    base_scores = topsis(raw_data, initial_weights, benefit_categories)
    base_ranking = rankdata(-base_scores)

    for i in range(len(initial_weights)):
        altered_weights = initial_weights.copy()
        for delta in np.linspace(-0.1, 0.1, 5):  # vary weights by ±10%
            if 0 <= initial_weights[i] + delta <= 1:
                altered_weights[i] = initial_weights[i] + delta
                # Ensure the weights sum to 1
                altered_weights /= np.sum(altered_weights)
                scores = topsis(raw_data, altered_weights, benefit_categories)
                ranking = rankdata(-scores)
                # Store the result using base_ranking as reference
                sensitivity_key = (i, delta)
                sensitivities[sensitivity_key] = pd.Series(ranking, index=alternatives)

    # Convert sensitivity results to DataFrame and align columns with initial ranking
    sensitivity_df = pd.DataFrame(sensitivities).T
    sensitivity_df.columns = alternatives  # Ensure correct column names for alternatives
    sensitivity_df.index.names = ['Criterion', 'Delta']
    sensitivity_df = sensitivity_df[ranking_df.sort_values("Initial Rank").index]

    return sensitivity_df

# Perform sensitivity analysis
sensitivity_df = sensitivity_analysis(raw_data, initial_weights, benefit_categories, alternatives)

print("Sensitivity Analysis:")
display(sensitivity_df)

"""**Bootstrapping Analysis**

Bootstrapping analysis is employed to evaluate the variability and stability of TOPSIS rankings by generating multiple resamples of the decision matrix and recalculating the TOPSIS scores for each resample. This approach helps in understanding the distribution of rankings and assessing the robustness of the decision outcomes.
"""

# Bootstrapping Analysis: Generating bootstrap samples and calculating TOPSIS scores
def bootstrap_analysis(raw_data, initial_weights, benefit_categories, num_samples=1000):
    m, n = raw_data.shape
    bootstrap_scores = np.zeros((num_samples, m))

    for i in range(num_samples):
        bootstrap_sample_indices = np.random.choice(m, m, replace=True)
        bootstrap_sample = raw_data[bootstrap_sample_indices]
        bootstrap_scores[i] = topsis(bootstrap_sample, initial_weights, benefit_categories)

    return bootstrap_scores

bootstrap_scores = bootstrap_analysis(raw_data, initial_weights, benefit_categories)

# Analyzing the bootstrap results
bootstrap_ranks = np.array([rankdata(-scores) for scores in bootstrap_scores])
bootstrap_mean_ranks = np.mean(bootstrap_ranks, axis=0)
bootstrap_rank_intervals = np.percentile(bootstrap_ranks, [2.5, 97.5], axis=0)

# Display bootstrap analysis results
bootstrap_df = pd.DataFrame({
    "TOPSIS Score": topsis(raw_data, initial_weights, benefit_categories),
    "Initial Rank": initial_ranks,
    "Mean Rank": bootstrap_mean_ranks,
    "2.5% Rank": bootstrap_rank_intervals[0],
    "97.5% Rank": bootstrap_rank_intervals[1]
}, index=alternatives).sort_values(by="Initial Rank")

print("Bootstrap Analysis Results (Descending Order):")
display(bootstrap_df)

"""**Non-Parametric Tests**

Non-parametric tests are utilized to evaluate the statistical significance of the differences in rankings obtained from the bootstrapping analysis. These tests do not assume a specific distribution for the data and are particularly useful for analyzing ordinal rankings.
"""

# Non-parametric Tests: Friedman Test
def friedman_test(bootstrap_ranks):
    # Perform the Friedman test
    stat, p = friedmanchisquare(*bootstrap_ranks.T)
    return stat, p

# Perform the Friedman test
stat, p = friedman_test(bootstrap_ranks)
print(f"Friedman Test Statistic: {stat}, p-value: {p}")

# Adding Friedman Test p-value to summary table
bootstrap_df["Friedman Test p-value"] = p
print("Final Summary Table with Friedman Test p-value (Descending Order):")
display(bootstrap_df)

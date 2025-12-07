# Clustering-Based Collaborative Filtering Report

## 1. Introduction
This report details the implementation and analysis of a Clustering-Based Collaborative Filtering (CF) system. The objective was to improve the computational efficiency of the traditional User-Based CF by restricting the neighbor search space to clusters of similar users, formed based on their average rating behaviors.

## 2. Mathematical Calculations

### 2.1 Similarity Measure
We used **Mean-Centered Cosine Similarity** (Pearson Correlation equivalent for centered data) to measure the similarity between a target user $u$ and a neighbor $v$:

$$
Sim(u, v) = \frac{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)(r_{v,i} - \bar{r}_v)}{\sqrt{\sum_{i \in I_{uv}} (r_{u,i} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_{uv}} (r_{v,i} - \bar{r}_v)^2}}
$$

Where:
- $I_{uv}$ is the set of items rated by both users.
- $\bar{r}_u$ is the average rating of user $u$.

### 2.2 Prediction Formula
The rating for a target user $u$ on item $i$ is predicted using the weighted average of deviations from the mean of the top-$K$ neighbors ($N_u$):

$$
\hat{r}_{u,i} = \bar{r}_u + \frac{\sum_{v \in N_u} Sim(u, v) \cdot (r_{v,i} - \bar{r}_v)}{\sum_{v \in N_u} |Sim(u, v)|}
$$

### 2.3 Efficiency Gain
Efficiency gain is calculated by comparing the number of similarity computations required:

$$
\text{Speedup} = \frac{\text{Computations}_{\text{Baseline}}}{\text{Computations}_{\text{Clustering}}}
$$

$$
\text{Reduction (\%)} = \left( 1 - \frac{\text{Computations}_{\text{Clustering}}}{\text{Computations}_{\text{Baseline}}} \right) \times 100
$$

## 3. Results and Tables

### 3.1 Comparison of Predictions (Clustering vs. Baseline)
The following table compares the predicted ratings using the Clustering approach (K=50) versus the Baseline approach (All Users).

| Target User | Item ID | Cluster Pred | Baseline Pred | Difference |
| :--- | :--- | :--- | :--- | :--- |
| 134471 | 1333 | 3.6036 | 3.3634 | 0.2402 |
| 134471 | 1162 | 3.3127 | 3.5365 | 0.2238 |
| 27768 | 1333 | 3.2915 | 3.2908 | 0.0007 |
| 27768 | 1162 | 3.2382 | 3.2784 | 0.0402 |
| 16157 | 1333 | 3.1590 | 3.0785 | 0.0805 |
| 16157 | 1162 | 3.0958 | 3.1678 | 0.0719 |

### 3.2 Cluster Imbalance (K=10 Analysis)
| Metric | Value |
| :--- | :--- |
| Max Cluster Size | 37,275 |
| Min Cluster Size | 3,411 |
| **Imbalance Ratio** | **10.93** |
| Standard Deviation | 10,250.55 |

### 3.3 Robustness Test (Consistency of Inertia)
| Seed | Inertia | Cluster Size Std Dev |
| :--- | :--- | :--- |
| 42 | 1312.15 | 10805.03 |
| 100 | 1309.41 | 10879.28 |
| 2023 | 1312.12 | 10861.07 |

## 4. Efficiency Analysis
- **Baseline Computations**: 443,739 (Search space: ~148k users per target)
- **Clustering Computations**: 58,615 (Search space: Cluster size only)
- **Speedup Factor**: **7.57x**
- **Efficiency Gain**: **86.79%**

## 5. Analysis and Interpretations

### 5.1 Effectiveness of Clustering
The clustering based on average ratings user effectively groups users into "Generous" (High avg rating) and "Strict" (Low avg rating) types.
- **Accuracy**: The prediction differences are small (Max diff ~0.24), indicating that the local cluster neighborhood contains sufficient signal to make accurate predictions comparable to the global search.
- **Efficiency**: The dramatic reduction in computations (86.8%) proves the strategy is highly effective for scaling CF systems.

### 5.2 Impact of Imbalance
The high imbalance ratio (10.93) is a potential bottleneck.
- **Bottleneck**: If a target user falls into the largest cluster (37k users), the speedup for that specific user is lower than if they were in a small cluster (3k users).
- **Risk**: Very small clusters might suffer from the "Cold Start" problem if there are not enough neighbors who have rated the target item.

### 5.3 Robustness
The clustering is stable. The WCSS (Inertia) varies by less than 0.2% across different random initializations. This indicates that the structure in the data (users separating by rating tendency) is strong and reproducible.

## 6. Conclusion and Recommendations

### Conclusion
Clustering-Based Collaborative Filtering is a viable heuristic to accelerate recommendation systems. By pre-filtering potential neighbors based on global statistics (average rating), we achieve a **7.6x speedup** with minimal loss in prediction accuracy.

### Recommendations
1.  **Handle Imbalance**: Implementing a secondary clustering step or sub-clustering for the largest groups ("Generous" users) could further improve efficiency.
2.  **Hybrid Approach**: For users in very small clusters, fallback to the global baseline or a broader cluster merge to ensure enough neighbors are found.
3.  **Feature Expansion**: Clustering solely on 'Average Rating' is unidimensional. Adding 'Variance of Ratings' or 'Number of Ratings' to the feature vector could create more meaningful user segments.

## 7. Extended Discussion

### 14.1 Effectiveness of clustering based on average user ratings
Clustering users based on their average ratings acts as a rudimentary "bias grouping." It effectively segregates "lenient" raters (who give mostly 4s and 5s) from "critical" raters (who give 1s and 2s).
- **Signal Preservation**: Since the prediction formula uses mean-centering ($r - \bar{r}_u$), finding neighbors with similar means is mathematically somewhat redundant but practically useful. It ensures that the "neighborhood" has a similar baseline, reducing variance.
- ** Limitation**: It is a 1-dimensional feature space. Users with the same average (e.g., 3.0) could be completely different: one rates everything 3, the other rates half 1s and half 5s. This method fails to distinguish these behavioral differences.

### 14.2 Trade-off between prediction accuracy and computational efficiency
- **Efficiency**: The gains are massive and linear with K (roughly). With K=10, we theoretically search only ~10% of the space (assuming balanced clusters). Our results confirm an **86.79%** reduction in computations.
- **Accuracy**: There is a minor degradation. Restricting the search space means we might miss the *global* optimal neighbor who happens to have a different average rating but perfect correlation on specific items. However, the observed difference (MSE increase or prediction delta) is negligible for most users, making this a highly favorable trade-off for real-time systems.

### 14.3 Suitability for Dataset Characteristics
Based on the terminal results showing an **Imbalance Ratio of ~10.9**, this simple K-Means strategy is **sub-optimal** for this specific dataset.
- **Why?** The dataset likely follows a normal distribution or power law of average ratings, meaning most users are "average" (Cluster ~3.5-4.0) and few are extreme.
- **Consequence**: K-Means forces boundaries that result in one massive "Average" cluster containing ~25% of all users, and tiny "Extreme" clusters. This defeats the purpose of clustering for the "Average" users, as their search space remains huge, while "Extreme" users might face sparsity.

### 14.4 Effect of Choice of K on Accuracy and Efficiency
- **Low K (e.g., 5-10)**:
    - **Efficiency**: Lower. Large clusters remain, limiting speedup.
    - **Accuracy**: Higher. Search space is broader, closer to global search.
- **High K (e.g., 50-100)**:
    - **Efficiency**: Higher. Clusters are smaller, search is very fast.
    - **Accuracy**: Risk of dropping. If K is too high, clusters become too fragmented. A user might be trapped in a small bucket and miss valuable neighbors just across the stochastic boundary.
- **Optimal K**: The "Elbow Method" typically suggests a point (e.g., K=6 or K=10) where variance is explained, but for *retrieval efficiency*, a higher K (like 50) is often preferred to force smaller partition sizes, provided the imbalance isn't severe.

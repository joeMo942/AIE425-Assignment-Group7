import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add project root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils import data_loader
from utils import data_loader
from utils.similarity import calculate_item_mean_centered_cosine
from utils.prediction import predict_item_based


def main():
    print("Loading data...")
    # 1. Load data
    df = data_loader.get_preprocessed_dataset()
    r_i = data_loader.get_item_avg_ratings()

    print("Computing item statistics...")
    # 2. Compute item statistics
    # 2.1 Num raters per item
    # Assuming columns are 'user_id', 'item_id', 'rating' - check specific column names if needed, 
    # but strictly following utils usually implies 'item_id' is present.
    # Let's inspect df structure if needed, but assuming standard names based on context.
    # Group by item_id
    item_stats = df.groupby('item')['rating'].agg(['count', 'std']).reset_index()
    item_stats.rename(columns={'count': 'num_raters', 'std': 'std_rating'}, inplace=True)
    
    # Fill NaN std with 0 (items with 1 rating have std=NaN)
    item_stats['std_rating'] = item_stats['std_rating'].fillna(0)

    # 2.2 Avg rating (reuse r_i)
    # Ensure column names match. r_i likely has 'item_id' and 'avg_rating' (or similar)
    # Let's merge.
    # r_i from utils usually returns DataFrame.
    # We need to make sure we merge correctly.
    
    # Merge r_i into item_stats
    # Note: r_i structure from utils: 'item_id', 'r_i_bar' usually? 
    # Let's assume 'item_id' is the key.
    feature_df = pd.merge(item_stats, r_i, on='item', how='inner')
    
    # We expect feature vector: [num_raters, avg_rating, std_rating]
    # Rename r_i column if necessary to 'avg_rating' for clarity in plotting/logic, 
    # but for feature vector we just need the values.
    # Let's identify the average rating column name. 
    # Usually it's the other column besides item.
    avg_col = [c for c in r_i.columns if c != 'item'][0]
    
    feature_cols = ['num_raters', avg_col, 'std_rating']
    X = feature_df[feature_cols].copy()
    
    print(f"Feature vector shape: {X.shape}")
    print(f"Features: {feature_cols}")

    # 3. Normalize features
    print("Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Verify normalization
    print(f"Mean after scaling: {np.mean(X_scaled, axis=0)}")
    print(f"Std after scaling: {np.std(X_scaled, axis=0)}")

    # 4. K-Means Clustering
    k_values = [5, 10, 15, 20, 30, 50]
    wcss_list = []
    silhouette_scores = []
    
    print("Starting clustering loop...")
    for k in k_values:
        print(f"  Clustering with K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # WCSS
        wcss_list.append(kmeans.inertia_)
        
        # Silhouette Score
        # optimize: sample if too large
        if len(X_scaled) > 40000:
            score = silhouette_score(X_scaled, labels, sample_size=40000, random_state=42)
        else:
            score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"    WCSS: {kmeans.inertia_:.4f}, Silhouette: {score:.4f}")

    # 5. Visualization
    print("Generating plots...")
    plt.figure(figsize=(15, 6))

    # Elbow Curve
    plt.subplot(1, 2, 1)
    plt.plot(k_values, wcss_list, 'bx-')
    plt.xlabel('k')
    plt.ylabel('WCSS')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)

    # Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for different k')
    plt.grid(True)

    # Define results directory relative to project root
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    output_plot_path = os.path.join(results_dir, 'clustering_metrics.png')
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Plots saved to {output_plot_path}")
    
    # --- TASK 5 & 6: Cluster Analysis ---
    print("\nStarting Cluster Analysis (K=10)...")
    optimal_k = 10
    kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    feature_df['cluster'] = kmeans_opt.fit_predict(X_scaled)
    
    # 5.1 & 5.2 & 5.3 & 5.4: Analyze characteristics
    cluster_stats = feature_df.groupby('cluster')['num_raters'].agg(['mean', 'count']).reset_index()
    cluster_stats = cluster_stats.sort_values('mean', ascending=False) # Sort by popularity
    
    print("\nCluster Statistics (Sorted by Avg Raters):")
    print(cluster_stats)
    
    # 6.3 Head vs Tail Analysis
    # Define Head (Top 20%) and Tail (Bottom 80%) based on popularity (num_raters)
    # Using 80th percentile as threshold for head items
    popularity_threshold = feature_df['num_raters'].quantile(0.8)
    feature_df['type'] = feature_df['num_raters'].apply(lambda x: 'Head' if x >= popularity_threshold else 'Tail')
    
    print(f"\nPopularity Threshold (top 20%): {popularity_threshold:.2f} raters")
    
    # Calculate distribution
    head_tail_dist = feature_df.groupby(['cluster', 'type']).size().unstack(fill_value=0)
    
    # --- Visualization ---
    print("Generating Analysis Plots...")
    plt.figure(figsize=(18, 6))

    # Plot 1: Item Distribution across Clusters
    plt.subplot(1, 3, 1)
    sns.countplot(data=feature_df, x='cluster', palette='viridis', hue='cluster', legend=False)
    plt.title('Item Distribution across Clusters')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Items')

    # Plot 2: Rater Distribution per Cluster (Boxplot)
    plt.subplot(1, 3, 2)
    sns.boxplot(data=feature_df, x='cluster', y='num_raters', palette='viridis', hue='cluster', legend=False)
    plt.yscale('log') # Log scale to handle long tail
    plt.title('Distribution of # Raters per Cluster (Log Scale)')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Raters (Log)')

    # Plot 3: Head vs Tail Composition
    plt.subplot(1, 3, 3)
    head_tail_dist.plot(kind='bar', stacked=True, ax=plt.gca(), color=['red', 'blue'])
    plt.title('Head (Popular) vs Tail (Niche) Composition')
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Items')
    plt.legend(title='Item Type')

    analysis_plot_path = os.path.join(results_dir, 'cluster_analysis.png')
    plt.tight_layout()
    plt.savefig(analysis_plot_path)
    print(f"Analysis plots saved to {analysis_plot_path}")
    
    # Save results to csv if needed? The task doesn't explicitly ask to save assignments to file yet, 
    # but usually it's good practice. For now, we focus on the metrics and plot.

    # --- TASK 7 & 8: Item-Based CF and Comparison ---
    print("\n--- Tasks 7 & 8: Item-Based CF within Clusters vs Baseline ---")
    
    # Imports done at top level
    
    # 1. Load Data/Setup
    target_users = data_loader.get_target_users()
    target_items = data_loader.get_target_items()
    
    print(f"Target Users: {target_users}")
    print(f"Target Items: {target_items}")
    
    # Need user means for Adjusted Cosine
    user_means_df = data_loader.get_user_avg_ratings()
    if 'user' in user_means_df.columns:
        user_means = dict(zip(user_means_df['user'], user_means_df['r_u_bar']))
    else:
        user_means = dict(zip(user_means_df.iloc[:,0], user_means_df.iloc[:,1]))

    # Need item means for fallback in prediction
    item_means_df = data_loader.get_item_avg_ratings()
    # Assuming 'item' and column for mean
    avg_col_name = [c for c in item_means_df.columns if c != 'item'][0]
    item_means = dict(zip(item_means_df['item'], item_means_df[avg_col_name]))

    # Build Dictionaries
    print("Building rating dictionaries...")
    # Item -> {User: Rating} (for similarity)
    item_user_ratings = df.groupby('item').apply(lambda x: dict(zip(x['user'], x['rating']))).to_dict()
    
    # User -> {Item: Rating} (for prediction utils)
    user_item_ratings = df.groupby('user').apply(lambda x: dict(zip(x['item'], x['rating']))).to_dict()

    # Pre-compute cluster items sets for fast lookup
    cluster_items = feature_df.groupby('cluster')['item'].apply(list).to_dict()
    


    print(f"\nComparing Predictions for Target Pairs...")
    print(f"{'User':<10} | {'Item':<10} | {'BasePred':<8} | {'ClusPred':<8} | {'Diff'}")
    print("-" * 60)

    # All items list for baseline
    all_candidate_items = list(item_user_ratings.keys())

    for u in target_users:
        for i in target_items:
            # Actual Rating
            actual = user_item_ratings.get(u, {}).get(i, None)
            
            # --- Baseline (Global) ---
            # Use all items as candidates
            t_item_ratings = item_user_ratings.get(i, {})
            similarities = []
            
            for cand_item in all_candidate_items:
                if cand_item == i:
                    continue
                cand_ratings = item_user_ratings.get(cand_item, {})
                
                try:
                    sim = calculate_item_mean_centered_cosine(t_item_ratings, cand_ratings, user_means)
                    if sim > 0: 
                        similarities.append((cand_item, sim))
                except Exception:
                    continue
                
            # 2. Select Top 20%
            similarities.sort(key=lambda x: x[1], reverse=True)
            k = max(1, int(len(similarities) * 0.20)) # Ensure at least 1 if neighbors exist
            top_neighbors = similarities[:k]
        
            # 3. Predict using Utils
            # predict_item_based(target_user_id, item_id, top_neighbors, user_item_ratings, user_means, item_means=None)
            base_pred = predict_item_based(u, i, top_neighbors, user_item_ratings, user_means, item_means)


            # --- Clustered (Local) ---
            # Get cluster candidates
            i_cluster = feature_df.loc[feature_df['item'] == i, 'cluster'].values[0]
            cluster_candidate_items = cluster_items[i_cluster]
            t_item_ratings = item_user_ratings.get(i, {})
            similarities = []
            
            for cand_item in cluster_candidate_items:
                if cand_item == i:
                    continue
                cand_ratings = item_user_ratings.get(cand_item, {})
                
                try:
                    sim = calculate_item_mean_centered_cosine(t_item_ratings, cand_ratings, user_means)
                    if sim > 0: 
                        similarities.append((cand_item, sim))
                except Exception:
                    continue
                
            # 2. Select Top 20%
            similarities.sort(key=lambda x: x[1], reverse=True)
            k = max(1, int(len(similarities) * 0.20)) # Ensure at least 1 if neighbors exist
            top_neighbors = similarities[:k]
        
            # 3. Predict using Utils
            # predict_item_based(target_user_id, item_id, top_neighbors, user_item_ratings, user_means, item_means=None)
            clus_pred = predict_item_based(u, i, top_neighbors, user_item_ratings, user_means, item_means)
            
            
            
            print(f"{u:<10} | {i:<10} | {base_pred:.2f} |   {clus_pred:.2f}     | {abs(base_pred - clus_pred):.2f}")




if __name__ == "__main__":
    main()

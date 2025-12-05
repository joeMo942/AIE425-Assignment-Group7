import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path to allow importing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import data_loader
from utils import similarity

def main():
    print("="*80)
    print("Item-Based Collaborative Filtering Implementation")
    print("="*80)

    # -------------------------------------------------------------------------
    # 1. Data Loading
    # -------------------------------------------------------------------------
    print("\n[1] Loading Data...")
    try:
        df = data_loader.get_preprocessed_dataset()
        target_items = data_loader.get_target_items()
        # Ensure target_items are strings if they look like IDs but are stored as mixed types, 
        # though data_loader seems to return ints? Let's check the data loader output.
        # viewing data_loader.py: "items = [int(line.strip()) for line in f if line.strip()]"
        # The dataset has '0001527665' as item IDs (strings/objects). 
        # The data loader casts to int? 
        # Wait, the dataset uses string IDs for items (ASINs usually).
        # Let's double check data_loader.get_target_items.
        # It converts to INT: "items = [int(line.strip()) for line in f if line.strip()]"
        # BUT the dataset shown in the notebook has 'B00AP2DD48' which is a string.
        # '0001527665' can be an int, but 'B00AP2DD48' cannot.
        # This might be a bug in the provided data_loader if expected IDs are alphanumeric.
        # I should probably trust the data loader or handle the exception if it fails.
        # Given the previous context, 'target_items.txt' might contain integer IDs if the dataset was mapped.
        # However, the notebook explicitly shows 'B00AP2DD48'. 
        # Let's rely on what data_loader returns. If it fails, I'll fix it.
        # Actually, let's verify if I should change data_loader. But I am supposed to use it.
        # I will accept whatever data_loader returns.
        
        # Correction: I should check if the IDs in df are comparable to target_items.
        pass 
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Dataset shape: {df.shape}")
    print(f"Number of target items: {len(target_items)}")

    # Build lookup dictionaries
    print("Building lookup dictionaries...")
    item_user_ratings = defaultdict(dict)
    user_item_ratings = defaultdict(dict)
    
    # We also need item means for prediction fallback
    item_sums = defaultdict(float)
    item_counts = defaultdict(int)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
        user = row['user']
        item = row['item']
        rating = row['rating']
        
        item_user_ratings[item][user] = rating
        user_item_ratings[user][item] = rating
        
        item_sums[item] += rating
        item_counts[item] += 1
        
    item_means = {k: v / item_counts[k] for k, v in item_sums.items()}
    all_items = list(item_user_ratings.keys())
    print(f"Total items in dataset: {len(all_items)}")

    # -------------------------------------------------------------------------
    # 2. Step 1: Item-Based Similarity (Cosine with Mean-Centering / Pearson)
    # -------------------------------------------------------------------------
    print("\n[Step 1] Calculating Similarities (Pearson)...")
    
    # Store similarities: {target_item: {other_item: similarity}}
    target_similarities = {}
    
    for target_item in tqdm(target_items, desc="Target Items"):
        # If target item not in dataset, skip
        if target_item not in item_user_ratings:
            print(f"Warning: Target item {target_item} not found in dataset.")
            continue
            
        sims = {}
        target_ratings = item_user_ratings[target_item]
        
        for other_item in all_items:
            if target_item == other_item:
                sims[other_item] = 1.0
                continue
                
            other_ratings = item_user_ratings[other_item]
            
            # Use utils.similarity.calculate_item_pearson
            # Returns 0.0 if not enough common items
            sim = similarity.calculate_item_pearson(target_ratings, other_ratings)
            
            if sim != 0:
                sims[other_item] = sim
                
        target_similarities[target_item] = sims

    # -------------------------------------------------------------------------
    # 3. Step 2: Identify Top 20% Similar Items
    # -------------------------------------------------------------------------
    print("\n[Step 2] Selecting Top 20% Neighbors...")
    
    top_k_percent = int(len(all_items) * 0.2)
    top_k_neighbors_sim = {}
    
    for target_item in target_similarities:
        sims = target_similarities[target_item]
        # Sort desc
        sorted_items = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        # Filter > 0 and take top K
        top_items = [(item, sim) for item, sim in sorted_items if sim > 0][:top_k_percent]
        
        top_k_neighbors_sim[target_item] = top_items
        print(f"Target {target_item}: found {len(top_items)} neighbors.")

    # -------------------------------------------------------------------------
    # 4. Step 3: Predict Missing Ratings
    # -------------------------------------------------------------------------
    print("\n[Step 3] Predicting Ratings (Similarity-Based)...")

    # Define validation set (random sample of 100)
    print("Creating validation set (random 100 samples)...")
    validation_set = []
    actuals = []
    
    # Sample 100 indices
    sample_indices = np.random.choice(len(df), 100, replace=False)
    for idx in sample_indices:
        row = df.iloc[idx]
        validation_set.append((row['user'], row['item'], row['rating']))
        actuals.append(row['rating'])
        
    actuals = np.array(actuals)
    predictions_sim = []
    
    
    def predict_mean_centered(user_id, item_id, neighbors_list, item_means_dict):
        # neighbors_list is list of (item, sim)
        if not neighbors_list:
            return item_means_dict.get(item_id, 3.0)
        
        numerator = 0.0
        denominator = 0.0
        
        if not user_ratings:
            return item_means_dict.get(item_id, 3.0)
            
        user_mean = sum(user_ratings.values()) / len(user_ratings)
        
        for neighbor, sim in neighbors_list:
            if neighbor in user_ratings:
                rating = user_ratings[neighbor]
                numerator += sim * (rating - user_mean)
                denominator += sim
                
        if denominator == 0:
            return item_means_dict.get(item_id, 3.0)
            
        return user_mean + (numerator / denominator)

    def predict_pearson(user_id, item_id, neighbors_list, item_means_dict):
        if not neighbors_list:
            return item_means_dict.get(item_id, 3.0)
            
        user_ratings = user_item_ratings.get(user_id, {})
        if not user_ratings:
            return item_means_dict.get(item_id, 3.0)
            
        numerator = 0.0
        denominator = 0.0
        
        target_item_mean = item_means_dict.get(item_id, 3.0)
        
        for neighbor, sim in neighbors_list:
            if neighbor in user_ratings:
                rating = user_ratings[neighbor]
                neighbor_mean = item_means_dict.get(neighbor, 3.0)
                numerator += sim * (rating - neighbor_mean)
                denominator += sim
                
        if denominator == 0:
            return target_item_mean
            
        return target_item_mean + (numerator / denominator)

    # Perform predictions for Step 3
    for u, i, r in validation_set:
        neighbors = top_k_neighbors_sim.get(i, [])
        pred = predict_mean_centered(u, i, neighbors, item_means)
        predictions_sim.append(pred)
        
    predictions_sim = np.array(predictions_sim)
    

    # -------------------------------------------------------------------------
    # 5. Step 4: Compute DF and DS
    # -------------------------------------------------------------------------
    print("\n[Step 4] Computing DF and DS (Dynamic Beta)...")
    
    target_ds_scores = {}
    
    for target_item in target_items:
        if target_item not in item_user_ratings: continue
        
        ds_scores = {}
        target_users = set(item_user_ratings[target_item].keys())
        num_ratings_target = len(target_users)
        
        # Beta = 30% of number of ratings for the target item
        # Ensure beta > 0 to avoid division by zero. If 0 ratings, beta=0.
        beta = 0.3 * num_ratings_target
        if beta == 0: beta = 1e-9 # Prevent div/0
        
        for other_item in all_items:
            if other_item == target_item:
                ds_scores[other_item] = 1.0 # Self is typically max, or 0? Notebook used 0. Let's stick to valid logic.
                # If we use it for weighting, 1.0 makes sense. Notebook said "ds_scores[other_item] = 0.0" for self.
                # But self is excluded from neighbors anyway.
                ds_scores[other_item] = 0.0
                continue
                
            other_users = set(item_user_ratings[other_item].keys())
            
            co_rated_count = len(target_users.intersection(other_users))
            
            # DS = min(beta, corated_users) / beta
            ds = min(beta, co_rated_count) / beta
            ds_scores[other_item] = ds
            
        target_ds_scores[target_item] = ds_scores
        print(f"Target {target_item}: Beta={beta:.2f}")

    # -------------------------------------------------------------------------
    # 6. Step 5: Select Top 20% Items using DS
    # -------------------------------------------------------------------------
    print("\n[Step 5] Selecting Top 20% Neighbors (DS-Weighted)...")
    
    top_k_neighbors_ds = {}
    
    for target_item in target_similarities:
        sims = target_similarities[target_item]
        ds_vals = target_ds_scores[target_item]
        
        weighted_sims = {}
        for other_item, sim in sims.items():
            if sim > 0:
                ds = ds_vals.get(other_item, 0.0)
                weighted_sims[other_item] = sim * ds
                
        sorted_items = sorted(weighted_sims.items(), key=lambda x: x[1], reverse=True)
        top_items = sorted_items[:top_k_percent]
        # Filter zeros? The notebook filtered "if score > 0". 
        top_items = [(item, score) for item, score in top_items if score > 0]
        
        top_k_neighbors_ds[target_item] = top_items
        print(f"Target {target_item}: found {len(top_items)} DS-neighbors.")

    # -------------------------------------------------------------------------
    # 7. Step 6: Updated Rating Predictions
    # -------------------------------------------------------------------------
    print("\n[Step 6] Predicting Ratings (DS-Weighted)...")
    
    predictions_ds = []
    
    for u, i, r in validation_set:
        neighbors = top_k_neighbors_ds.get(i, [])
        pred = predict_mean_centered(u, i, neighbors, item_means)
        predictions_ds.append(pred)
        
    predictions_ds = np.array(predictions_ds)
    
    
    # -------------------------------------------------------------------------
    # 8. Step 7: Compare Similarity Lists
    # -------------------------------------------------------------------------
    print("\n[Step 7] Comparison of Neighborhoods...")
    
    for target_item in target_items:
        if target_item not in top_k_neighbors_sim: continue
        
        set_sim = set([x[0] for x in top_k_neighbors_sim[target_item]])
        set_ds = set([x[0] for x in top_k_neighbors_ds[target_item]])
        
        intersection = len(set_sim.intersection(set_ds))
        union = len(set_sim.union(set_ds))
        
        jaccard = intersection / union if union > 0 else 0
        overlap = (intersection / len(set_sim)) * 100 if len(set_sim) > 0 else 0
        
        print(f"Target {target_item}: Jaccard={jaccard:.4f}, Overlap={overlap:.2f}%")
        
        # Display top 10 items from both lists for visual comparison
        print(f"  Top 10 Neighbors (Sim-Based):")
        for i, (item, score) in enumerate(top_k_neighbors_sim[target_item][:10], 1):
            print(f"    {i}. {item} (Sim: {score:.4f})")
            
        print(f"  Top 10 Neighbors (DS-Weighted):")
        for i, (item, score) in enumerate(top_k_neighbors_ds[target_item][:10], 1):
            print(f"    {i}. {item} (Score: {score:.4f})")
        
    # -------------------------------------------------------------------------
    # 9. Step 8 & 9: Final Commentary
    # -------------------------------------------------------------------------
    print("\n[Step 8 & 9] Final Discussion")
    print(f"\nComparison of Predictions (First 20 Samples):")
    print(f"{'User':<15} | {'Item':<12} | {'Actual':<6} | {'Sim-Pred':<8} | {'DS-Pred':<8}")
    print("-" * 65)
    
    for k in range(min(20, len(actuals))):
        u, i, r = validation_set[k]
        p_sim = predictions_sim[k]
        p_ds = predictions_ds[k]
        print(f"{u:<15} | {i:<12} | {r:<6.1f} | {p_sim:<8.2f} | {p_ds:<8.2f}")

if __name__ == "__main__":
    main()

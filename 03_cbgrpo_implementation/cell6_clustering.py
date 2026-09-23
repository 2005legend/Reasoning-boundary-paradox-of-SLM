from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import Counter

def cluster_training_prompts(
    prompts: List[str],
    n_clusters: int = 16,
    embedding_model: str = "all-MiniLM-L6-v2",
    cluster_file: str = "cluster_assignments.pkl",
    force_recompute: bool = False
) -> Dict[str, int]:
    """
    Clusters training prompts by semantic similarity for CB-GRPO.
    
    This function:
    1. Embeds all training prompts using sentence-transformers
    2. Clusters embeddings using KMeans
    3. Assigns cluster ID to each prompt
    4. Saves cluster assignments to disk for reuse
    5. Loads existing assignments if available (unless force_recompute=True)
    
    Requirements:
    - 5.1: Embed prompts using sentence-transformers (all-MiniLM-L6-v2)
    - 5.2: Cluster embeddings using KMeans with n_clusters=16
    - 5.3: Assign cluster ID to each training prompt
    - 5.4: Save cluster assignments to disk
    - 5.5: Load existing assignments if available
    - 5.6: Display cluster size distribution
    - 5.7: Maintain cluster ID mapping throughout training
    
    Args:
        prompts: List of training prompt strings
        n_clusters: Number of clusters for KMeans (default: 16)
        embedding_model: Sentence-transformers model name (default: all-MiniLM-L6-v2)
        cluster_file: Path to save/load cluster assignments
        force_recompute: If True, recompute clusters even if file exists
    
    Returns:
        Dictionary mapping prompt text to cluster ID (0 to n_clusters-1)
    """
    print("=" * 60)
    print("PROMPT CLUSTERING FOR CB-GRPO")
    print("=" * 60)
    print(f"\nTotal prompts to cluster: {len(prompts)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"Embedding model: {embedding_model}")
    print(f"Cluster file: {cluster_file}")
    
    cluster_path = Path(cluster_file)
    
    # Requirement 5.5: Load existing assignments if available
    if cluster_path.exists() and not force_recompute:
        print("\n" + "=" * 60)
        print("LOADING EXISTING CLUSTER ASSIGNMENTS")
        print("=" * 60)
        print(f"\nFound existing cluster assignments at: {cluster_path}")
        print("Loading from disk to avoid recomputation...")
        
        try:
            with open(cluster_path, 'rb') as f:
                cluster_data = pickle.load(f)
            
            cluster_assignments = cluster_data['assignments']
            loaded_n_clusters = cluster_data['n_clusters']
            loaded_model = cluster_data['embedding_model']
            
            print(f"\n✓ Loaded {len(cluster_assignments)} cluster assignments")
            print(f"  → Clusters: {loaded_n_clusters}")
            print(f"  → Embedding model: {loaded_model}")
            
            # Verify cluster assignments match current prompts
            if len(cluster_assignments) != len(prompts):
                print(f"\n⚠️  Warning: Loaded {len(cluster_assignments)} assignments but have {len(prompts)} prompts.")
                print("  This may indicate dataset version mismatch. Recomputing clusters...")
            else:
                # Display cluster size distribution
                _display_cluster_distribution(cluster_assignments, n_clusters)
                
                print("\n" + "=" * 60)
                print("✅ CLUSTER LOADING COMPLETE")
                print("=" * 60)
                return cluster_assignments
        
        except Exception as e:
            print(f"\n❌ Failed to load cluster assignments: {str(e)}")
            print("Recomputing clusters from scratch...")
    
    # Compute new cluster assignments
    print("\n" + "=" * 60)
    print("COMPUTING CLUSTER ASSIGNMENTS")
    print("=" * 60)
    
    # Requirement 5.1: Embed all training prompts using sentence-transformers
    print(f"\nStep 1/3: Loading embedding model '{embedding_model}'...")
    try:
        model = SentenceTransformer(embedding_model)
        print("✓ Embedding model loaded successfully")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load embedding model: {str(e)}")
        print("\nRecommended actions:")
        print("  1. Ensure sentence-transformers is installed: pip install sentence-transformers")
        print("  2. Check internet connection (model may need to be downloaded)")
        print("  3. Try a different embedding model name")
        raise RuntimeError(f"Failed to load embedding model: {str(e)}")
    
    print(f"\nStep 2/3: Embedding {len(prompts)} prompts...")
    print("  (This may take 1-2 minutes depending on dataset size)")
    
    try:
        # Encode all prompts to embeddings
        # Use show_progress_bar=True to display progress during encoding
        embeddings = model.encode(
            prompts,
            show_progress_bar=True,
            batch_size=32,  # Process in batches for efficiency
            convert_to_numpy=True
        )
        
        print(f"\n✓ Embeddings computed successfully")
        print(f"  → Embedding shape: {embeddings.shape}")
        print(f"  → Embedding dimension: {embeddings.shape[1]}")
    
    except Exception as e:
        print(f"\n❌ ERROR: Failed to compute embeddings: {str(e)}")
        raise RuntimeError(f"Failed to compute embeddings: {str(e)}")
    
    # Requirement 5.2: Cluster embeddings using KMeans with n_clusters=16
    print(f"\nStep 3/3: Clustering embeddings with KMeans (k={n_clusters})...")
    
    try:
        # Initialize KMeans clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,  # For reproducibility
            n_init=10,  # Number of initializations (default in scikit-learn >= 1.2)
            max_iter=300,
            verbose=0
        )
        
        # Fit KMeans and get cluster labels
        cluster_labels = kmeans.fit_predict(embeddings)
        
        print("✓ KMeans clustering complete")
        print(f"  → Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
        print(f"  → Number of iterations: {kmeans.n_iter_}")
    
    except Exception as e:
        print(f"\n❌ ERROR: Failed to cluster embeddings: {str(e)}")
        raise RuntimeError(f"Failed to cluster embeddings: {str(e)}")
    
    # Requirement 5.3: Assign cluster ID to each training prompt
    print("\n" + "=" * 60)
    print("CREATING CLUSTER ASSIGNMENTS")
    print("=" * 60)
    
    # Create dictionary mapping prompt text to cluster ID
    cluster_assignments = {
        prompt: int(cluster_id)
        for prompt, cluster_id in zip(prompts, cluster_labels)
    }
    
    print(f"\n✓ Created {len(cluster_assignments)} cluster assignments")
    
    # Requirement 5.4: Save cluster assignments to disk
    print("\n" + "=" * 60)
    print("SAVING CLUSTER ASSIGNMENTS")
    print("=" * 60)
    
    cluster_data = {
        'assignments': cluster_assignments,
        'n_clusters': n_clusters,
        'embedding_model': embedding_model,
        'cluster_centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'n_iter': kmeans.n_iter_
    }
    
    try:
        # Create parent directory if it doesn't exist
        cluster_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cluster_path, 'wb') as f:
            pickle.dump(cluster_data, f)
        
        print(f"\n✓ Cluster assignments saved to: {cluster_path}")
        
        # Display file size
        file_size_kb = cluster_path.stat().st_size / 1024
        print(f"  → File size: {file_size_kb:.2f} KB")
    
    except Exception as e:
        print(f"\n⚠️  Warning: Failed to save cluster assignments: {str(e)}")
        print("  Clustering will work but assignments won't be cached for future runs.")
    
    # Requirement 5.6: Display cluster size distribution
    _display_cluster_distribution(cluster_assignments, n_clusters)
    
    print("\n" + "=" * 60)
    print("✅ PROMPT CLUSTERING COMPLETE")
    print("=" * 60)
    print(f"\n✓ {len(cluster_assignments)} prompts assigned to {n_clusters} clusters")
    print("  → Cluster assignments stored in 'cluster_assignments' variable")
    print("  → Use get_cluster_members(cluster_id) to inspect specific clusters")
    
    return cluster_assignments

def _display_cluster_distribution(cluster_assignments: Dict[str, int], n_clusters: int):
    """
    Displays cluster size distribution (Requirement 5.6).
    
    Args:
        cluster_assignments: Dictionary mapping prompts to cluster IDs
        n_clusters: Total number of clusters
    """
    print("\n" + "=" * 60)
    print("CLUSTER SIZE DISTRIBUTION")
    print("=" * 60)
    
    # Count prompts per cluster
    cluster_counts = Counter(cluster_assignments.values())
    
    # Ensure all clusters are represented (even if empty)
    for cluster_id in range(n_clusters):
        if cluster_id not in cluster_counts:
            cluster_counts[cluster_id] = 0
    
    # Sort by cluster ID
    sorted_counts = sorted(cluster_counts.items())
    
    # Calculate statistics
    total_prompts = len(cluster_assignments)
    cluster_sizes = [count for _, count in sorted_counts]
    mean_size = np.mean(cluster_sizes)
    std_size = np.std(cluster_sizes)
    min_size = np.min(cluster_sizes)
    max_size = np.max(cluster_sizes)
    
    print(f"\nCluster Statistics:")
    print(f"  Total prompts: {total_prompts}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Mean cluster size: {mean_size:.1f}")
    print(f"  Std deviation: {std_size:.1f}")
    print(f"  Min cluster size: {min_size}")
    print(f"  Max cluster size: {max_size}")
    
    # Display per-cluster breakdown
    print(f"\nPer-Cluster Breakdown:")
    print("  Cluster ID | Size | Percentage | Bar")
    print("  " + "-" * 56)
    
    for cluster_id, count in sorted_counts:
        percentage = (count / total_prompts) * 100 if total_prompts > 0 else 0
        bar_length = int(percentage / 2)  # Scale bar to fit display (max 50 chars)
        bar = "█" * bar_length
        print(f"  {cluster_id:10d} | {count:4d} | {percentage:6.2f}%   | {bar}")
    
    # Optional: Create histogram visualization
    try:
        print("\nGenerating cluster size histogram...")
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(n_clusters), cluster_sizes, color='steelblue', edgecolor='black')
        plt.axhline(y=mean_size, color='red', linestyle='--', label=f'Mean: {mean_size:.1f}')
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Prompts', fontsize=12)
        plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
        plt.xticks(range(n_clusters))
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("✓ Histogram displayed above")
    except Exception as e:
        print(f"  ⚠️  Could not generate histogram: {str(e)}")

def get_cluster_members(
    cluster_id: int,
    cluster_assignments: Dict[str, int],
    max_display: int = 5
) -> List[str]:
    """
    Retrieves all prompts belonging to a specific cluster.
    
    Args:
        cluster_id: Cluster ID to retrieve members for
        cluster_assignments: Dictionary mapping prompts to cluster IDs
        max_display: Maximum number of prompts to display (default: 5)
    
    Returns:
        List of prompts in the specified cluster
    """
    members = [
        prompt for prompt, cid in cluster_assignments.items()
        if cid == cluster_id
    ]
    
    print("=" * 60)
    print(f"CLUSTER {cluster_id} MEMBERS")
    print("=" * 60)
    print(f"\nTotal members: {len(members)}")
    
    if len(members) > 0:
        print(f"\nShowing first {min(max_display, len(members))} members:")
        print("\n" + "-" * 60)
        
        for i, prompt in enumerate(members[:max_display], 1):
            # Truncate long prompts for display
            display_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
            print(f"\n{i}. {display_prompt}")
            print("-" * 60)
        
        if len(members) > max_display:
            print(f"\n... and {len(members) - max_display} more members")
    else:
        print("\n⚠️  This cluster has no members.")
    
    return members

# Execute prompt clustering
if __name__ == "__main__":
    # Ensure GSM8K train data is loaded
    if 'gsm8k_train' not in locals():
        print("❌ ERROR: GSM8K train data not found.")
        print("Please run Cell 4 (GSM8K Dataset Loader) first.")
    else:
        # Cluster training prompts
        cluster_assignments = cluster_training_prompts(
            prompts=gsm8k_train['problems'],
            n_clusters=16,
            embedding_model="all-MiniLM-L6-v2",
            cluster_file="cluster_assignments.pkl",
            force_recompute=False  # Set to True to recompute even if file exists
        )
        
        # Demo: Display members of cluster 0
        print("\n\n" + "=" * 60)
        print("DEMO: Inspecting Cluster 0 Members")
        print("=" * 60)
        cluster_0_members = get_cluster_members(0, cluster_assignments, max_display=3)

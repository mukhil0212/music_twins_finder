# Music Taste Twins - API Documentation

## Overview

This documentation covers the Python API for the Music Taste Twins project, including all modules, classes, and functions.

## Table of Contents

1. [Data Collection](#data-collection)
2. [Feature Engineering](#feature-engineering)
3. [Clustering](#clustering)
4. [Similarity Matching](#similarity-matching)
5. [Visualization](#visualization)
6. [Utilities](#utilities)

---

## Data Collection

### `src.data_collection.spotify_auth`

#### Class: `SpotifyAuth`

Handles Spotify OAuth authentication and API rate limiting.

```python
auth = SpotifyAuth()
```

**Methods:**

- `spotify` (property): Returns authenticated Spotify client
- `get_user_info()`: Get current user information
- `validate_user_access(username=None)`: Validate user access
- `@rate_limit_handler`: Decorator for handling rate limits

### `src.data_collection.spotify_collector`

#### Class: `SpotifyCollector`

Collects user data from Spotify API.

```python
collector = SpotifyCollector()
```

**Methods:**

- `collect_user_data(username=None, save_to_file=True) -> Dict`
  - Collects comprehensive user data including tracks, artists, and audio features
  - Returns dictionary with user profile data

- `get_top_tracks(time_range='medium_term', limit=50) -> List[Dict]`
  - Returns user's top tracks for specified time range

- `get_top_artists(time_range='medium_term', limit=50) -> List[Dict]`
  - Returns user's top artists for specified time range

- `get_audio_features(track_ids: List[str]) -> List[Dict]`
  - Returns audio features for given track IDs

- `collect_multiple_users(user_ids: List[str]) -> List[Dict]`
  - Batch collection for multiple users

---

## Feature Engineering

### `src.feature_engineering.audio_features`

#### Class: `AudioFeatureExtractor`

Extracts and processes audio features from user data.

```python
extractor = AudioFeatureExtractor(scaling_method='standard')
```

**Methods:**

- `extract_features(user_data: Dict) -> np.ndarray`
  - Extracts comprehensive feature vector from user data
  - Returns numpy array of features

- `fit_transform(users_data: List[Dict]) -> Tuple[np.ndarray, List[str]]`
  - Extract and scale features for multiple users
  - Returns scaled features and user IDs

- `get_feature_importance(X: np.ndarray) -> pd.DataFrame`
  - Calculate feature importance based on variance
  - Returns DataFrame with importance scores

### `src.feature_engineering.user_profiles`

#### Class: `UserProfileBuilder`

Builds comprehensive user profiles from raw data.

```python
builder = UserProfileBuilder()
```

**Methods:**

- `build_profiles(users_data: List[Dict], use_feature_selection=False, n_features=None) -> Tuple[pd.DataFrame, List[str]]`
  - Build user profiles with optional feature selection
  - Returns DataFrame of profiles and feature names

- `create_profile_summary(user_profile: pd.Series) -> Dict`
  - Create human-readable summary of user profile
  - Returns dictionary with music characteristics and preferences

- `calculate_profile_statistics(profiles_df: pd.DataFrame) -> Dict`
  - Calculate statistics across all profiles
  - Returns summary statistics

---

## Clustering

### `src.clustering.kmeans_clustering`

#### Class: `KMeansClustering`

Implements K-means clustering with optimization.

```python
kmeans = KMeansClustering(n_clusters=None, random_state=42)
```

**Methods:**

- `find_optimal_k(X: np.ndarray, k_range=None) -> Dict`
  - Find optimal number of clusters using elbow method
  - Returns dictionary with metrics and recommendations

- `fit(X: np.ndarray, n_clusters=None) -> KMeansClustering`
  - Fit clustering model
  - Returns self

- `predict(X: np.ndarray) -> np.ndarray`
  - Predict cluster labels for new data

- `get_cluster_statistics(X: np.ndarray, feature_names: List[str]) -> Dict`
  - Calculate detailed statistics for each cluster

### `src.clustering.hierarchical_clustering`

#### Class: `HierarchicalClustering`

Implements hierarchical clustering methods.

```python
hierarchical = HierarchicalClustering(n_clusters=None, linkage_method='ward')
```

**Methods:**

- `fit_predict(X: np.ndarray, n_clusters=None) -> np.ndarray`
  - Fit and predict cluster labels

- `plot_dendrogram(user_ids=None, save_path=None, max_display=50)`
  - Plot hierarchical clustering dendrogram

- `compare_linkage_methods(X: np.ndarray) -> Dict`
  - Compare different linkage methods

### `src.clustering.evaluation_metrics`

#### Class: `ClusteringEvaluator`

Evaluates clustering quality with multiple metrics.

```python
evaluator = ClusteringEvaluator()
```

**Methods:**

- `evaluate_clustering(X: np.ndarray, labels: np.ndarray) -> Dict`
  - Calculate comprehensive clustering metrics
  - Returns silhouette score, Davies-Bouldin index, etc.

- `stability_analysis(X: np.ndarray, clustering_func, n_iterations=10) -> Dict`
  - Analyze clustering stability through subsampling

---

## Similarity Matching

### `src.similarity.similarity_matcher`

#### Class: `SimilarityMatcher`

Finds similar users based on feature vectors.

```python
matcher = SimilarityMatcher(metric='cosine')
```

**Methods:**

- `fit(user_features: np.ndarray, user_ids: List[str], cluster_labels=None)`
  - Fit similarity matcher with user data

- `find_similar_users(user_id: str, top_n=10, same_cluster_only=False) -> List[Dict]`
  - Find top N similar users
  - Returns list with similarity scores

- `find_taste_twins(user_id: str, similarity_threshold=0.8) -> List[Dict]`
  - Find users with very high similarity

- `calculate_similarity_matrix(user_ids=None) -> np.ndarray`
  - Calculate full similarity matrix

---

## Visualization

### `src.visualization.dimensionality_reduction`

#### Class: `DimensionalityReducer`

Implements PCA, t-SNE, and UMAP visualizations.

```python
reducer = DimensionalityReducer()
```

**Methods:**

- `apply_pca(X: np.ndarray, n_components=3) -> Dict`
  - Apply PCA and return results

- `apply_tsne(X: np.ndarray, perplexity_values=[5, 30, 50]) -> Dict`
  - Apply t-SNE with multiple perplexity values

- `apply_umap(X: np.ndarray, n_neighbors_values=[5, 15, 30]) -> Dict`
  - Apply UMAP with different parameters

- `plot_all_methods_comparison(cluster_labels=None, save_path=None)`
  - Create comparison plot of all methods

### `src.visualization.heatmap_generator`

#### Class: `HeatmapGenerator`

Creates various heatmap visualizations.

```python
heatmap_gen = HeatmapGenerator()
```

**Methods:**

- `create_feature_correlation_heatmap(features_df, feature_names=None, save_path=None, clustered=True)`
  - Create correlation heatmap with optional clustering

- `create_user_similarity_heatmap(similarity_matrix, user_ids, cluster_labels=None, save_path=None)`
  - Create user-to-user similarity heatmap

- `create_genre_distribution_heatmap(genre_data, cluster_labels=None, save_path=None)`
  - Create genre preference heatmap

### `src.visualization.eda_visualizer`

#### Class: `EDAVisualizer`

Comprehensive EDA visualizations.

```python
eda_viz = EDAVisualizer()
```

**Methods:**

- `create_feature_distributions(data, feature_columns, save_path=None)`
  - Create distribution plots for all features

- `create_statistical_summary(data, feature_columns, cluster_column=None, save_path=None)`
  - Create comprehensive statistical summary

---

## Utilities

### `src.utils.helpers`

Utility functions for data processing and management.

**Functions:**

- `load_json(filepath: str) -> Dict`
  - Load JSON file

- `save_json(data: Dict, filepath: str)`
  - Save data to JSON

- `create_sample_data(n_users=50, random_state=42) -> List[Dict]`
  - Create sample user data for testing

- `validate_data_quality(data: pd.DataFrame, required_features: List[str]) -> Dict`
  - Validate data quality and return report

- `normalize_genre_name(genre: str) -> str`
  - Normalize genre names for consistency

---

## Web API Endpoints

### Flask Routes

#### `GET /`
Main page with user interface.

#### `POST /analyze`
Analyze user's Spotify data.

**Request Body:**
```json
{
  "username": "spotify_username"
}
```

**Response:**
```json
{
  "summary": {
    "total_users": 100,
    "n_clusters": 5,
    "cluster_sizes": {"0": 20, "1": 25, ...}
  },
  "user_analysis": {
    "user_id": "spotify_username",
    "cluster": 2,
    "similar_users": [...],
    "taste_twins": [...],
    "profile_summary": {...}
  },
  "visualizations": {
    "pca_analysis": "filename.png",
    ...
  }
}
```

#### `GET /demo`
Run demo with sample data.

#### `GET /api/clusters`
Get cluster information.

#### `GET /api/similar/<user_id>`
Find similar users for specific user.

---

## Example Usage

### Complete Pipeline

```python
# 1. Collect Data
from src.data_collection import SpotifyCollector

collector = SpotifyCollector()
users_data = collector.collect_multiple_users(['user1', 'user2', 'user3'])

# 2. Feature Engineering
from src.feature_engineering import UserProfileBuilder

builder = UserProfileBuilder()
profiles_df, feature_names = builder.build_profiles(users_data)

# 3. Clustering
from src.clustering import KMeansClustering

X = profiles_df[feature_names].values
kmeans = KMeansClustering()
cluster_labels = kmeans.fit_predict(X)

# 4. Similarity Matching
from src.similarity import SimilarityMatcher

matcher = SimilarityMatcher()
matcher.fit(X, profiles_df['user_id'].tolist(), cluster_labels)
similar_users = matcher.find_similar_users('user1', top_n=10)

# 5. Visualization
from src.visualization import DimensionalityReducer

reducer = DimensionalityReducer()
pca_results = reducer.apply_pca(X)
reducer.plot_pca_analysis(feature_names, cluster_labels)
```

### Using Sample Data

```python
from src.utils.helpers import create_sample_data

# Create sample data for testing
sample_data = create_sample_data(n_users=100)

# Process as normal
builder = UserProfileBuilder()
profiles_df, _ = builder.build_profiles(sample_data)
```

## Error Handling

All methods include proper error handling:

```python
try:
    user_data = collector.collect_user_data('invalid_user')
except Exception as e:
    print(f"Error: {e}")
```

## Performance Considerations

- Use batch processing for large datasets
- Enable caching for repeated operations
- Consider feature selection for high-dimensional data
- Use approximate algorithms for very large datasets

## Configuration

See `config/spotify_config.py` for all configuration options:
- API credentials
- Rate limiting
- Clustering parameters
- Feature selection
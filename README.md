# Music Taste Twins Finder

A machine learning project that finds users with similar music tastes using Spotify data, clustering algorithms, and similarity matching.

## Features

- **Data Collection**: Automated Spotify data collection including top tracks, artists, and audio features
- **Feature Engineering**: Comprehensive user profile creation with 50+ dimensional feature vectors
- **Clustering**: Multiple clustering algorithms (K-means, Hierarchical) with automatic hyperparameter tuning
- **Similarity Matching**: Fast nearest neighbor search to find your "music taste twins"
- **Visualizations**: 
  - Dimensionality reduction (PCA, t-SNE, UMAP)
  - Comprehensive heatmaps and correlation analysis
  - Interactive cluster exploration
  - Detailed EDA visualizations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/music-taste-twins.git
cd music-taste-twins
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Spotify API credentials:
   - Create a Spotify App at https://developer.spotify.com/dashboard/
   - Copy `.env.example` to `.env`
   - Fill in your Spotify credentials

5. Run the application:
```bash
python app/main.py
```

## Project Structure

```
music-taste-twins/
├── src/                    # Source code
│   ├── data_collection/    # Spotify data collection
│   ├── feature_engineering/# Feature extraction
│   ├── clustering/         # Clustering algorithms
│   ├── similarity/         # Similarity matching
│   └── visualization/      # Data visualization
├── notebooks/              # Jupyter notebooks
├── app/                    # Flask web application
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

## Usage

### Data Collection
```python
from src.data_collection import SpotifyCollector

collector = SpotifyCollector()
user_data = collector.collect_user_data(username="spotify_username")
```

### Clustering
```python
from src.clustering import KMeansClustering

clustering = KMeansClustering(n_clusters=5)
clusters = clustering.fit_predict(user_features)
```

### Finding Similar Users
```python
from src.similarity import SimilarityMatcher

matcher = SimilarityMatcher()
twins = matcher.find_similar_users(user_id, top_n=10)
```

## Visualizations

The project includes comprehensive visualizations:
- PCA, t-SNE, and UMAP for dimensionality reduction
- Feature correlation heatmaps
- Cluster distribution analysis
- User similarity networks
- Time-based listening patterns

## Testing

Run tests with:
```bash
pytest tests/
```

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
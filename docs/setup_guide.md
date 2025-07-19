# Music Taste Twins - Setup Guide

## Prerequisites

- Python 3.8 or higher
- Spotify Developer Account
- pip or conda package manager

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/music-taste-twins.git
cd music-taste-twins
```

### 2. Create Virtual Environment

Using venv:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Using conda:
```bash
conda create -n music-twins python=3.9
conda activate music-twins
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Spotify API Setup

1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. Create a new app
3. Note your Client ID and Client Secret
4. Add `http://localhost:8888/callback` to Redirect URIs

### 5. Configure Environment Variables

Copy the example environment file:
```bash
cp .env.example .env
```

Edit `.env` and add your Spotify credentials:
```
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
SPOTIFY_REDIRECT_URI=http://localhost:8888/callback
```

### 6. Create Required Directories

```bash
mkdir -p data/{raw,processed,cache}
mkdir -p static/visualizations
mkdir -p models
```

## Running the Application

### Web Application

Start the Flask web server:
```bash
python app/main.py
```

Open your browser and navigate to `http://localhost:8888`

### Jupyter Notebooks

Launch Jupyter:
```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and run notebooks in order:
1. `01_data_exploration.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_clustering_analysis.ipynb`
4. `04_similarity_matching.ipynb`
5. `05_comprehensive_eda.ipynb`

### Demo Mode

To try the application with sample data:
1. Start the web application
2. Click "Try Demo" button
3. No Spotify authentication required

## Data Collection

### Using the Web Interface

1. Enter your Spotify username
2. Click "Find My Twins"
3. Authenticate with Spotify when prompted
4. Wait for analysis to complete

### Using Python Scripts

```python
from src.data_collection import SpotifyCollector

collector = SpotifyCollector()
user_data = collector.collect_user_data("spotify_username")
```

### Batch Processing

To collect data for multiple users:
```python
user_ids = ["user1", "user2", "user3"]
all_data = collector.collect_multiple_users(user_ids)
```

## Troubleshooting

### Common Issues

1. **Authentication Error**
   - Verify Spotify credentials in `.env`
   - Check redirect URI matches exactly
   - Ensure app is not in development mode on Spotify

2. **Rate Limiting**
   - The app handles rate limiting automatically
   - For large datasets, collection may take time

3. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

4. **Permission Errors**
   ```bash
   chmod -R 755 data/
   chmod -R 755 static/
   ```

### Logging

Check logs for detailed error information:
- Application logs: Console output
- Spotify API logs: `data/cache/.spotify_cache`

## Performance Optimization

### Large Datasets

For datasets with >1000 users:
1. Use batch processing
2. Enable caching in configuration
3. Consider using subset of features
4. Use approximate algorithms (e.g., LSH for similarity)

### Memory Management

```python
# Process in chunks
chunk_size = 100
for i in range(0, len(users), chunk_size):
    chunk = users[i:i+chunk_size]
    process_chunk(chunk)
```

## Advanced Configuration

### Custom Feature Sets

Edit `config/spotify_config.py`:
```python
AUDIO_FEATURES = [
    'danceability', 'energy', 'valence',
    # Add/remove features as needed
]
```

### Clustering Parameters

Adjust in `config/spotify_config.py`:
```python
MIN_CLUSTERS = 2
MAX_CLUSTERS = 15
RANDOM_STATE = 42
```

### API Rate Limits

Configure in `.env`:
```
API_RATE_LIMIT=10  # Requests per second
API_TIMEOUT=30     # Timeout in seconds
```

## Testing

Run the test suite:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=src tests/
```

## Deployment

### Local Production

1. Set Flask to production mode:
   ```bash
   export FLASK_ENV=production
   ```

2. Use a production WSGI server:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app.main:app
   ```

### Docker (Optional)

Build and run with Docker:
```bash
docker build -t music-taste-twins .
docker run -p 8888:8888 --env-file .env music-taste-twins
```

## Support

For issues and questions:
- Check the [API Documentation](api_documentation.md)
- Open an issue on GitHub
- Review existing notebooks for examples
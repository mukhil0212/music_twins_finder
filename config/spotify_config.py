import os
from dotenv import load_dotenv

load_dotenv()

class SpotifyConfig:
    CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID')
    CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET')
    REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8888/callback')
    
    SCOPE = ' '.join([
        'user-top-read',
        'user-read-recently-played',
        'user-library-read',
        'playlist-read-private',
        'playlist-read-collaborative',
        'user-read-private',
        'user-read-email'
    ])
    
    CACHE_PATH = os.getenv('CACHE_PATH', './data/cache')
    
    # API Rate Limiting
    RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', 10))
    TIMEOUT = int(os.getenv('API_TIMEOUT', 30))
    
    # Batch Processing
    BATCH_SIZE = 50
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    
    # Data Collection Limits
    TOP_TRACKS_LIMIT = 50
    TOP_ARTISTS_LIMIT = 50
    RECENT_TRACKS_LIMIT = 50
    
    # Time Ranges for Top Items
    TIME_RANGES = ['short_term', 'medium_term', 'long_term']
    
    # Audio Features
    AUDIO_FEATURES = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo', 'time_signature'
    ]
    
    # Clustering Configuration
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 15
    RANDOM_STATE = 42
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.CLIENT_ID or not cls.CLIENT_SECRET:
            raise ValueError("Spotify API credentials not found. Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file")
        return True
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import time
import logging
from functools import wraps
from config.spotify_config import SpotifyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyAuth:
    def __init__(self):
        SpotifyConfig.validate()
        self.auth_manager = SpotifyOAuth(
            client_id=SpotifyConfig.CLIENT_ID,
            client_secret=SpotifyConfig.CLIENT_SECRET,
            redirect_uri=SpotifyConfig.REDIRECT_URI,
            scope=SpotifyConfig.SCOPE,
            cache_path=os.path.join(SpotifyConfig.CACHE_PATH, '.spotify_cache')
        )
        self._spotify = None
        self._last_request_time = 0
        
    @property
    def spotify(self):
        """Get authenticated Spotify client with automatic token refresh."""
        if self._spotify is None or self._token_expired():
            self._spotify = spotipy.Spotify(auth_manager=self.auth_manager)
            logger.info("Spotify client authenticated successfully")
        return self._spotify
    
    def _token_expired(self):
        """Check if the current token has expired."""
        token_info = self.auth_manager.get_cached_token()
        if token_info:
            return self.auth_manager.is_token_expired(token_info)
        return True
    
    def rate_limit_handler(func):
        """Decorator to handle rate limiting."""
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < (1.0 / SpotifyConfig.RATE_LIMIT):
                sleep_time = (1.0 / SpotifyConfig.RATE_LIMIT) - time_since_last
                time.sleep(sleep_time)
            
            retries = 0
            while retries < SpotifyConfig.MAX_RETRIES:
                try:
                    result = func(self, *args, **kwargs)
                    self._last_request_time = time.time()
                    return result
                except spotipy.SpotifyException as e:
                    if e.http_status == 429:  # Rate limit exceeded
                        retry_after = int(e.headers.get('Retry-After', SpotifyConfig.RETRY_DELAY))
                        logger.warning(f"Rate limit hit. Waiting {retry_after} seconds...")
                        time.sleep(retry_after)
                        retries += 1
                    elif e.http_status == 401:  # Unauthorized
                        logger.info("Token expired. Refreshing...")
                        self._spotify = None
                        retries += 1
                    else:
                        raise
                except Exception as e:
                    logger.error(f"Error in {func.__name__}: {str(e)}")
                    raise
            
            raise Exception(f"Max retries ({SpotifyConfig.MAX_RETRIES}) exceeded for {func.__name__}")
        return wrapper
    
    def get_user_info(self):
        """Get current user information."""
        return self.spotify.current_user()
    
    def validate_user_access(self, username=None):
        """Validate that we have access to user data."""
        try:
            user_info = self.get_user_info()
            if username and user_info['id'] != username:
                logger.warning(f"Authenticated as {user_info['id']}, not {username}")
            return user_info
        except Exception as e:
            logger.error(f"Failed to validate user access: {str(e)}")
            raise
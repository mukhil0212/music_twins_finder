import json
import os
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np

from .spotify_auth import SpotifyAuth
from config.spotify_config import SpotifyConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpotifyCollector(SpotifyAuth):
    def __init__(self):
        super().__init__()
        self.data_dir = os.path.join(SpotifyConfig.CACHE_PATH, '..', 'raw')
        os.makedirs(self.data_dir, exist_ok=True)
        
    @SpotifyAuth.rate_limit_handler
    def get_top_tracks(self, time_range='medium_term', limit=50):
        """Get user's top tracks for a specific time range."""
        tracks = []
        results = self.spotify.current_user_top_tracks(
            time_range=time_range,
            limit=min(limit, 50)
        )
        
        tracks.extend(results['items'])
        
        while results['next'] and len(tracks) < limit:
            results = self.spotify.next(results)
            tracks.extend(results['items'])
            
        return tracks[:limit]
    
    @SpotifyAuth.rate_limit_handler
    def get_top_artists(self, time_range='medium_term', limit=50):
        """Get user's top artists for a specific time range."""
        artists = []
        results = self.spotify.current_user_top_artists(
            time_range=time_range,
            limit=min(limit, 50)
        )
        
        artists.extend(results['items'])
        
        while results['next'] and len(artists) < limit:
            results = self.spotify.next(results)
            artists.extend(results['items'])
            
        return artists[:limit]
    
    @SpotifyAuth.rate_limit_handler
    def get_audio_features(self, track_ids: List[str]):
        """Get audio features for a list of tracks."""
        try:
            features = []

            for i in range(0, len(track_ids), SpotifyConfig.BATCH_SIZE):
                batch_ids = track_ids[i:i + SpotifyConfig.BATCH_SIZE]
                try:
                    batch_features = self.spotify.audio_features(batch_ids)
                    features.extend([f for f in batch_features if f is not None])
                except Exception as e:
                    if "403" in str(e) or "Forbidden" in str(e):
                        logger.warning("Audio features API is deprecated/forbidden. Skipping audio features collection.")
                        return []  # Return empty list instead of failing
                    else:
                        raise e  # Re-raise other errors

            return features
        except Exception as e:
            logger.error(f"Failed to get audio features: {str(e)}")
            logger.warning("Continuing without audio features...")
            return []  # Return empty list to continue processing
    
    @SpotifyAuth.rate_limit_handler
    def get_recently_played(self, limit=50):
        """Get user's recently played tracks."""
        results = self.spotify.current_user_recently_played(limit=limit)
        return results['items']
    
    @SpotifyAuth.rate_limit_handler
    def get_user_playlists(self, limit=50):
        """Get user's playlists."""
        playlists = []
        results = self.spotify.current_user_playlists(limit=min(limit, 50))
        
        playlists.extend(results['items'])
        
        while results['next'] and len(playlists) < limit:
            results = self.spotify.next(results)
            playlists.extend(results['items'])
            
        return playlists[:limit]
    
    @SpotifyAuth.rate_limit_handler
    def get_playlist_tracks(self, playlist_id: str):
        """Get tracks from a specific playlist."""
        tracks = []
        results = self.spotify.playlist_tracks(playlist_id)
        
        tracks.extend(results['items'])
        
        while results['next']:
            results = self.spotify.next(results)
            tracks.extend(results['items'])
            
        return tracks
    
    def extract_genres(self, artists: List[Dict]) -> Dict[str, float]:
        """Extract and normalize genre distribution from artists."""
        genre_counts = {}
        
        for artist in artists:
            for genre in artist.get('genres', []):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        total = sum(genre_counts.values())
        if total > 0:
            return {genre: count/total for genre, count in genre_counts.items()}
        return {}
    
    def extract_temporal_patterns(self, recently_played: List[Dict]) -> Dict:
        """Extract temporal listening patterns."""
        patterns = {
            'hour_distribution': [0] * 24,
            'day_distribution': [0] * 7,
            'listening_diversity': 0
        }
        
        if not recently_played:
            return patterns
        
        for item in recently_played:
            played_at = datetime.fromisoformat(item['played_at'].replace('Z', '+00:00'))
            patterns['hour_distribution'][played_at.hour] += 1
            patterns['day_distribution'][played_at.weekday()] += 1
        
        # Normalize distributions
        total_plays = len(recently_played)
        patterns['hour_distribution'] = [count/total_plays for count in patterns['hour_distribution']]
        patterns['day_distribution'] = [count/total_plays for count in patterns['day_distribution']]
        
        # Calculate listening diversity (Shannon entropy)
        unique_tracks = len(set(item['track']['id'] for item in recently_played))
        patterns['listening_diversity'] = unique_tracks / total_plays
        
        return patterns
    
    def collect_user_data(self, username: Optional[str] = None, save_to_file: bool = True) -> Dict:
        """Collect comprehensive user data from Spotify."""
        logger.info("Starting data collection...")
        
        # Validate user access
        user_info = self.validate_user_access(username)
        user_id = user_info['id']
        
        user_data = {
            'user_id': user_id,
            'display_name': user_info.get('display_name', user_id),
            'collected_at': datetime.now().isoformat(),
            'top_tracks': {},
            'top_artists': {},
            'audio_features': [],
            'genre_distribution': {},
            'listening_patterns': {},
            'playlists': []
        }
        
        # Collect top tracks for all time ranges
        logger.info("Collecting top tracks...")
        all_track_ids = set()
        
        for time_range in tqdm(SpotifyConfig.TIME_RANGES, desc="Time ranges"):
            tracks = self.get_top_tracks(time_range, SpotifyConfig.TOP_TRACKS_LIMIT)
            user_data['top_tracks'][time_range] = [
                {
                    'id': track['id'],
                    'name': track['name'],
                    'artists': [artist['name'] for artist in track['artists']],
                    'popularity': track['popularity'],
                    'duration_ms': track['duration_ms']
                }
                for track in tracks
            ]
            all_track_ids.update(track['id'] for track in tracks)
        
        # Collect top artists
        logger.info("Collecting top artists...")
        all_artists = []
        
        for time_range in tqdm(SpotifyConfig.TIME_RANGES, desc="Time ranges"):
            artists = self.get_top_artists(time_range, SpotifyConfig.TOP_ARTISTS_LIMIT)
            user_data['top_artists'][time_range] = [
                {
                    'id': artist['id'],
                    'name': artist['name'],
                    'genres': artist.get('genres', []),
                    'popularity': artist['popularity'],
                    'followers': artist['followers']['total']
                }
                for artist in artists
            ]
            all_artists.extend(artists)
        
        # Extract genre distribution
        user_data['genre_distribution'] = self.extract_genres(all_artists)
        
        # Collect audio features
        logger.info("Collecting audio features...")
        track_ids = list(all_track_ids)
        audio_features = self.get_audio_features(track_ids)
        user_data['audio_features'] = audio_features
        
        # Collect recently played tracks
        logger.info("Collecting recently played tracks...")
        recently_played = self.get_recently_played(SpotifyConfig.RECENT_TRACKS_LIMIT)
        user_data['recently_played'] = [
            {
                'track_id': item['track']['id'],
                'track_name': item['track']['name'],
                'played_at': item['played_at']
            }
            for item in recently_played
        ]
        
        # Extract temporal patterns
        user_data['listening_patterns'] = self.extract_temporal_patterns(recently_played)
        
        # Collect playlists
        logger.info("Collecting playlists...")
        playlists = self.get_user_playlists(20)
        user_data['playlists'] = [
            {
                'id': playlist['id'],
                'name': playlist['name'],
                'tracks_total': playlist['tracks']['total'],
                'public': playlist['public']
            }
            for playlist in playlists
        ]
        
        # Calculate summary statistics
        user_data['summary_stats'] = self._calculate_summary_stats(user_data)
        
        if save_to_file:
            filename = os.path.join(self.data_dir, f"{user_id}_data.json")
            with open(filename, 'w') as f:
                json.dump(user_data, f, indent=2)
            logger.info(f"Data saved to {filename}")
            
        return user_data
    
    def _calculate_summary_stats(self, user_data: Dict) -> Dict:
        """Calculate summary statistics from collected data."""
        audio_features = user_data.get('audio_features', [])

        stats = {}

        # Calculate audio feature stats if available
        if audio_features:
            for feature in SpotifyConfig.AUDIO_FEATURES:
                values = [f[feature] for f in audio_features if f and feature in f]
                if values:
                    stats[f"{feature}_mean"] = np.mean(values)
                    stats[f"{feature}_std"] = np.std(values)
                    stats[f"{feature}_median"] = np.median(values)
        else:
            logger.info("No audio features available - calculating alternative stats")

        # Calculate alternative stats based on available data
        top_tracks = user_data.get('top_tracks', {})
        top_artists = user_data.get('top_artists', {})

        # Track and artist diversity
        total_tracks = sum(len(tracks) for tracks in top_tracks.values())
        total_artists = sum(len(artists) for artists in top_artists.values())

        stats['total_tracks_collected'] = total_tracks
        stats['total_artists_collected'] = total_artists
        stats['artist_diversity'] = len(set(
            artist['id'] for artists in top_artists.values()
            for artist in artists
        ))

        # Genre diversity
        genre_dist = user_data.get('genre_distribution', {})
        stats['genre_count'] = len(genre_dist)
        stats['top_genre'] = max(genre_dist.items(), key=lambda x: x[1])[0] if genre_dist else None

        return stats
    
    def collect_multiple_users(self, user_ids: List[str], save_individual: bool = True) -> List[Dict]:
        """Collect data for multiple users."""
        all_users_data = []
        
        for user_id in tqdm(user_ids, desc="Collecting users"):
            try:
                user_data = self.collect_user_data(user_id, save_individual)
                all_users_data.append(user_data)
            except Exception as e:
                logger.error(f"Failed to collect data for {user_id}: {str(e)}")
                continue
        
        # Save combined data
        if all_users_data:
            filename = os.path.join(self.data_dir, "all_users_data.json")
            with open(filename, 'w') as f:
                json.dump(all_users_data, f, indent=2)
            logger.info(f"Combined data saved to {filename}")
        
        return all_users_data
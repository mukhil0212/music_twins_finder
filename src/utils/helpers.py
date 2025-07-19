import json
import os
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json(data: Dict, filepath: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {filepath}")

def load_pickle(filepath: str) -> Any:
    """Load pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_pickle(data: Any, filepath: str):
    """Save data to pickle file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
    logger.info(f"Saved pickle to {filepath}")

def ensure_directory(directory: str):
    """Ensure directory exists."""
    os.makedirs(directory, exist_ok=True)

def get_timestamp() -> str:
    """Get current timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def normalize_genre_name(genre: str) -> str:
    """Normalize genre names for consistency."""
    # Common replacements
    replacements = {
        'hip hop': 'hiphop',
        'r&b': 'rnb',
        'edm': 'electronic',
        'electro': 'electronic',
        'indie rock': 'indie',
        'indie pop': 'indie',
        'alt rock': 'alternative',
        'alternative rock': 'alternative'
    }
    
    genre_lower = genre.lower().strip()
    
    for old, new in replacements.items():
        if old in genre_lower:
            genre_lower = genre_lower.replace(old, new)
    
    return genre_lower

def aggregate_user_data(users_data: List[Dict]) -> pd.DataFrame:
    """Aggregate multiple user data dictionaries into a DataFrame."""
    aggregated_data = []
    
    for user_data in users_data:
        user_summary = {
            'user_id': user_data['user_id'],
            'display_name': user_data.get('display_name', user_data['user_id'])
        }
        
        # Add summary statistics if available
        if 'summary_stats' in user_data:
            user_summary.update(user_data['summary_stats'])
        
        # Add genre distribution
        if 'genre_distribution' in user_data:
            for genre, weight in user_data['genre_distribution'].items():
                normalized_genre = normalize_genre_name(genre)
                user_summary[f'genre_{normalized_genre}'] = weight
        
        # Add listening patterns
        if 'listening_patterns' in user_data:
            patterns = user_data['listening_patterns']
            if 'hour_distribution' in patterns:
                for hour, weight in enumerate(patterns['hour_distribution']):
                    user_summary[f'hour_{hour}'] = weight
            if 'listening_diversity' in patterns:
                user_summary['listening_diversity'] = patterns['listening_diversity']
        
        aggregated_data.append(user_summary)
    
    return pd.DataFrame(aggregated_data).fillna(0)

def create_sample_data(n_users: int = 50, random_state: int = 42) -> List[Dict]:
    """Create realistic sample user data with distinct music taste profiles."""
    np.random.seed(random_state)

    sample_users = []

    # First, create two "Music Taste Twins" for demo purposes
    twins_data = create_music_taste_twins()
    sample_users.extend(twins_data)

    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]

    # Define realistic music taste archetypes
    music_archetypes = {
        'edm_lover': {
            'name_patterns': ['DJ_', 'Rave_', 'Beat_', 'Drop_', 'Bass_'],
            'genres': ['electronic', 'house', 'techno', 'dubstep', 'trance'],
            'audio_profile': {
                'danceability': (0.8, 0.1), 'energy': (0.9, 0.1), 'valence': (0.7, 0.15),
                'acousticness': (0.1, 0.05), 'instrumentalness': (0.6, 0.2), 'tempo': (128, 20)
            },
            'artists': ['Calvin Harris', 'Deadmau5', 'Skrillex', 'Tiësto', 'David Guetta']
        },
        'indie_rock': {
            'name_patterns': ['Indie_', 'Alt_', 'Vinyl_', 'Garage_', 'Hipster_'],
            'genres': ['indie', 'alternative', 'indie rock', 'garage rock', 'post-punk'],
            'audio_profile': {
                'danceability': (0.5, 0.1), 'energy': (0.6, 0.15), 'valence': (0.5, 0.2),
                'acousticness': (0.3, 0.15), 'instrumentalness': (0.2, 0.1), 'tempo': (120, 25)
            },
            'artists': ['Arctic Monkeys', 'The Strokes', 'Vampire Weekend', 'Tame Impala', 'Mac DeMarco']
        },
        'classical': {
            'name_patterns': ['Maestro_', 'Symphony_', 'Classical_', 'Opera_', 'Concert_'],
            'genres': ['classical', 'opera', 'chamber music', 'symphony', 'baroque'],
            'audio_profile': {
                'danceability': (0.2, 0.1), 'energy': (0.3, 0.15), 'valence': (0.4, 0.2),
                'acousticness': (0.9, 0.05), 'instrumentalness': (0.8, 0.1), 'tempo': (100, 30)
            },
            'artists': ['Bach', 'Mozart', 'Beethoven', 'Chopin', 'Vivaldi']
        },
        'hiphop_head': {
            'name_patterns': ['MC_', 'Rap_', 'Hip_', 'Beats_', 'Flow_'],
            'genres': ['hip hop', 'rap', 'trap', 'old school hip hop', 'conscious rap'],
            'audio_profile': {
                'danceability': (0.7, 0.1), 'energy': (0.7, 0.1), 'valence': (0.6, 0.2),
                'speechiness': (0.3, 0.1), 'acousticness': (0.1, 0.05), 'tempo': (95, 15)
            },
            'artists': ['Kendrick Lamar', 'Drake', 'J. Cole', 'Kanye West', 'Travis Scott']
        }
    }
    
    archetype_names = list(music_archetypes.keys())

    # Generate remaining users (subtract 2 for the twins)
    remaining_users = n_users - 2
    for i in range(remaining_users):
        # Assign archetype (with some mixing for realism)
        primary_archetype = archetype_names[i % len(archetype_names)]
        archetype = music_archetypes[primary_archetype]

        # Generate realistic username
        name_prefix = np.random.choice(archetype['name_patterns'])
        username = f"{name_prefix}{np.random.randint(100, 999)}"

        # Get base audio features from archetype
        base_features = {}
        for feature, (mean, std) in archetype['audio_profile'].items():
            base_features[feature] = np.clip(np.random.normal(mean, std), 0, 1) if feature != 'tempo' else max(60, np.random.normal(mean, std))

        # Set preferred genres and artists
        preferred_genres = archetype['genres']
        preferred_artists = archetype['artists']
        
        # Generate complete audio features based on archetype
        audio_feature_values = []
        for track_idx in range(50):  # 50 tracks
            track_features = {}
            for feature in audio_features:
                if feature in base_features:
                    # Add some variation around the base value
                    variation = np.random.normal(0, 0.05)
                    if feature == 'tempo':
                        value = max(60, base_features[feature] + variation * 20)
                    elif feature == 'loudness':
                        value = base_features.get(feature, -10) + variation * 5
                    else:
                        value = np.clip(base_features[feature] + variation, 0, 1)
                else:
                    # Use archetype-appropriate defaults for missing features
                    if feature == 'loudness':
                        value = -8 + np.random.normal(0, 3)
                    elif feature == 'speechiness':
                        value = base_features.get('speechiness', 0.1 + np.random.beta(1, 4) * 0.3)
                    elif feature == 'liveness':
                        value = 0.1 + np.random.beta(1, 9) * 0.4
                    else:
                        value = np.random.beta(2, 2)
                track_features[feature] = value
            audio_feature_values.append(track_features)
        
        # Calculate summary statistics
        summary_stats = {}
        for feature in audio_features:
            values = [t[feature] for t in audio_feature_values]
            summary_stats[f'{feature}_mean'] = np.mean(values)
            summary_stats[f'{feature}_std'] = np.std(values)
            summary_stats[f'{feature}_median'] = np.median(values)
        
        # Generate realistic genre distribution based on archetype
        genre_distribution = {}
        total_weight = 0

        # Primary genres get high weights
        for genre in preferred_genres:
            weight = np.random.uniform(0.15, 0.35)
            genre_distribution[genre] = weight
            total_weight += weight

        # Add some secondary genres with lower weights
        all_genres = ['pop', 'rock', 'hip hop', 'electronic', 'indie', 'jazz',
                     'classical', 'metal', 'r&b', 'country', 'latin', 'alternative',
                     'house', 'techno', 'dubstep', 'trance', 'garage rock', 'post-punk',
                     'opera', 'chamber music', 'symphony', 'baroque', 'trap', 'conscious rap']

        secondary_genres = [g for g in all_genres if g not in preferred_genres]
        for _ in range(np.random.randint(1, 4)):  # 1-3 secondary genres
            genre = np.random.choice(secondary_genres)
            if genre not in genre_distribution:
                weight = np.random.uniform(0.02, 0.08)
                genre_distribution[genre] = weight
                total_weight += weight

        # Normalize weights
        genre_distribution = {k: v/total_weight for k, v in genre_distribution.items()}
        
        # Generate listening patterns based on archetype
        hour_distribution = np.zeros(24)
        if primary_archetype == 'classical':
            peak_hours = [10, 15, 19]  # Classical listeners prefer daytime
        elif primary_archetype == 'edm_lover':
            peak_hours = [22, 23, 0, 1]  # EDM lovers are night owls
        elif primary_archetype == 'indie_rock':
            peak_hours = [16, 18, 21]  # Indie rock fans prefer evening
        else:  # hiphop_head
            peak_hours = [12, 17, 20]  # Hip-hop fans spread throughout day

        for hour in peak_hours:
            hour_distribution[hour] = np.random.uniform(0.1, 0.3)
        # Add some random listening at other times
        for _ in range(3):
            random_hour = np.random.randint(0, 24)
            hour_distribution[random_hour] += np.random.uniform(0.02, 0.08)
        hour_distribution = hour_distribution / hour_distribution.sum()
        
        # Generate realistic track names based on archetype
        def generate_track_names(archetype_name, count):
            track_patterns = {
                'edm_lover': ['Beat Drop', 'Electric Nights', 'Bass Line', 'Rave On', 'Synth Wave'],
                'indie_rock': ['Midnight Drive', 'Coffee Shop', 'Vinyl Dreams', 'City Lights', 'Garage Sessions'],
                'classical': ['Symphony No.', 'Concerto in', 'Prelude', 'Sonata', 'Etude'],
                'hiphop_head': ['Street Dreams', 'Flow State', 'Beats & Rhymes', 'Urban Tales', 'Mic Check']
            }
            patterns = track_patterns.get(archetype_name, ['Song', 'Track', 'Music'])
            return [f"{np.random.choice(patterns)} {j+1}" for j in range(count)]

        track_names = generate_track_names(primary_archetype, 30)

        # Create user data with realistic information
        user_data = {
            'user_id': username,
            'display_name': username.replace('_', ' '),
            'music_archetype': primary_archetype,
            'collected_at': datetime.now().isoformat(),
            'audio_features': audio_feature_values,
            'summary_stats': summary_stats,
            'genre_distribution': genre_distribution,
            'listening_patterns': {
                'hour_distribution': hour_distribution.tolist(),
                'listening_diversity': np.random.uniform(0.3, 0.8)
            },
            'top_tracks': {
                'short_term': [{'id': f'track_{j}', 'name': track_names[j], 'popularity': np.random.randint(40, 95)} for j in range(10)],
                'medium_term': [{'id': f'track_{j}', 'name': track_names[j+10], 'popularity': np.random.randint(30, 90)} for j in range(10)],
                'long_term': [{'id': f'track_{j}', 'name': track_names[j+20], 'popularity': np.random.randint(20, 85)} for j in range(10)]
            },
            'top_artists': {
                'short_term': [{'id': f'artist_{j}', 'name': preferred_artists[j % len(preferred_artists)], 'popularity': np.random.randint(60, 100), 'followers': np.random.randint(100000, 10000000)} for j in range(5)],
                'medium_term': [{'id': f'artist_{j}', 'name': preferred_artists[j % len(preferred_artists)], 'popularity': np.random.randint(50, 95), 'followers': np.random.randint(50000, 8000000)} for j in range(5)],
                'long_term': [{'id': f'artist_{j}', 'name': preferred_artists[j % len(preferred_artists)], 'popularity': np.random.randint(40, 90), 'followers': np.random.randint(10000, 5000000)} for j in range(5)]
            }
        }
        
        sample_users.append(user_data)
    
    return sample_users

def create_music_taste_twins() -> List[Dict]:
    """Create two users with highly similar music preferences for demo."""

    # Shared EDM profile characteristics
    shared_edm_profile = {
        'danceability': 0.85,
        'energy': 0.92,
        'valence': 0.78,
        'acousticness': 0.08,
        'instrumentalness': 0.65,
        'tempo': 128,
        'loudness': -6,
        'speechiness': 0.12,
        'liveness': 0.15
    }

    # Shared genres and artists
    shared_genres = ['electronic', 'house', 'techno', 'edm', 'progressive house']
    shared_artists = ['Calvin Harris', 'Deadmau5', 'Swedish House Mafia', 'Avicii', 'Tiësto']

    twins = []

    for i, (username, display_name) in enumerate([
        ('DJ_Beat_Master', 'DJ Beat Master'),
        ('Rave_King_420', 'Rave King 420')
    ]):

        # Create very similar audio features (small variations)
        audio_feature_values = []
        for track_idx in range(50):
            track_features = {}
            for feature, base_value in shared_edm_profile.items():
                # Add tiny variations (±5%) to make them similar but not identical
                variation = np.random.normal(0, 0.02)  # Very small variation
                if feature == 'tempo':
                    value = max(60, base_value + variation * 10)
                elif feature == 'loudness':
                    value = base_value + variation * 2
                else:
                    value = np.clip(base_value + variation, 0, 1)
                track_features[feature] = value
            audio_feature_values.append(track_features)

        # Calculate summary stats
        summary_stats = {}
        audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

        for feature in audio_features:
            values = [track[feature] for track in audio_feature_values]
            summary_stats[f'{feature}_mean'] = np.mean(values)
            summary_stats[f'{feature}_std'] = np.std(values)
            summary_stats[f'{feature}_median'] = np.median(values)

        # Create nearly identical genre distribution
        genre_distribution = {
            'electronic': 0.35 + np.random.normal(0, 0.02),
            'house': 0.25 + np.random.normal(0, 0.02),
            'techno': 0.20 + np.random.normal(0, 0.02),
            'edm': 0.15 + np.random.normal(0, 0.02),
            'progressive house': 0.05 + np.random.normal(0, 0.01)
        }
        # Normalize
        total = sum(genre_distribution.values())
        genre_distribution = {k: v/total for k, v in genre_distribution.items()}

        # Similar listening patterns (night owls)
        hour_distribution = np.zeros(24)
        peak_hours = [21, 22, 23, 0, 1]  # Late night EDM sessions
        for hour in peak_hours:
            hour_distribution[hour] = np.random.uniform(0.15, 0.25)
        # Add some variation
        for _ in range(2):
            random_hour = np.random.randint(18, 24)
            hour_distribution[random_hour] += np.random.uniform(0.02, 0.05)
        hour_distribution = hour_distribution / hour_distribution.sum()

        # Create track names
        edm_tracks = [
            'Electric Pulse', 'Bass Drop Revolution', 'Neon Nights', 'Synth Storm',
            'Beat Machine', 'Digital Dreams', 'Rave Anthem', 'Club Crusher',
            'Electronic Euphoria', 'Dance Floor Fire', 'Laser Light Show', 'Beat Therapy',
            'Midnight Rave', 'Electric Energy', 'Bass Line Madness', 'Synth Wave',
            'Club Banger', 'Dance Revolution', 'Electronic Paradise', 'Beat Drop',
            'Rave Nation', 'Digital Beats', 'Electric Storm', 'Bass Thunder',
            'Synth Magic', 'Dance Fever', 'Electronic Vibes', 'Beat Factory',
            'Rave Dreams', 'Digital Paradise'
        ]

        user_data = {
            'user_id': username,
            'display_name': display_name,
            'music_archetype': 'edm_lover',
            'is_taste_twin': True,  # Mark as taste twin for demo
            'twin_pair_id': 'edm_twins_demo',
            'collected_at': datetime.now().isoformat(),
            'audio_features': audio_feature_values,
            'summary_stats': summary_stats,
            'genre_distribution': genre_distribution,
            'listening_patterns': {
                'hour_distribution': hour_distribution.tolist(),
                'listening_diversity': 0.65 + np.random.normal(0, 0.05)
            },
            'top_tracks': {
                'short_term': [{'id': f'track_{j}', 'name': edm_tracks[j], 'popularity': np.random.randint(70, 95)} for j in range(10)],
                'medium_term': [{'id': f'track_{j}', 'name': edm_tracks[j+10], 'popularity': np.random.randint(65, 90)} for j in range(10)],
                'long_term': [{'id': f'track_{j}', 'name': edm_tracks[j+20], 'popularity': np.random.randint(60, 85)} for j in range(10)]
            },
            'top_artists': {
                'short_term': [{'id': f'artist_{j}', 'name': shared_artists[j % len(shared_artists)], 'popularity': np.random.randint(80, 100), 'followers': np.random.randint(5000000, 20000000)} for j in range(5)],
                'medium_term': [{'id': f'artist_{j}', 'name': shared_artists[j % len(shared_artists)], 'popularity': np.random.randint(75, 95), 'followers': np.random.randint(3000000, 15000000)} for j in range(5)],
                'long_term': [{'id': f'artist_{j}', 'name': shared_artists[j % len(shared_artists)], 'popularity': np.random.randint(70, 90), 'followers': np.random.randint(1000000, 10000000)} for j in range(5)]
            }
        }

        twins.append(user_data)

    return twins

def calculate_cluster_centroids(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Calculate cluster centroids."""
    unique_labels = np.unique(labels)
    centroids = np.zeros((len(unique_labels), features.shape[1]))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        centroids[i] = np.mean(features[mask], axis=0)
    
    return centroids

def format_duration(milliseconds: int) -> str:
    """Format duration from milliseconds to readable format."""
    seconds = milliseconds // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"

def get_feature_display_name(feature: str) -> str:
    """Get human-readable display name for feature."""
    display_names = {
        'danceability_mean': 'Danceability',
        'energy_mean': 'Energy',
        'valence_mean': 'Musical Positivity',
        'acousticness_mean': 'Acousticness',
        'instrumentalness_mean': 'Instrumentalness',
        'speechiness_mean': 'Speechiness',
        'liveness_mean': 'Liveness',
        'loudness_mean': 'Loudness',
        'tempo_mean': 'Tempo',
        'genre_diversity': 'Genre Diversity',
        'listening_diversity': 'Listening Diversity',
        'artist_consistency': 'Artist Loyalty',
        'peak_listening_hour': 'Peak Listening Hour',
        'weekend_ratio': 'Weekend Listening Ratio'
    }
    
    return display_names.get(feature, feature.replace('_', ' ').title())

def validate_data_quality(data: pd.DataFrame, required_features: List[str]) -> Dict[str, Any]:
    """Validate data quality and return report."""
    report = {
        'total_records': len(data),
        'missing_features': [],
        'missing_values': {},
        'data_types': {},
        'warnings': []
    }
    
    # Check for required features
    for feature in required_features:
        if feature not in data.columns:
            report['missing_features'].append(feature)
    
    # Check missing values
    for col in data.columns:
        missing_count = data[col].isnull().sum()
        if missing_count > 0:
            report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(data) * 100)
            }
    
    # Check data types
    for col in data.columns:
        report['data_types'][col] = str(data[col].dtype)
    
    # Add warnings
    if report['missing_features']:
        report['warnings'].append(f"Missing {len(report['missing_features'])} required features")
    
    if len(data) < 10:
        report['warnings'].append("Very few records - results may not be reliable")
    
    high_missing = [col for col, info in report['missing_values'].items() 
                   if info['percentage'] > 50]
    if high_missing:
        report['warnings'].append(f"{len(high_missing)} features have >50% missing values")
    
    return report
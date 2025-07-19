import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import logging

from .audio_features import AudioFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserProfileBuilder:
    def __init__(self, feature_extractor: Optional[AudioFeatureExtractor] = None):
        self.feature_extractor = feature_extractor or AudioFeatureExtractor()
        self.feature_selector = None
        self.pca = None
        self.selected_features = None
        
    def build_profiles(self, users_data: List[Dict], 
                      use_feature_selection: bool = False,
                      n_features: Optional[int] = None) -> Tuple[pd.DataFrame, List[str]]:
        """Build comprehensive user profiles from raw data."""
        logger.info("Building user profiles...")
        
        # Extract features
        X, user_ids = self.feature_extractor.fit_transform(users_data)
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(X, columns=self.feature_extractor.feature_names)
        df['user_id'] = user_ids
        
        # Apply feature selection if requested
        if use_feature_selection and n_features:
            X_selected = self._select_features(X, n_features)
            selected_columns = [self.feature_extractor.feature_names[i] 
                              for i in self.selected_features]
            df_selected = pd.DataFrame(X_selected, columns=selected_columns)
            df_selected['user_id'] = user_ids
            df = df_selected
        
        # Add cluster assignment placeholder
        df['cluster_assignment'] = -1
        
        # Add metadata
        for i, user_data in enumerate(users_data):
            if user_data['user_id'] in df['user_id'].values:
                idx = df[df['user_id'] == user_data['user_id']].index[0]
                df.loc[idx, 'display_name'] = user_data.get('display_name', user_data['user_id'])
                
        return df, self.feature_extractor.feature_names
    
    def _select_features(self, X: np.ndarray, n_features: int) -> np.ndarray:
        """Select top n features using mutual information."""
        logger.info(f"Selecting top {n_features} features...")
        
        # Create dummy target for feature selection (can be replaced with actual clusters later)
        y = np.random.randint(0, 5, size=X.shape[0])
        
        # Use mutual information for feature selection
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Store selected feature indices
        self.selected_features = selector.get_support(indices=True)
        self.feature_selector = selector
        
        return X_selected
    
    def create_profile_summary(self, user_profile: pd.Series) -> Dict:
        """Create a human-readable summary of a user profile."""
        summary = {
            'user_id': user_profile['user_id'],
            'display_name': user_profile.get('display_name', user_profile['user_id']),
            'music_characteristics': {},
            'listening_habits': {},
            'preferences': {}
        }
        
        # Music characteristics
        audio_features = [
            'energy_mean', 'valence_mean', 'danceability_mean',
            'acousticness_mean', 'instrumentalness_mean'
        ]
        
        for feature in audio_features:
            if feature in user_profile:
                feature_name = feature.replace('_mean', '').capitalize()
                value = user_profile[feature]
                
                # Convert to descriptive level
                if value < 0.3:
                    level = "Low"
                elif value < 0.7:
                    level = "Medium"
                else:
                    level = "High"
                    
                summary['music_characteristics'][feature_name] = {
                    'value': float(value),
                    'level': level
                }
        
        # Listening habits
        if 'peak_listening_hour' in user_profile:
            hour = int(user_profile['peak_listening_hour'])
            if 6 <= hour < 12:
                time_of_day = "Morning"
            elif 12 <= hour < 18:
                time_of_day = "Afternoon"
            elif 18 <= hour < 24:
                time_of_day = "Evening"
            else:
                time_of_day = "Night"
            summary['listening_habits']['peak_time'] = f"{hour}:00 ({time_of_day})"
        
        if 'weekend_ratio' in user_profile:
            summary['listening_habits']['weekend_preference'] = float(user_profile['weekend_ratio'])
        
        if 'listening_diversity' in user_profile:
            diversity = float(user_profile['listening_diversity'])
            if diversity < 0.3:
                diversity_level = "Focused (prefers familiar tracks)"
            elif diversity < 0.7:
                diversity_level = "Balanced"
            else:
                diversity_level = "Explorative (enjoys discovering new music)"
            summary['listening_habits']['diversity'] = diversity_level
        
        # Genre preferences
        genre_features = [col for col in user_profile.index if col.startswith('genre_') and not col.endswith('_diversity')]
        top_genres = []
        
        for genre_col in genre_features:
            if user_profile[genre_col] > 0:
                genre_name = genre_col.replace('genre_', '').replace('_', ' ').title()
                top_genres.append((genre_name, float(user_profile[genre_col])))
        
        top_genres.sort(key=lambda x: x[1], reverse=True)
        summary['preferences']['top_genres'] = [
            {'genre': genre, 'weight': weight} 
            for genre, weight in top_genres[:5]
        ]
        
        # Artist diversity
        if 'artist_consistency' in user_profile:
            consistency = float(user_profile['artist_consistency'])
            if consistency < 0.3:
                consistency_desc = "Highly varied (explores many different artists)"
            elif consistency < 0.7:
                consistency_desc = "Moderate (balanced between favorites and new discoveries)"
            else:
                consistency_desc = "Consistent (strong loyalty to favorite artists)"
            summary['preferences']['artist_loyalty'] = consistency_desc
        
        return summary
    
    def calculate_profile_statistics(self, profiles_df: pd.DataFrame) -> Dict:
        """Calculate statistics across all user profiles."""
        stats = {}
        
        # Remove non-numeric columns
        numeric_df = profiles_df.select_dtypes(include=[np.number])
        
        # Basic statistics
        stats['summary'] = {
            'total_users': len(profiles_df),
            'total_features': len(numeric_df.columns),
            'feature_correlations': {}
        }
        
        # Feature statistics
        stats['feature_stats'] = {}
        for col in numeric_df.columns:
            if col not in ['cluster_assignment']:
                stats['feature_stats'][col] = {
                    'mean': float(numeric_df[col].mean()),
                    'std': float(numeric_df[col].std()),
                    'min': float(numeric_df[col].min()),
                    'max': float(numeric_df[col].max()),
                    'median': float(numeric_df[col].median())
                }
        
        # Top correlated features
        correlation_matrix = numeric_df.corr()
        correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.5:  # Only strong correlations
                    correlations.append({
                        'feature1': feature1,
                        'feature2': feature2,
                        'correlation': float(corr_value)
                    })
        
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        stats['feature_correlations'] = correlations[:20]  # Top 20 correlations
        
        return stats
    
    def reduce_dimensions(self, profiles_df: pd.DataFrame, n_components: int = 2) -> np.ndarray:
        """Reduce profile dimensions using PCA."""
        # Select only numeric features
        feature_cols = [col for col in profiles_df.columns 
                       if col not in ['user_id', 'display_name', 'cluster_assignment']]
        X = profiles_df[feature_cols].values
        
        # Apply PCA
        self.pca = PCA(n_components=n_components)
        X_reduced = self.pca.fit_transform(X)
        
        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}")
        
        return X_reduced
    
    def save_profiles(self, profiles_df: pd.DataFrame, output_path: str):
        """Save user profiles to file."""
        profiles_df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(profiles_df)} user profiles to {output_path}")
    
    def load_profiles(self, input_path: str) -> pd.DataFrame:
        """Load user profiles from file."""
        profiles_df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(profiles_df)} user profiles from {input_path}")
        return profiles_df
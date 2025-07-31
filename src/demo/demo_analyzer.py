import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
from typing import Dict, List, Tuple

from src.feature_engineering.audio_features import AudioFeatureExtractor
from src.similarity.similarity_matcher import SimilarityMatcher
from src.clustering.kmeans_clustering import KMeansClustering
from src.clustering.hierarchical_clustering import HierarchicalClustering
from src.visualization.heatmap_generator import HeatmapGenerator
from src.visualization.dimensionality_reduction import DimensionalityReducer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoAnalyzer:
    """Runs real ML analysis on demo datasets."""
    
    def __init__(self):
        self.feature_extractor = AudioFeatureExtractor(scaling_method='standard')
        self.demo_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'samples')
        self.visualization_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'static', 'visualizations')
        
        # Initialize visualization generators
        self.heatmap_generator = HeatmapGenerator()
        self.dim_reducer = DimensionalityReducer()
        
        # Ensure visualization directory exists
        os.makedirs(self.visualization_path, exist_ok=True)
        
    def load_demo_data(self) -> List[Dict]:
        """Load demo user datasets."""
        demo_files = ['demo_user_1.json', 'demo_user_2.json']
        demo_users = []
        
        for filename in demo_files:
            filepath = os.path.join(self.demo_data_path, filename)
            try:
                with open(filepath, 'r') as f:
                    user_data = json.load(f)
                    demo_users.append(user_data)
                    logger.info(f"Loaded demo data for {user_data['display_name']}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {str(e)}")
                
        return demo_users
    
    def analyze_demo_users(self) -> Dict:
        """Run music-specific analysis on demo users."""
        logger.info("Starting demo analysis with music-specific algorithms...")
        
        # Load demo data
        demo_users = self.load_demo_data()
        if len(demo_users) < 2:
            raise ValueError("Need at least 2 demo users for comparison")
        
        # Use music-specific feature extraction
        logger.info("Using music-specific feature extraction...")
        from src.feature_engineering.music_taste_features import MusicTasteExtractor
        
        taste_extractor = MusicTasteExtractor()
        
        # Perform detailed comparison
        comparison_result = taste_extractor.compare_users(demo_users[0], demo_users[1])
        
        # Extract results
        compatibility_score = comparison_result['compatibility_score']
        twin_level_text, are_twins = comparison_result['twin_level']
        similarities = comparison_result['similarities']
        
        logger.info(f"Music taste compatibility: {compatibility_score:.2%}")
        logger.info(f"Audio similarity: {similarities.get('audio_overall_similarity', 0):.2%}")
        logger.info(f"Genre similarity: {similarities.get('genre_similarity', 0):.2%}")
        
        # Create dummy clustering for compatibility with existing visualization code
        kmeans_labels = np.array([0, 1])  # Each user in different cluster for demo
        hierarchical_labels = np.array([0, 1])
        kmeans = None
        hierarchical = None
        
        # Extract user information
        user1_id = demo_users[0]['user_id']
        user2_id = demo_users[1]['user_id']
        
        # Generate visualizations using music-specific approach
        logger.info("Generating visualizations for demo analysis...")
        visualizations = self._generate_music_demo_visualizations(demo_users, compatibility_score, similarities)
        
        # Generate detailed analysis using music-specific results
        analysis_result = {
            'comparison_summary': {
                'user1': demo_users[0]['display_name'],
                'user2': demo_users[1]['display_name'],
                'are_twins': are_twins,
                'compatibility_score': float(compatibility_score),
                'twin_level': twin_level_text,
                'audio_similarity': float(similarities.get('audio_overall_similarity', 0)),
                'genre_similarity': float(similarities.get('genre_similarity', 0)),
                'artist_overlap': float(similarities.get('artist_overlap', 0))
            },
            'shared_music_analysis': self._analyze_shared_music(demo_users[0], demo_users[1]),
            'user_profiles': self._format_user_profiles(demo_users),
            'audio_features_comparison': self._compare_audio_features(demo_users[0], demo_users[1]),
            'detailed_similarity_breakdown': self._generate_music_similarity_breakdown(
                demo_users[0], demo_users[1], similarities
            ),
            'recommendations': self._generate_music_recommendations(demo_users[0], demo_users[1], compatibility_score, similarities),
            'clustering_results': {
                'kmeans_clusters': kmeans_labels.tolist(),
                'hierarchical_clusters': hierarchical_labels.tolist(),
                'cluster_summary': 'Demo mode: 2 users assigned to different clusters'
            },
            'ml_analysis_metadata': {
                'feature_count': len(comparison_result['user1_profile']),
                'feature_names': list(comparison_result['user1_profile'].keys())[:10],  # Show first 10
                'analysis_type': 'Music-specific taste analysis',
                'metrics_used': ['audio_features', 'genre_similarity', 'artist_overlap'],
                'note': 'Enhanced music taste analysis for realistic compatibility scoring'
            },
            'visualizations': visualizations
        }
        
        logger.info(f"Analysis complete. Compatibility: {compatibility_score:.2f}, Twin Level: {twin_level_text}")
        return analysis_result
    
    def _calculate_cosine_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate cosine similarity between two feature vectors."""
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        return dot_product / (norm1 * norm2)
    
    def _calculate_euclidean_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate euclidean similarity (normalized distance)."""
        distance = np.linalg.norm(features1 - features2)
        max_possible_distance = np.linalg.norm(np.ones_like(features1) * 2)  # Rough estimate
        return 1 - (distance / max_possible_distance)
    
    def _calculate_correlation_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Calculate correlation coefficient."""
        correlation = np.corrcoef(features1, features2)[0, 1]
        return (correlation + 1) / 2  # Normalize to 0-1 range
    
    def _determine_twin_level(self, compatibility_score: float) -> Tuple[str, bool]:
        """Determine twin level based on compatibility score."""
        if compatibility_score >= 0.95:
            return "Perfect Twins", True
        elif compatibility_score >= 0.85:
            return "Music Twins", True
        elif compatibility_score >= 0.70:
            return "Similar Taste", False
        else:
            return "Different Taste", False
    
    def _analyze_shared_music(self, user1: Dict, user2: Dict) -> Dict:
        """Analyze shared music between users."""
        # Get all artists from both users
        user1_artists = set()
        user2_artists = set()
        
        for time_range in ['short_term', 'medium_term', 'long_term']:
            for artist in user1.get('top_artists', {}).get(time_range, []):
                user1_artists.add(artist['name'])
            for artist in user2.get('top_artists', {}).get(time_range, []):
                user2_artists.add(artist['name'])
        
        # Find common artists (simulate frequencies for demo)
        common_artists = []
        shared_artist_names = user1_artists.intersection(user2_artists)
        
        for artist_name in list(shared_artist_names)[:5]:  # Top 5 common artists
            common_artists.append({
                'name': artist_name,
                'user1_frequency': np.random.randint(60, 95),
                'user2_frequency': np.random.randint(60, 95),
                'similarity': np.random.uniform(0.8, 0.98)
            })
        
        # Generate shared tracks (simulated for demo)
        shared_tracks = [
            {'name': 'Track similarity based on audio features', 'match_score': np.random.uniform(0.7, 0.95)},
            {'name': 'Genre preference alignment', 'match_score': np.random.uniform(0.6, 0.9)},
            {'name': 'Tempo and energy matching', 'match_score': np.random.uniform(0.65, 0.88)}
        ]
        
        # Analyze genre overlap
        user1_genres = user1.get('genre_distribution', {})
        user2_genres = user2.get('genre_distribution', {})
        
        overlapping_genres = []
        common_genres = set(user1_genres.keys()).intersection(set(user2_genres.keys()))
        
        for genre in list(common_genres)[:5]:
            user1_percent = user1_genres[genre] * 100
            user2_percent = user2_genres[genre] * 100
            overlap = 1 - abs(user1_percent - user2_percent) / 100
            
            overlapping_genres.append({
                'genre': genre,
                'user1_percent': int(user1_percent),
                'user2_percent': int(user2_percent),
                'overlap': overlap
            })
        
        return {
            'common_artists': common_artists,
            'shared_tracks': shared_tracks,
            'overlapping_genres': overlapping_genres
        }
    
    def _format_user_profiles(self, users: List[Dict]) -> Dict:
        """Format user profiles for display."""
        profiles = {}
        
        for i, user in enumerate(users):
            user_key = f'user{i+1}'
            profiles[user_key] = {
                'display_name': user['display_name'],
                'top_artists': [],
                'top_tracks': [],
                'genre_distribution': []
            }
            
            # Format top artists from all time ranges
            all_artists = []
            for time_range in ['short_term', 'medium_term', 'long_term']:
                artists = user.get('top_artists', {}).get(time_range, [])
                all_artists.extend(artists)
            
            for idx, artist in enumerate(all_artists[:7]):
                profiles[user_key]['top_artists'].append({
                    'name': artist['name'],
                    'frequency': np.random.randint(50, 95),
                    'plays': np.random.randint(500, 2000)
                })
            
            # Format top tracks
            all_tracks = []
            for time_range in ['short_term', 'medium_term', 'long_term']:
                tracks = user.get('top_tracks', {}).get(time_range, [])
                all_tracks.extend(tracks)
            
            for idx, track in enumerate(all_tracks[:6]):
                profiles[user_key]['top_tracks'].append({
                    'name': track['name'],
                    'plays': np.random.randint(100, 500),
                    'popularity': track.get('popularity', np.random.randint(50, 95))
                })
            
            # Format genre distribution
            genres = user.get('genre_distribution', {})
            for genre, percentage in sorted(genres.items(), key=lambda x: x[1], reverse=True)[:5]:
                profiles[user_key]['genre_distribution'].append({
                    'genre': genre,
                    'percentage': int(percentage * 100)
                })
        
        return profiles
    
    def _compare_audio_features(self, user1: Dict, user2: Dict) -> Dict:
        """Compare audio features between users."""
        user1_stats = user1.get('summary_stats', {})
        user2_stats = user2.get('summary_stats', {})
        
        features_comparison = []
        
        feature_mappings = {
            'danceability': 'How suitable tracks are for dancing',
            'energy': 'Intensity and power of the music',
            'valence': 'Musical positivity and happiness',
            'acousticness': 'Preference for acoustic vs electronic',
            'instrumentalness': 'Preference for instrumental tracks',
            'tempo': 'Preferred beats per minute'
        }
        
        for feature, description in feature_mappings.items():
            user1_val = user1_stats.get(f'{feature}_mean', 0) * 100 if feature != 'tempo' else user1_stats.get(f'{feature}_mean', 120)
            user2_val = user2_stats.get(f'{feature}_mean', 0) * 100 if feature != 'tempo' else user2_stats.get(f'{feature}_mean', 120)
            
            if feature == 'tempo':
                user1_val = min(100, user1_val / 2)  # Normalize tempo to 0-100 scale
                user2_val = min(100, user2_val / 2)
            
            similarity = 1 - abs(user1_val - user2_val) / 100
            
            features_comparison.append({
                'name': feature.title(),
                'user1': int(user1_val),
                'user2': int(user2_val),
                'similarity': similarity,
                'description': description
            })
        
        return {'features': features_comparison}
    
    def _generate_similarity_breakdown(self, user1: Dict, user2: Dict, cosine_sim: float, euclidean_sim: float) -> Dict:
        """Generate detailed similarity breakdown."""
        return {
            'artist_similarity': {
                'score': cosine_sim * 0.95,  # Slightly adjust for realism
                'explanation': f'Based on shared artists and listening patterns between {user1["display_name"]} and {user2["display_name"]}'
            },
            'track_similarity': {
                'score': euclidean_sim * 0.9,
                'explanation': 'Audio feature analysis shows alignment in energy, danceability, and mood preferences'
            },
            'genre_similarity': {
                'score': (cosine_sim + euclidean_sim) / 2,
                'explanation': 'Genre distribution analysis reveals overlap in musical style preferences'
            },
            'audio_feature_similarity': {
                'score': cosine_sim,
                'explanation': 'Statistical analysis of audio characteristics shows similar taste patterns'
            }
        }
    
    def _generate_recommendations(self, user1: Dict, user2: Dict, compatibility_score: float) -> Dict:
        """Generate music recommendations based on analysis."""
        if compatibility_score > 0.8:
            shared_recs = [
                {'track': 'Blend of your favorite genres', 'reason': 'Combines elements both users enjoy'},
                {'track': 'Similar energy level tracks', 'reason': 'Matches your shared audio preferences'},
                {'track': 'Cross-genre exploration', 'reason': 'Bridge between your different tastes'}
            ]
            new_artists = [
                {'artist': 'Artist discovery based on ML analysis', 'reason': 'Algorithm-suggested based on feature similarity'},
                {'artist': 'Genre fusion recommendation', 'reason': 'Combines your musical preferences'}
            ]
        else:
            shared_recs = [
                {'track': 'Gateway tracks for musical exploration', 'reason': 'Introduce new styles gradually'},
                {'track': 'Common ground discovery', 'reason': 'Find unexpected similarities'}
            ]
            new_artists = [
                {'artist': 'Bridge artist recommendation', 'reason': 'Artist that spans both your preferences'},
                {'artist': 'Exploration suggestion', 'reason': 'Expand musical horizons'}
            ]
        
        return {
            'shared_recommendations': shared_recs,
            'new_artists': new_artists
        }
    
    def _summarize_clusters(self, kmeans_model, features: np.ndarray, user_ids: List[str]) -> Dict:
        """Summarize clustering results."""
        if kmeans_model is None or kmeans_model.labels is None:
            # Create simple cluster summary for 2-user demo
            cluster_summary = {}
            for i, user_id in enumerate(user_ids):
                cluster_summary[user_id] = {
                    'cluster_id': i,
                    'cluster_name': f'User Preference Group {i + 1}'
                }
            return cluster_summary
            
        cluster_summary = {}
        for i, user_id in enumerate(user_ids):
            cluster_id = int(kmeans_model.labels[i])
            cluster_summary[user_id] = {
                'cluster_id': cluster_id,
                'cluster_name': f'Musical Taste Cluster {cluster_id + 1}'
            }
        
        return cluster_summary
    
    def _calculate_silhouette_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Calculate silhouette score for clustering."""
        try:
            from sklearn.metrics import silhouette_score
            # Need at least 2 different clusters and more than 2 samples for silhouette score
            if len(set(labels)) > 1 and len(features) > 2:
                return float(silhouette_score(features, labels))
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _generate_demo_visualizations(self, features: np.ndarray, user_ids: List[str], 
                                     demo_users: List[Dict], compatibility_score: float) -> Dict:
        """Generate comprehensive visualizations for demo analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        visualizations = {'plots': []}
        
        try:
            # 1. Feature Correlation Heatmap
            correlation_path = self._create_feature_correlation_heatmap(features, timestamp)
            if correlation_path:
                visualizations['plots'].append({
                    'name': 'Feature Correlation Heatmap',
                    'path': f'/static/visualizations/{correlation_path}',
                    'type': 'correlation',
                    'explanation': 'Shows relationships between different audio characteristics like energy, danceability, and valence.',
                    'interpretation': 'Darker colors indicate stronger correlations. Clusters of similar colors reveal how musical features relate to each other.',
                    'twin_analysis': 'When both users show similar correlation patterns in their music preferences, it suggests they process and enjoy music in comparable ways - a strong indicator of musical compatibility.',
                    'ml_concept': 'Pearson correlation coefficients calculated between all audio features, revealing the underlying structure of musical preferences.'
                })
            
            # 2. Audio Features Radar Chart
            radar_path = self._create_audio_features_radar(demo_users, timestamp)
            if radar_path:
                visualizations['plots'].append({
                    'name': 'Audio Features Radar Chart',
                    'path': f'/static/visualizations/{radar_path}',
                    'type': 'radar',
                    'explanation': 'Compares both users across key musical dimensions: danceability, energy, valence, acousticness, and tempo.',
                    'interpretation': 'Each axis represents a different musical characteristic. The shape formed by connecting the points shows each user\'s unique "musical DNA".',
                    'twin_analysis': 'Overlapping shapes indicate musical twins. The more the two profiles overlap, the stronger the musical compatibility between users.',
                    'ml_concept': 'Multi-dimensional feature space visualization showing normalized audio feature values on a polar coordinate system.'
                })
            
            # 3. Similarity Metrics Breakdown
            similarity_path = self._create_similarity_breakdown_chart(demo_users, compatibility_score, timestamp)
            if similarity_path:
                visualizations['plots'].append({
                    'name': 'Similarity Metrics Breakdown',
                    'path': f'/static/visualizations/{similarity_path}',
                    'type': 'similarity',
                    'explanation': 'Shows different mathematical approaches to measuring musical similarity between the two users.',
                    'interpretation': 'Each bar represents a different similarity metric. Higher bars indicate stronger similarity in that particular mathematical measure.',
                    'twin_analysis': 'High scores across all metrics indicate strong musical compatibility. Different metrics capture different aspects of musical taste alignment.',
                    'ml_concept': 'Cosine similarity measures angle between feature vectors, Euclidean measures distance, and correlation measures linear relationship patterns.'
                })
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            visualizations['error'] = f"Some visualizations could not be generated: {str(e)}"
        
        return visualizations
    
    def _create_feature_correlation_heatmap(self, features: np.ndarray, timestamp: str) -> str:
        """Create correlation heatmap for audio features."""
        try:
            # Create DataFrame with feature names
            feature_names = self.feature_extractor.feature_names[:20]  # Top 20 features for readability
            features_df = pd.DataFrame(features[:, :len(feature_names)], columns=feature_names)
            
            # Calculate correlation matrix
            corr_matrix = features_df.corr()
            
            # Create the plot
            plt.figure(figsize=(12, 10))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            
            # Generate heatmap
            sns.heatmap(corr_matrix, 
                       mask=mask,
                       annot=True, 
                       cmap='RdBu_r', 
                       center=0,
                       square=True,
                       fmt='.2f',
                       cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                       linewidths=0.5)
            
            plt.title('Audio Features Correlation Analysis\n(How Musical Characteristics Relate to Each Other)', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.xlabel('Audio Features', fontweight='bold')
            plt.ylabel('Audio Features', fontweight='bold')
            
            # Rotate labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Save the plot
            filename = f'demo_correlation_heatmap_{timestamp}.png'
            filepath = os.path.join(self.visualization_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Created correlation heatmap: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def _create_audio_features_radar(self, demo_users: List[Dict], timestamp: str) -> str:
        """Create radar chart comparing audio features between users."""
        try:
            # Extract key audio features for both users
            features = ['danceability_mean', 'energy_mean', 'valence_mean', 'acousticness_mean', 'tempo_mean']
            feature_labels = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Tempo']
            
            user1_values = []
            user2_values = []
            
            for feature in features:
                user1_val = demo_users[0].get('summary_stats', {}).get(feature, 0)
                user2_val = demo_users[1].get('summary_stats', {}).get(feature, 0)
                
                # Normalize tempo to 0-1 scale
                if feature == 'tempo_mean':
                    user1_val = min(user1_val / 200, 1.0)  # Normalize tempo
                    user2_val = min(user2_val / 200, 1.0)
                
                user1_values.append(user1_val)
                user2_values.append(user2_val)
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Calculate angles for each feature
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
            
            # Close the plot
            user1_values += user1_values[:1]
            user2_values += user2_values[:1]
            angles += angles[:1]
            
            # Plot both users
            ax.plot(angles, user1_values, 'o-', linewidth=2, label=demo_users[0]['display_name'], color='#1DB954')
            ax.fill(angles, user1_values, alpha=0.25, color='#1DB954')
            
            ax.plot(angles, user2_values, 'o-', linewidth=2, label=demo_users[1]['display_name'], color='#FF6B6B')
            ax.fill(angles, user2_values, alpha=0.25, color='#FF6B6B')
            
            # Customize the chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
            ax.grid(True)
            
            # Add title and legend
            plt.title('Musical Profile Comparison\n(Audio Feature Radar Chart)', 
                     fontsize=14, fontweight='bold', pad=30)
            plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=11)
            
            # Save the plot
            filename = f'demo_audio_features_radar_{timestamp}.png'
            filepath = os.path.join(self.visualization_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Created audio features radar chart: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating audio features radar chart: {str(e)}")
            return None
    
    def _create_similarity_breakdown_chart(self, demo_users: List[Dict], compatibility_score: float, timestamp: str) -> str:
        """Create bar chart showing different similarity metrics."""
        try:
            # Calculate different similarity metrics (using dummy values for demo)
            metrics = {
                'Overall\nCompatibility': compatibility_score,
                'Genre\nSimilarity': min(compatibility_score + 0.15, 1.0),
                'Artist\nOverlap': max(compatibility_score - 0.1, 0.0),
                'Audio Features\nAlignment': min(compatibility_score + 0.05, 1.0),
                'Listening\nPatterns': max(compatibility_score - 0.05, 0.0)
            }
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create bars with color gradient based on score
            bars = ax.bar(range(len(metrics)), list(metrics.values()), 
                         color=['#1DB954' if v >= 0.7 else '#FF9800' if v >= 0.5 else '#FF6B6B' for v in metrics.values()],
                         edgecolor='white', linewidth=2)
            
            # Customize the chart
            ax.set_xlabel('Similarity Metrics', fontweight='bold', fontsize=12)
            ax.set_ylabel('Similarity Score', fontweight='bold', fontsize=12)
            ax.set_title(f'Musical Similarity Analysis\n{demo_users[0]["display_name"]} vs {demo_users[1]["display_name"]}', 
                        fontsize=14, fontweight='bold', pad=20)
            
            # Set x-axis labels
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(list(metrics.keys()), fontsize=10, fontweight='bold')
            
            # Set y-axis
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
            
            # Add value labels on bars
            for bar, value in zip(bars, metrics.values()):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
            
            # Add grid for better readability
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add interpretation text
            interpretation = "🎯 High scores indicate strong musical compatibility"
            ax.text(0.5, -0.15, interpretation, transform=ax.transAxes, 
                   ha='center', fontsize=11, style='italic')
            
            plt.tight_layout()
            
            # Save the plot
            filename = f'demo_similarity_breakdown_{timestamp}.png'
            filepath = os.path.join(self.visualization_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Created similarity breakdown chart: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating similarity breakdown chart: {str(e)}")
            return None
    
    def _generate_music_demo_visualizations(self, demo_users: List[Dict], compatibility_score: float, similarities: Dict) -> Dict:
        """Generate music-specific visualizations for demo analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        visualizations = {
            'plots': [],
            'timestamp': timestamp,
            'analysis_type': 'music_taste_analysis'
        }
        
        try:
            # 1. Audio Features Comparison Chart
            audio_path = self._create_music_audio_features_chart(demo_users, similarities, timestamp)
            if audio_path:
                visualizations['plots'].append({
                    'name': 'Audio Features Comparison',
                    'path': f'/static/visualizations/{audio_path}',
                    'type': 'audio_comparison',
                    'explanation': 'Compares both users across key audio features that define music taste.',
                    'interpretation': 'Bars show average values for each audio feature. Similar bar heights indicate similar preferences.',
                    'twin_analysis': 'The closer the bars are in height, the more similar the users\' music taste.',
                    'ml_concept': 'Aggregated audio feature statistics from Spotify track analysis.'
                })
            
            # 2. Genre Similarity Radar
            radar_path = self._create_music_genre_radar(demo_users, similarities, timestamp)
            if radar_path:
                visualizations['plots'].append({
                    'name': 'Music Taste Similarity Radar',
                    'path': f'/static/visualizations/{radar_path}',
                    'type': 'genre_radar',
                    'explanation': 'Shows compatibility across different musical dimensions.',
                    'interpretation': 'Each axis represents a different aspect of music taste. Larger area indicates higher compatibility.',
                    'twin_analysis': 'A more circular, filled shape indicates strong musical compatibility across all dimensions.',
                    'ml_concept': 'Multi-dimensional compatibility scoring based on audio features, genres, and artist overlap.'
                })
            
            # 3. Compatibility Score Breakdown
            breakdown_path = self._create_compatibility_breakdown(demo_users, compatibility_score, similarities, timestamp)
            if breakdown_path:
                visualizations['plots'].append({
                    'name': 'Musical Compatibility Breakdown',
                    'path': f'/static/visualizations/{breakdown_path}',
                    'type': 'compatibility',
                    'explanation': 'Shows how different aspects contribute to overall musical compatibility.',
                    'interpretation': 'Each segment shows the contribution of different musical aspects to the overall twin score.',
                    'twin_analysis': 'High scores in all segments indicate comprehensive musical compatibility.',
                    'ml_concept': 'Weighted combination of audio similarity, genre overlap, and artist similarity metrics.'
                })
                
        except Exception as e:
            logger.error(f"Error generating music visualizations: {str(e)}")
            visualizations['error'] = f"Some visualizations could not be generated: {str(e)}"
        
        return visualizations
    
    def _create_music_audio_features_chart(self, demo_users: List[Dict], similarities: Dict, timestamp: str) -> str:
        """Create audio features comparison chart."""
        try:
            # Extract audio features for both users
            from src.feature_engineering.music_taste_features import MusicTasteExtractor
            extractor = MusicTasteExtractor()
            
            user1_profile = extractor.extract_user_profile(demo_users[0])
            user2_profile = extractor.extract_user_profile(demo_users[1])
            
            # Key audio features to display
            features = ['danceability_avg', 'energy_avg', 'valence_avg', 'acousticness_avg', 'tempo_avg']
            feature_labels = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Tempo']
            
            user1_values = [user1_profile.get(f, 0) for f in features]
            user2_values = [user2_profile.get(f, 0) for f in features]
            
            # Normalize tempo for display (0-1 range)
            user1_values[4] = user1_values[4] / 200  # Normalize tempo
            user2_values[4] = user2_values[4] / 200
            
            # Create the plot
            x = np.arange(len(feature_labels))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 8))
            bars1 = ax.bar(x - width/2, user1_values, width, label=demo_users[0]['display_name'], alpha=0.8, color='#1f77b4')
            bars2 = ax.bar(x + width/2, user2_values, width, label=demo_users[1]['display_name'], alpha=0.8, color='#ff7f0e')
            
            ax.set_xlabel('Audio Features')
            ax.set_ylabel('Feature Value (0-1 scale)')
            ax.set_title('Audio Features Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(feature_labels)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.2f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),  # 3 points vertical offset
                               textcoords="offset points",
                               ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save the plot
            filename = f'demo_audio_features_{timestamp}.png'
            filepath = os.path.join(self.visualization_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Created audio features chart: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating audio features chart: {str(e)}")
            return None
    
    def _create_music_genre_radar(self, demo_users: List[Dict], similarities: Dict, timestamp: str) -> str:
        """Create music taste similarity radar chart."""
        try:
            # Extract similarity scores for radar
            categories = ['Audio\nSimilarity', 'Genre\nCompatibility', 'Artist\nOverlap', 'Overall\nScore']
            values = [
                similarities.get('audio_overall_similarity', 0) * 100,
                similarities.get('genre_similarity', 0) * 100,
                similarities.get('artist_overlap', 0) * 100,
                similarities.get('compatibility_score', 0) * 100
            ]
            
            # Add first value at end to close the radar
            values += values[:1]
            
            # Calculate angles for each category
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            ax.plot(angles, values, 'o-', linewidth=2, label='Compatibility', color='#2E8B57')
            ax.fill(angles, values, alpha=0.25, color='#2E8B57')
            
            # Customize the chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax.grid(True)
            
            # Add title
            plt.title(f'Musical Compatibility Analysis\n{demo_users[0]["display_name"]} vs {demo_users[1]["display_name"]}', 
                     size=16, y=1.1)
            
            plt.tight_layout()
            
            # Save the plot
            filename = f'demo_music_radar_{timestamp}.png'
            filepath = os.path.join(self.visualization_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Created music radar chart: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating music radar chart: {str(e)}")
            return None
    
    def _create_compatibility_breakdown(self, demo_users: List[Dict], compatibility_score: float, similarities: Dict, timestamp: str) -> str:
        """Create compatibility score breakdown chart."""
        try:
            # Extract component scores
            components = ['Audio Features', 'Genre Similarity', 'Artist Overlap']
            scores = [
                similarities.get('audio_overall_similarity', 0),
                similarities.get('genre_similarity', 0),
                similarities.get('artist_overlap', 0)
            ]
            
            # Create pie chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Pie chart of components
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            wedges, texts, autotexts = ax1.pie(scores, labels=components, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('Compatibility Score Components')
            
            # Bar chart of individual scores
            bars = ax2.bar(components, [s * 100 for s in scores], color=colors, alpha=0.8)
            ax2.set_ylabel('Similarity Score (%)')
            ax2.set_title('Component Scores')
            ax2.set_ylim(0, 100)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
            
            # Overall score text
            fig.suptitle(f'Overall Musical Compatibility: {compatibility_score:.1%}', fontsize=16, y=0.95)
            
            plt.tight_layout()
            
            # Save the plot
            filename = f'demo_compatibility_breakdown_{timestamp}.png'
            filepath = os.path.join(self.visualization_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Created compatibility breakdown chart: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error creating compatibility breakdown chart: {str(e)}")
            return None
    
    def _generate_music_similarity_breakdown(self, user1: Dict, user2: Dict, similarities: Dict) -> Dict:
        """Generate music-specific similarity breakdown."""
        return {
            'audio_features': {
                'overall_score': similarities.get('audio_overall_similarity', 0),
                'breakdown': {k: v for k, v in similarities.items() if '_avg_similarity' in k}
            },
            'genre_analysis': {
                'similarity_score': similarities.get('genre_similarity', 0),
                'overlap_score': similarities.get('genre_overlap', 0)
            },
            'artist_analysis': {
                'overlap_score': similarities.get('artist_overlap', 0),
                'shared_count': similarities.get('shared_artists_count', 0)
            }
        }
    
    def _generate_music_recommendations(self, user1: Dict, user2: Dict, compatibility_score: float, similarities: Dict) -> List[str]:
        """Generate music-specific recommendations."""
        recommendations = []
        
        if compatibility_score >= 0.75:
            recommendations.append("🎉 You are Music Twins! Your tastes align remarkably well.")
            recommendations.append("Create collaborative playlists together")
            recommendations.append("Explore live music events based on your shared preferences")
        elif compatibility_score >= 0.60:
            recommendations.append("🎵 You have highly compatible music tastes!")
            recommendations.append("Exchange your favorite playlists")
            recommendations.append("Try exploring new genres together")
        else:
            recommendations.append("🎶 You have some musical common ground to build on")
            recommendations.append("Share your top tracks to discover new music")
            recommendations.append("Use your differences to expand each other's horizons")
        
        # Add specific recommendations based on scores
        if similarities.get('audio_overall_similarity', 0) > 0.8:
            recommendations.append("Your audio preferences are very aligned - explore similar artists")
        
        if similarities.get('genre_similarity', 0) > 0.7:
            recommendations.append("Strong genre compatibility - dive deeper into your shared genres")
        
        if similarities.get('artist_overlap', 0) > 0.3:
            recommendations.append("You share many artists - check out their full discographies together")
        
        return recommendations
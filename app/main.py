from flask import Flask, render_template, request, jsonify, session, send_from_directory
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import SpotifyCollector
from src.feature_engineering import AudioFeatureExtractor, UserProfileBuilder
from src.clustering import KMeansClustering
from src.similarity import SimilarityMatcher
from src.visualization import (
    DimensionalityReducer, HeatmapGenerator, 
    ClusterVisualizer, EDAVisualizer
)
from src.utils.helpers import load_json, save_json, create_sample_data
from config.spotify_config import SpotifyConfig

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dev-secret-key-change-in-production')

# Global variables for storing model and data
model_data = {
    'user_profiles': None,
    'clustering_model': None,
    'similarity_matcher': None,
    'dimensionality_reducer': None,
    'feature_names': None
}

@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')

@app.route('/demo')
def demo():
    """Demo mode with sample data."""
    # Generate sample data
    sample_users = create_sample_data(n_users=100)
    
    # Process data
    results = process_user_data(sample_users, is_demo=True)
    
    return jsonify(results)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze user's Spotify data."""
    try:
        data = request.get_json()
        username = data.get('username')
        
        if not username:
            return jsonify({'error': 'Username is required'}), 400
        
        # Collect user data
        collector = SpotifyCollector()
        user_data = collector.collect_user_data(username)
        
        # Load existing user data if available
        existing_data_path = os.path.join('data', 'processed', 'all_users.json')
        if os.path.exists(existing_data_path):
            all_users = load_json(existing_data_path)
            
            # Check if user already exists
            user_exists = any(u['user_id'] == user_data['user_id'] for u in all_users)
            if not user_exists:
                all_users.append(user_data)
                save_json(all_users, existing_data_path)
        else:
            all_users = [user_data]
            save_json(all_users, existing_data_path)
        
        # Process data
        results = process_user_data(all_users, target_user_id=user_data['user_id'])
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_user_data(users_data, target_user_id=None, is_demo=False):
    """Process user data and return analysis results."""
    # Feature engineering
    profile_builder = UserProfileBuilder()
    user_profiles, feature_names = profile_builder.build_profiles(users_data)
    
    # Store feature names
    model_data['feature_names'] = feature_names
    model_data['user_profiles'] = user_profiles
    
    # Get features for clustering
    feature_cols = [col for col in user_profiles.columns 
                   if col not in ['user_id', 'display_name', 'cluster_assignment']]
    X = user_profiles[feature_cols].values
    
    # Clustering
    clustering = KMeansClustering()
    cluster_labels = clustering.fit_predict(X)
    user_profiles['cluster'] = cluster_labels
    
    # Store clustering model
    model_data['clustering_model'] = clustering
    
    # Get cluster statistics
    cluster_stats = clustering.get_cluster_statistics(X, feature_cols)
    
    # Similarity matching
    matcher = SimilarityMatcher(metric='cosine')
    matcher.fit(X, user_profiles['user_id'].tolist(), cluster_labels)
    model_data['similarity_matcher'] = matcher
    
    # Dimensionality reduction
    dim_reducer = DimensionalityReducer()
    pca_results = dim_reducer.apply_pca(X, n_components=3)
    tsne_results = dim_reducer.apply_tsne(X, perplexity_values=[30])
    umap_results = dim_reducer.apply_umap(X, n_neighbors_values=[15], min_dist_values=[0.1])
    model_data['dimensionality_reducer'] = dim_reducer
    
    # Prepare results
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    results = {
        'summary': {
            'total_users': int(len(user_profiles)),
            'n_clusters': int(len(unique_labels)),
            'cluster_sizes': {int(label): int(count) for label, count in zip(unique_labels, counts)}
        },
        'visualizations': {},
        'is_demo': is_demo
    }
    
    # Generate visualizations
    vis_dir = os.path.join('static', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create various plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. PCA analysis
    dim_reducer.plot_pca_analysis(
        feature_cols, cluster_labels,
        save_path=os.path.join(vis_dir, f'pca_analysis_{timestamp}.png')
    )
    results['visualizations']['pca_analysis'] = f'pca_analysis_{timestamp}.png'
    
    # 2. t-SNE visualization
    dim_reducer.plot_tsne_comparison(
        cluster_labels,
        save_path=os.path.join(vis_dir, f'tsne_comparison_{timestamp}.png')
    )
    results['visualizations']['tsne_comparison'] = f'tsne_comparison_{timestamp}.png'
    
    # 3. UMAP visualization
    dim_reducer.plot_umap_comparison(
        cluster_labels,
        save_path=os.path.join(vis_dir, f'umap_comparison_{timestamp}.png')
    )
    results['visualizations']['umap_comparison'] = f'umap_comparison_{timestamp}.png'
    
    # 4. Methods comparison
    dim_reducer.plot_all_methods_comparison(
        cluster_labels,
        save_path=os.path.join(vis_dir, f'methods_comparison_{timestamp}.png')
    )
    results['visualizations']['methods_comparison'] = f'methods_comparison_{timestamp}.png'
    
    # 5. Feature correlation heatmap
    heatmap_gen = HeatmapGenerator()
    heatmap_gen.create_feature_correlation_heatmap(
        user_profiles[feature_cols[:20]],  # Top 20 features
        save_path=os.path.join(vis_dir, f'feature_correlation_{timestamp}.png')
    )
    results['visualizations']['feature_correlation'] = f'feature_correlation_{timestamp}.png'
    
    # 6. Cluster feature heatmap
    heatmap_gen.create_cluster_feature_heatmap(
        cluster_stats, feature_cols[:15],
        save_path=os.path.join(vis_dir, f'cluster_features_{timestamp}.png')
    )
    results['visualizations']['cluster_features'] = f'cluster_features_{timestamp}.png'
    
    # 7. EDA visualizations
    eda_viz = EDAVisualizer()
    eda_viz.create_statistical_summary(
        user_profiles, feature_cols,
        save_path=os.path.join(vis_dir, f'statistical_summary_{timestamp}.png')
    )
    results['visualizations']['statistical_summary'] = f'statistical_summary_{timestamp}.png'
    
    # If specific user is targeted
    if target_user_id:
        user_idx = user_profiles[user_profiles['user_id'] == target_user_id].index[0]
        user_cluster = cluster_labels[user_idx]
        
        # Find similar users
        similar_users = matcher.find_similar_users_with_explanation(
            target_user_id, top_n=10, feature_names=feature_cols
        )
        
        # Find taste twins
        taste_twins = matcher.find_taste_twins(target_user_id, similarity_threshold=0.85)
        
        results['user_analysis'] = {
            'user_id': target_user_id,
            'cluster': int(user_cluster),
            'cluster_size': int(np.sum(cluster_labels == user_cluster)),
            'similar_users': similar_users,
            'taste_twins': taste_twins,
            'profile_summary': profile_builder.create_profile_summary(
                user_profiles.iloc[user_idx]
            )
        }

    # Convert numpy types to Python native types for JSON serialization
    results = convert_numpy_types(results)

    return results

@app.route('/api/clusters')
def get_clusters():
    """Get cluster information."""
    if model_data['user_profiles'] is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    profiles = model_data['user_profiles']
    clusters = profiles.groupby('cluster').agg({
        'user_id': 'count',
        'energy_mean': 'mean',
        'valence_mean': 'mean',
        'danceability_mean': 'mean'
    }).to_dict('index')
    
    return jsonify(clusters)

@app.route('/api/similar/<user_id>')
def find_similar(user_id):
    """Find similar users for a specific user."""
    if model_data['similarity_matcher'] is None:
        return jsonify({'error': 'No model loaded'}), 400
    
    try:
        similar_users = model_data['similarity_matcher'].find_similar_users(
            user_id, top_n=20
        )
        return jsonify(similar_users)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/visualizations/<viz_type>')
def get_visualization(viz_type):
    """Get specific visualization data."""
    if model_data['dimensionality_reducer'] is None:
        return jsonify({'error': 'No data loaded'}), 400
    
    if viz_type == 'pca_3d':
        # Return 3D PCA data
        pca_data = model_data['dimensionality_reducer'].pca_results
        if pca_data and pca_data['n_components'] >= 3:
            data = {
                'coordinates': pca_data['transformed_data'][:, :3].tolist(),
                'labels': model_data['clustering_model'].labels.tolist(),
                'user_ids': model_data['user_profiles']['user_id'].tolist()
            }
            return jsonify(data)
    
    return jsonify({'error': 'Visualization type not found'}), 404

@app.route('/static/visualizations/<filename>')
def serve_visualization(filename):
    """Serve visualization images with proper headers."""
    try:
        viz_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'visualizations')
        return send_from_directory(viz_path, filename)
    except FileNotFoundError:
        return "Visualization not found", 404

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/visualizations', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Run app
    app.run(debug=True, port=8888)
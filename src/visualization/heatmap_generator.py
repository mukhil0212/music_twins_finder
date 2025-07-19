import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HeatmapGenerator:
    def __init__(self):
        self.default_cmap = 'coolwarm'
        self.diverging_cmap = 'RdBu'
        self.sequential_cmap = 'viridis'
        
    def create_feature_correlation_heatmap(self, features_df: pd.DataFrame,
                                         feature_names: Optional[List[str]] = None,
                                         save_path: Optional[str] = None,
                                         clustered: bool = True,
                                         figsize: Tuple[int, int] = (12, 10)):
        """Create comprehensive feature correlation heatmap."""
        # Select only numeric columns
        if feature_names:
            numeric_df = features_df[feature_names]
        else:
            numeric_df = features_df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()

        # Handle NaN and infinite values
        corr_matrix = corr_matrix.fillna(0)
        corr_matrix = corr_matrix.replace([np.inf, -np.inf], 0)

        # Apply hierarchical clustering if requested
        if clustered:
            # Compute linkage matrix
            linkage_matrix = linkage(corr_matrix, method='ward')
            
            # Get the order of features after clustering
            dendro = dendrogram(linkage_matrix, no_plot=True)
            clustered_order = dendro['leaves']
            
            # Reorder correlation matrix
            corr_matrix = corr_matrix.iloc[clustered_order, clustered_order]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap with annotations
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix,
                   mask=mask,
                   cmap=self.diverging_cmap,
                   center=0,
                   annot=True,
                   fmt='.2f',
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8, "label": "Correlation"},
                   ax=ax,
                   vmin=-1,
                   vmax=1)
        
        ax.set_title('Feature Correlation Matrix' + (' (Hierarchically Clustered)' if clustered else ''),
                    fontsize=16, pad=20)
        
        # Rotate labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature correlation heatmap to {save_path}")
        
        plt.close()
        
        return corr_matrix
    
    def create_user_similarity_heatmap(self, similarity_matrix: np.ndarray,
                                     user_ids: List[str],
                                     cluster_labels: Optional[np.ndarray] = None,
                                     save_path: Optional[str] = None,
                                     max_users: int = 100):
        """Create user-to-user similarity heatmap."""
        # Limit users if too many
        if len(user_ids) > max_users:
            logger.warning(f"Too many users ({len(user_ids)}). Limiting to {max_users} for visualization.")
            indices = np.random.choice(len(user_ids), max_users, replace=False)
            similarity_matrix = similarity_matrix[indices][:, indices]
            user_ids = [user_ids[i] for i in indices]
            if cluster_labels is not None:
                cluster_labels = cluster_labels[indices]
        
        # Sort by cluster if labels provided
        if cluster_labels is not None:
            sorted_indices = np.argsort(cluster_labels)
            similarity_matrix = similarity_matrix[sorted_indices][:, sorted_indices]
            user_ids = [user_ids[i] for i in sorted_indices]
            cluster_labels = cluster_labels[sorted_indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap=self.sequential_cmap, aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Similarity Score', rotation=270, labelpad=20)
        
        # Add cluster boundaries if provided
        if cluster_labels is not None:
            # Find cluster boundaries
            boundaries = []
            current_cluster = cluster_labels[0]
            for i, cluster in enumerate(cluster_labels):
                if cluster != current_cluster:
                    boundaries.append(i - 0.5)
                    current_cluster = cluster
            
            # Draw lines at boundaries
            for boundary in boundaries:
                ax.axhline(y=boundary, color='red', linewidth=2, alpha=0.7)
                ax.axvline(x=boundary, color='red', linewidth=2, alpha=0.7)
        
        ax.set_title('User Similarity Heatmap' + 
                    (' (Organized by Clusters)' if cluster_labels is not None else ''),
                    fontsize=16, pad=20)
        
        # Set labels
        if len(user_ids) <= 50:  # Only show labels if reasonable number
            ax.set_xticks(range(len(user_ids)))
            ax.set_yticks(range(len(user_ids)))
            ax.set_xticklabels(user_ids, rotation=90, ha='center')
            ax.set_yticklabels(user_ids)
        else:
            ax.set_xlabel('User Index')
            ax.set_ylabel('User Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved user similarity heatmap to {save_path}")
        
        plt.close()
    
    def create_cluster_feature_heatmap(self, cluster_stats: Dict,
                                     feature_names: List[str],
                                     save_path: Optional[str] = None,
                                     normalize: bool = True):
        """Create heatmap showing average feature values per cluster."""
        # Extract cluster means
        clusters = sorted(cluster_stats.keys())
        n_clusters = len(clusters)
        n_features = len(feature_names)
        
        # Create matrix of cluster means
        cluster_means = np.zeros((n_clusters, n_features))
        
        for i, cluster in enumerate(clusters):
            for j, feature in enumerate(feature_names):
                cluster_means[i, j] = cluster_stats[cluster]['feature_means'].get(feature, 0)
        
        # Normalize if requested
        if normalize:
            # Normalize each feature to [0, 1]
            for j in range(n_features):
                col = cluster_means[:, j]
                if np.std(col) > 0:
                    cluster_means[:, j] = (col - np.min(col)) / (np.max(col) - np.min(col))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(cluster_means,
                   xticklabels=feature_names,
                   yticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                   cmap='YlOrRd' if normalize else self.sequential_cmap,
                   annot=True,
                   fmt='.2f' if normalize else '.3f',
                   cbar_kws={"label": "Normalized Value" if normalize else "Feature Value"},
                   ax=ax)
        
        ax.set_title('Cluster Feature Heatmap' + (' (Normalized)' if normalize else ''),
                    fontsize=16, pad=20)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster feature heatmap to {save_path}")
        
        plt.close()
    
    def create_genre_distribution_heatmap(self, genre_data: Dict[str, Dict],
                                        cluster_labels: Optional[Dict[str, int]] = None,
                                        save_path: Optional[str] = None,
                                        top_n_genres: int = 20):
        """Create genre distribution heatmap."""
        # Collect all genres
        all_genres = set()
        for user_genres in genre_data.values():
            all_genres.update(user_genres.keys())
        
        # Get top genres by overall frequency
        genre_totals = {}
        for user_genres in genre_data.values():
            for genre, weight in user_genres.items():
                genre_totals[genre] = genre_totals.get(genre, 0) + weight
        
        top_genres = sorted(genre_totals.items(), key=lambda x: x[1], reverse=True)[:top_n_genres]
        top_genre_names = [g[0] for g in top_genres]
        
        if cluster_labels:
            # Create cluster-genre matrix
            n_clusters = max(cluster_labels.values()) + 1
            genre_matrix = np.zeros((n_clusters, len(top_genre_names)))
            cluster_counts = np.zeros(n_clusters)
            
            for user_id, genres in genre_data.items():
                if user_id in cluster_labels:
                    cluster = cluster_labels[user_id]
                    cluster_counts[cluster] += 1
                    for i, genre in enumerate(top_genre_names):
                        if genre in genres:
                            genre_matrix[cluster, i] += genres[genre]
            
            # Normalize by cluster size
            for i in range(n_clusters):
                if cluster_counts[i] > 0:
                    genre_matrix[i] /= cluster_counts[i]
            
            row_labels = [f'Cluster {i}' for i in range(n_clusters)]
        else:
            # Create user-genre matrix
            user_ids = list(genre_data.keys())[:50]  # Limit users
            genre_matrix = np.zeros((len(user_ids), len(top_genre_names)))
            
            for i, user_id in enumerate(user_ids):
                genres = genre_data[user_id]
                for j, genre in enumerate(top_genre_names):
                    if genre in genres:
                        genre_matrix[i, j] = genres[genre]
            
            row_labels = user_ids
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(genre_matrix,
                   xticklabels=top_genre_names,
                   yticklabels=row_labels,
                   cmap='Blues',
                   cbar_kws={"label": "Genre Preference Weight"},
                   ax=ax)
        
        ax.set_title(f'Top {top_n_genres} Genre Distribution' + 
                    (' by Cluster' if cluster_labels else ' by User'),
                    fontsize=16, pad=20)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved genre distribution heatmap to {save_path}")
        
        plt.close()
    
    def create_temporal_heatmap(self, temporal_data: Dict[str, Dict],
                              save_path: Optional[str] = None):
        """Create time-based listening pattern heatmap."""
        # Extract hour distributions
        n_users = len(temporal_data)
        hour_matrix = np.zeros((min(n_users, 50), 24))  # Limit to 50 users
        day_matrix = np.zeros((min(n_users, 50), 7))
        
        user_ids = list(temporal_data.keys())[:50]
        
        for i, user_id in enumerate(user_ids):
            patterns = temporal_data[user_id]
            if 'hour_distribution' in patterns:
                hour_matrix[i] = patterns['hour_distribution']
            if 'day_distribution' in patterns:
                day_matrix[i] = patterns['day_distribution']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Hour distribution heatmap
        sns.heatmap(hour_matrix,
                   xticklabels=range(24),
                   yticklabels=user_ids if len(user_ids) <= 20 else False,
                   cmap='YlOrRd',
                   cbar_kws={"label": "Listening Frequency"},
                   ax=ax1)
        
        ax1.set_title('Listening Activity by Hour of Day', fontsize=14)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('User')
        
        # Day distribution heatmap
        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        sns.heatmap(day_matrix,
                   xticklabels=day_labels,
                   yticklabels=user_ids if len(user_ids) <= 20 else False,
                   cmap='YlGnBu',
                   cbar_kws={"label": "Listening Frequency"},
                   ax=ax2)
        
        ax2.set_title('Listening Activity by Day of Week', fontsize=14)
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('User')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved temporal heatmap to {save_path}")
        
        plt.close()
    
    def create_interactive_heatmap(self, data: np.ndarray,
                                 x_labels: List[str],
                                 y_labels: List[str],
                                 title: str,
                                 save_path: Optional[str] = None,
                                 colorscale: str = 'Viridis'):
        """Create interactive heatmap using Plotly."""
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            hoverongaps=False,
            hovertemplate='%{x}<br>%{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(tickangle=-45),
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive heatmap to {save_path}")
        
        return fig
    
    def create_cluster_comparison_heatmap(self, cluster_stats: Dict,
                                        metrics: List[str],
                                        save_path: Optional[str] = None):
        """Create heatmap comparing different metrics across clusters."""
        clusters = sorted(cluster_stats.keys())
        n_clusters = len(clusters)
        n_metrics = len(metrics)
        
        # Create comparison matrix
        comparison_matrix = np.zeros((n_clusters, n_metrics))
        
        for i, cluster in enumerate(clusters):
            for j, metric in enumerate(metrics):
                if metric in cluster_stats[cluster]:
                    comparison_matrix[i, j] = cluster_stats[cluster][metric]
        
        # Normalize each metric
        for j in range(n_metrics):
            col = comparison_matrix[:, j]
            if np.std(col) > 0:
                comparison_matrix[:, j] = (col - np.mean(col)) / np.std(col)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(comparison_matrix,
                   xticklabels=metrics,
                   yticklabels=[f'Cluster {i}' for i in range(n_clusters)],
                   cmap=self.diverging_cmap,
                   center=0,
                   annot=True,
                   fmt='.2f',
                   cbar_kws={"label": "Normalized Score"},
                   ax=ax)
        
        ax.set_title('Cluster Comparison Heatmap (Z-scores)', fontsize=16, pad=20)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster comparison heatmap to {save_path}")
        
        plt.close()
    
    def create_co_occurrence_heatmap(self, items_per_user: Dict[str, List[str]],
                                   item_type: str = "Genre",
                                   save_path: Optional[str] = None,
                                   top_n: int = 20):
        """Create co-occurrence heatmap (e.g., which genres appear together)."""
        # Count co-occurrences
        from collections import Counter
        from itertools import combinations
        
        co_occurrence = Counter()
        item_counts = Counter()
        
        for user_items in items_per_user.values():
            # Count individual items
            for item in user_items:
                item_counts[item] += 1
            
            # Count pairs
            for item1, item2 in combinations(sorted(set(user_items)), 2):
                co_occurrence[(item1, item2)] += 1
                co_occurrence[(item2, item1)] += 1  # Make symmetric
        
        # Get top items
        top_items = [item for item, _ in item_counts.most_common(top_n)]
        
        # Create co-occurrence matrix
        n = len(top_items)
        matrix = np.zeros((n, n))
        
        for i, item1 in enumerate(top_items):
            matrix[i, i] = item_counts[item1]  # Diagonal: total count
            for j, item2 in enumerate(top_items):
                if i != j:
                    matrix[i, j] = co_occurrence.get((item1, item2), 0)
        
        # Normalize by diagonal (convert to correlation-like measure)
        normalized_matrix = np.zeros_like(matrix)
        for i in range(n):
            for j in range(n):
                if matrix[i, i] > 0 and matrix[j, j] > 0:
                    normalized_matrix[i, j] = matrix[i, j] / np.sqrt(matrix[i, i] * matrix[j, j])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(normalized_matrix, dtype=bool), k=1)
        sns.heatmap(normalized_matrix,
                   mask=mask,
                   xticklabels=top_items,
                   yticklabels=top_items,
                   cmap='RdYlBu_r',
                   center=0,
                   annot=True,
                   fmt='.2f',
                   square=True,
                   cbar_kws={"label": "Co-occurrence Strength"},
                   ax=ax,
                   vmin=0,
                   vmax=1)
        
        ax.set_title(f'{item_type} Co-occurrence Heatmap', fontsize=16, pad=20)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved co-occurrence heatmap to {save_path}")
        
        plt.close()
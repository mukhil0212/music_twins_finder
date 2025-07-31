"""
Music Twins Finder - ML Pipeline Flowchart Generator
Creates a comprehensive flowchart showing the end-to-end ML pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_ml_pipeline_flowchart():
    """Create a comprehensive ML pipeline flowchart for presentation."""
    
    # Create figure with high DPI for presentation quality
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'input': '#E3F2FD',      # Light blue
        'processing': '#FFF3E0',  # Light orange  
        'ml': '#E8F5E8',         # Light green
        'output': '#F3E5F5',     # Light purple
        'decision': '#FFEBEE'     # Light red
    }
    
    # Helper function to create boxes
    def create_box(x, y, width, height, text, color, fontsize=10):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=fontsize, 
                weight='bold', wrap=True)
    
    # Helper function to create arrows
    def create_arrow(x1, y1, x2, y2, style='->'):
        arrow = ConnectionPatch(
            (x1, y1), (x2, y2), "data", "data",
            arrowstyle=style, shrinkA=5, shrinkB=5,
            mutation_scale=20, fc="black", lw=2
        )
        ax.add_patch(arrow)
    
    # Title
    ax.text(5, 11.5, 'Music Twins Finder - ML Pipeline', 
            ha='center', va='center', fontsize=20, weight='bold')
    
    # 1. Data Input Layer
    create_box(0.5, 10, 2, 0.8, 'User Data\n(JSON Files)', colors['input'], 12)
    create_box(3, 10, 2, 0.8, 'Spotify API\n(Real-time)', colors['input'], 12)
    create_box(5.5, 10, 2, 0.8, 'Audio Features\n(Spotify)', colors['input'], 12)
    
    # 2. Data Processing Layer
    create_box(1, 8.5, 3, 0.8, 'Data Validation &\nEnhancement', colors['processing'], 11)
    create_box(5, 8.5, 3, 0.8, 'Feature Engineering\nPipeline', colors['processing'], 11)
    
    # 3. Feature Extraction Layer
    create_box(0.5, 7, 1.8, 0.8, 'Audio Features\n(101 dimensions)', colors['ml'], 10)
    create_box(2.5, 7, 1.8, 0.8, 'Genre Distribution\n(Jensen-Shannon)', colors['ml'], 10)
    create_box(4.5, 7, 1.8, 0.8, 'Artist Overlap\n(Jaccard Index)', colors['ml'], 10)
    create_box(6.5, 7, 1.8, 0.8, 'Temporal Patterns\n(Listening Habits)', colors['ml'], 10)
    
    # 4. ML Processing Layer
    create_box(1, 5.5, 2.5, 0.8, 'L2 Normalization\n(Per User)', colors['ml'], 11)
    create_box(4, 5.5, 2.5, 0.8, 'Similarity Calculation\n(Cosine + Overlaps)', colors['ml'], 11)
    create_box(7, 5.5, 1.5, 0.8, 'Clustering\n(K-Means)', colors['ml'], 11)
    
    # 5. Decision Layer
    create_box(3, 4, 3, 0.8, 'Enhanced Compatibility\nScore Calculation', colors['decision'], 11)
    
    # 6. Twin Detection
    create_box(1.5, 2.5, 2, 0.8, 'Threshold Check\n(≥80% = Twins)', colors['decision'], 11)
    create_box(5.5, 2.5, 2, 0.8, 'Twin Level\nClassification', colors['decision'], 11)
    
    # 7. Output Layer
    create_box(0.5, 1, 1.8, 0.8, 'Compatibility\nScore', colors['output'], 10)
    create_box(2.5, 1, 1.8, 0.8, 'Twin Status\n(Yes/No)', colors['output'], 10)
    create_box(4.5, 1, 1.8, 0.8, 'Detailed\nAnalysis', colors['output'], 10)
    create_box(6.5, 1, 1.8, 0.8, 'Visualizations\n(Charts)', colors['output'], 10)
    
    # Create arrows for flow
    # Input to Processing
    create_arrow(1.5, 10, 2.5, 9.3)
    create_arrow(4, 10, 3.5, 9.3)
    create_arrow(6.5, 10, 6.5, 9.3)
    
    # Processing to Features
    create_arrow(2.5, 8.5, 1.4, 7.8)
    create_arrow(2.5, 8.5, 3.4, 7.8)
    create_arrow(6.5, 8.5, 5.4, 7.8)
    create_arrow(6.5, 8.5, 7.4, 7.8)
    
    # Features to ML
    create_arrow(1.4, 7, 2.2, 6.3)
    create_arrow(3.4, 7, 2.8, 6.3)
    create_arrow(5.4, 7, 5.2, 6.3)
    create_arrow(7.4, 7, 7.7, 6.3)
    
    # ML to Decision
    create_arrow(2.2, 5.5, 4, 4.8)
    create_arrow(5.2, 5.5, 5, 4.8)
    
    # Decision to Twin Detection
    create_arrow(4.5, 4, 2.5, 3.3)
    create_arrow(4.5, 4, 6.5, 3.3)
    
    # Twin Detection to Output
    create_arrow(2.5, 2.5, 1.4, 1.8)
    create_arrow(2.5, 2.5, 3.4, 1.8)
    create_arrow(6.5, 2.5, 5.4, 1.8)
    create_arrow(6.5, 2.5, 7.4, 1.8)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Data Input'),
        mpatches.Patch(color=colors['processing'], label='Data Processing'),
        mpatches.Patch(color=colors['ml'], label='ML Processing'),
        mpatches.Patch(color=colors['decision'], label='Decision Logic'),
        mpatches.Patch(color=colors['output'], label='Output Results')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    # Add technical details box
    tech_details = """Key Technical Features:
• 101-dimensional audio feature space
• Jensen-Shannon divergence for genres
• Jaccard similarity for artist overlap
• L2 normalization (no per-pair scaling)
• Enhanced compatibility: 45% audio + 35% artists + 20% genres
• 80% threshold for twin classification"""
    
    ax.text(0.2, 0.2, tech_details, fontsize=9, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    return fig

def create_algorithm_detail_flowchart():
    """Create a detailed algorithm flowchart focusing on the similarity calculation."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Color scheme for algorithm steps
    colors = {
        'start': '#4CAF50',      # Green
        'process': '#2196F3',    # Blue
        'calculation': '#FF9800', # Orange
        'decision': '#F44336',   # Red
        'end': '#9C27B0'         # Purple
    }
    
    def create_diamond(x, y, width, height, text, color):
        """Create diamond shape for decisions"""
        diamond = mpatches.RegularPolygon(
            (x + width/2, y + height/2), 4, 
            radius=width/2, orientation=np.pi/4,
            facecolor=color, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(diamond)
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=9, weight='bold')
    
    def create_rect(x, y, width, height, text, color):
        """Create rectangle for processes"""
        rect = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='black', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, 
                ha='center', va='center', fontsize=9, weight='bold', wrap=True)
    
    # Title
    ax.text(5, 9.5, 'Music Twins Algorithm - Detailed Flow', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Algorithm steps
    create_rect(4, 8.5, 2, 0.6, 'Load User Data\n(2 Users)', colors['start'])
    create_rect(3.5, 7.5, 3, 0.6, 'Extract Audio Features\n(101 dimensions)', colors['process'])
    create_rect(3.5, 6.5, 3, 0.6, 'Calculate Artist Overlap\n(Jaccard Index)', colors['process'])
    create_rect(3.5, 5.5, 3, 0.6, 'Calculate Genre Similarity\n(Jensen-Shannon)', colors['process'])
    create_rect(3.5, 4.5, 3, 0.6, 'Apply L2 Normalization\n(Per User)', colors['calculation'])
    create_rect(3.5, 3.5, 3, 0.6, 'Compute Cosine Similarity\n(Audio Features)', colors['calculation'])
    create_rect(2, 2.5, 6, 0.6, 'Enhanced Score = 0.45×Audio + 0.35×Artists + 0.20×Genres', colors['calculation'])
    create_diamond(4, 1.3, 2, 0.8, 'Score ≥ 80%?', colors['decision'])
    create_rect(1, 0.2, 2, 0.6, 'Music Twins!', colors['end'])
    create_rect(7, 0.2, 2, 0.6, 'Not Twins', colors['end'])
    
    # Create arrows
    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    # Flow arrows
    arrow(5, 8.5, 5, 8.1)
    arrow(5, 7.5, 5, 7.1)
    arrow(5, 6.5, 5, 6.1)
    arrow(5, 5.5, 5, 5.1)
    arrow(5, 4.5, 5, 4.1)
    arrow(5, 3.5, 5, 3.1)
    arrow(5, 2.5, 5, 2.1)
    arrow(4.5, 1.3, 2, 0.8)  # Yes branch
    arrow(5.5, 1.3, 8, 0.8)  # No branch
    
    # Add Yes/No labels
    ax.text(3, 1, 'Yes', fontsize=10, weight='bold', color='green')
    ax.text(7, 1, 'No', fontsize=10, weight='bold', color='red')
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Create both flowcharts
    print("🎨 Creating ML Pipeline Flowcharts...")
    
    # Main pipeline flowchart
    fig1 = create_ml_pipeline_flowchart()
    fig1.savefig('ml_pipeline_flowchart.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Saved: ml_pipeline_flowchart.png")
    
    # Algorithm detail flowchart  
    fig2 = create_algorithm_detail_flowchart()
    fig2.savefig('algorithm_detail_flowchart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Saved: algorithm_detail_flowchart.png")
    
    print("\n🎯 Flowcharts ready for presentation!")
    print("📁 Files created:")
    print("  • ml_pipeline_flowchart.png - Complete end-to-end pipeline")
    print("  • algorithm_detail_flowchart.png - Detailed algorithm flow")

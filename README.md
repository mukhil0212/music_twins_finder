# Music Taste Twins Finder

A machine learning project that analyzes Spotify listening data to find users with similar music tastes and determine "music twins" through advanced audio feature analysis and compatibility scoring.

## Features

- **Enhanced Data Collection**: Comprehensive Spotify data collection with synthesized audio features for tracks
- **Music-Specific Analysis**: Audio features (danceability, energy, valence, acousticness) + genre distribution + listening patterns
- **Twin Compatibility Algorithm**: Multi-factor scoring system with weighted similarity metrics
- **Real-time Web Interface**: Flask app with interactive twin comparison and visualization
- **Advanced Visualizations**: 
  - Correlation heatmaps for audio features
  - Radar charts for music taste profiles
  - Compatibility breakdown charts
  - Audio feature comparison plots

## Setup

1. Create virtual environment & install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set up Spotify API credentials:
   - Create Spotify App at https://developer.spotify.com/dashboard/
   - Add to `.env`: `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, `SPOTIFY_REDIRECT_URI`

3. Run the web application:
```bash
python app/main.py
```

## Current Architecture

```
music-taste-twins/
├── app/                           # Flask web application
│   ├── main.py                   # Main Flask app with twin comparison
│   ├── templates/index.html      # Web UI with visualization display
│   └── static/visualizations/    # Generated plots & charts
├── src/                          # Core ML & data processing
│   ├── data_collection/          # Spotify API integration
│   │   ├── spotify_auth.py       # OAuth2 authentication
│   │   └── spotify_collector.py  # Data collection logic
│   ├── similarity/               # Twin matching algorithms
│   └── visualization/            # Chart generation
├── data/                         # Data storage
│   ├── raw/                      # User datasets (JSON)
│   ├── analysis/                 # Analysis results
│   └── cache/                    # Spotify auth cache
├── config/                       # Configuration
├── notebooks/                    # Jupyter analysis
└── scripts/                      # Data collection scripts
    ├── collect_*_data.py         # Individual user collectors
    ├── clear_all_auth.py         # Authentication cleaner
    └── analyze_three_twins.py    # Multi-user comparison
```

## Usage

### 1. Web Interface (Recommended)
```bash
python app/main.py
# Visit http://localhost:5000
# Enter two usernames to compare compatibility
```

### 2. Data Collection Scripts
```bash
# Clear auth cache for user switching
python clear_all_auth.py

# Collect individual user data
python collect_mukhil_data.py
python collect_aasha_data.py
python collect_sakshi_data.py
```

### 3. Multi-User Analysis
```bash
# Compare three users simultaneously
python analyze_three_twins.py
```

### 4. Programmatic Usage
```python
from src.data_collection.spotify_collector import SpotifyCollector

collector = SpotifyCollector()
user_data = collector.collect_user_data("username")
```

## Twin Analysis Algorithm

The compatibility scoring system uses weighted factors:
- **Audio Features** (35%): danceability, energy, valence, acousticness
- **Genre Similarity** (25%): genre distribution overlap
- **Artist Overlap** (25%): shared artists between users
- **Listening Patterns** (15%): hourly/daily listening habits

**Twin Threshold**: 0.65 compatibility score

## Key Files

- `app/main.py`: Flask web app with `/compare-twins` endpoint
- `src/data_collection/spotify_collector.py`: Enhanced data collection with audio feature synthesis
- `analyze_three_twins.py`: Multi-user twin comparison analysis
- `data/raw/`: User datasets (31pim5etbfm5vqhgn3btlbdlts64, aashamusic, sakshic5)

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
# Running Route Optimizer

An intelligent running route optimization tool for Christiansburg, Virginia, using OpenStreetMap data and high-resolution elevation data. Generate optimized running routes based on distance, elevation gain, or balanced objectives.

## Features

- **Multiple Route Objectives**
  - Shortest distance routes
  - Maximum elevation gain (hill training)
  - Balanced routes (mix of distance and elevation)
  - Easiest routes (avoid steep grades)

- **High-Resolution Elevation Data**
  - 3DEP 1-meter precision elevation data
  - Automatic fallback to SRTM 90m data
  - Accurate grade calculations and elevation profiles

- **Advanced Optimization**
  - Genetic algorithm-based route optimization
  - Segment-based encoding for superior route quality
  - Population-based search with adaptive strategies
  - 8 configurable initialization strategies

- **Two Interface Options**
  - Command-line interface (CLI) for power users
  - Interactive web interface (Streamlit) for visual exploration

- **Detailed Analysis**
  - Turn-by-turn directions
  - Elevation profiles with peaks/valleys
  - Route difficulty ratings
  - Grade distribution analysis

## Quick Start

### Prerequisites

- Python 3.12.3+
- Virtual environment with dependencies installed

### Installation

```bash
# Clone repository
git clone <repository-url>
cd run_routes

# Activate virtual environment
source venv/bin/activate

# Generate graph caches (one-time setup, ~2-5 minutes)
python setup_cache.py
```

### Usage

#### Web Interface (Recommended for Beginners)

```bash
streamlit run running_route_app.py
```

Open your browser to the URL shown (typically http://localhost:8501).

**Features:**
- Interactive map for selecting start location
- Real-time route visualization
- Elevation profile charts
- Exportable route data

#### Command-Line Interface

**Interactive Mode:**
```bash
python cli_route_planner.py --interactive
```

**Direct Route Generation:**
```bash
# Generate 5km route optimizing for elevation gain
python cli_route_planner.py --distance 5.0 --objective elevation

# Generate 8km balanced route
python cli_route_planner.py --distance 8.0 --objective balanced

# Generate shortest 3km route
python cli_route_planner.py --distance 3.0 --objective distance
```

**Options:**
- `--distance` / `-d`: Target distance in kilometers (e.g., 5.0)
- `--objective` / `-o`: Route objective (distance, elevation, balanced, difficulty)
- `--algorithm` / `-a`: Algorithm to use (genetic = genetic algorithm)
- `--interactive` / `-i`: Start interactive mode

## Project Architecture

The project follows a clean layered architecture:

```
utils/              # Bottom layer - shared utilities
  ├── elevation.py  # Unified elevation data access
  ├── geometry.py   # Geographic calculations
  ├── graph_utils.py # Graph operations
  └── cache.py      # Graph caching

genetic_algorithm/  # Route optimization algorithms
  ├── optimizer.py  # Main GA implementation
  ├── population.py # Population initialization
  ├── operators.py  # Genetic operators
  ├── fitness.py    # Fitness evaluation
  └── chromosome.py # Route representation

route_services/     # High-level services
  ├── network_manager.py    # Graph management
  ├── route_optimizer.py    # Route optimization
  ├── route_analyzer.py     # Route analysis
  ├── elevation_service.py  # Elevation profiling
  └── route_formatter.py    # Output formatting

applications/       # User interfaces
  ├── cli_route_planner.py  # CLI interface
  └── running_route_app.py  # Web interface
```

## Configuration

### Elevation Data

The system automatically uses the best available elevation data:
1. **3DEP 1m** (if available in `elevation_data/3dep_1m/`)
2. **SRTM 90m** (fallback, in `elevation_data/srtm_90m/`)

No manual configuration needed - the system handles fallback automatically.

### Cache Management

**Generate new cache:**
```bash
python generate_cached_graph.py --radius 800
```

**List available caches:**
```bash
python -c "from utils.cache import list_cached_graphs; list_cached_graphs()"
```

**Clean old caches:**
```bash
python -c "from utils.cache import clean_cache; clean_cache()"
```

## Development

### Running Tests

```bash
# Run all tests
python tests/run_tests.py all

# Run specific test categories
python tests/run_tests.py unit         # Unit tests (fast, mocked)
python tests/run_tests.py integration  # Integration tests
python tests/run_tests.py smoke        # Smoke tests (real dependencies)
python tests/run_tests.py ga           # Genetic algorithm tests
```

### Code Coverage

```bash
# Generate coverage report
python tests/run_tests.py coverage

# View HTML report
open htmlcov/index.html
```

### Using as a Library

```python
from route_services import NetworkManager, RouteOptimizer
from utils.elevation import get_elevation_service
from utils.geometry import haversine_distance

# Initialize services
nm = NetworkManager()
graph = nm.load_network()

# Optimize route
optimizer = RouteOptimizer(graph)
result = optimizer.optimize_route(
    start_node=1529188403,
    target_distance_km=5.0,
    objective='maximize_elevation',
    algorithm='genetic'
)

# Access results
print(f"Distance: {result['stats']['total_distance_km']:.2f} km")
print(f"Elevation gain: {result['stats']['total_elevation_gain_m']:.0f} m")
print(f"Route: {result['route']}")
```

## Recent Changes

**March 2026 - Comprehensive Refactoring:**
- 45% code reduction (10,000 → 5,500 lines)
- 38% fewer files (40+ → 25 files)
- Unified elevation system with automatic fallback
- Consolidated population initializers (5 → 1)
- Clear layered architecture with no circular imports
- Maintained >90% test coverage throughout

See [CLAUDE.md](CLAUDE.md) for detailed technical documentation.

## Data Sources

- **Street Network**: OpenStreetMap via OSMnx
- **Elevation Data**:
  - USGS 3DEP 1m (primary, Virginia coverage)
  - SRTM 90m (fallback, global coverage)

## License

[Add license information]

## Contributing

[Add contribution guidelines]

## Support

For issues or questions:
- Check [CLAUDE.md](CLAUDE.md) for development guidance
- Review test suite for usage examples
- Open an issue in the repository

## Acknowledgments

- OpenStreetMap contributors for street network data
- USGS 3DEP for high-resolution elevation data
- NASA SRTM for global elevation coverage

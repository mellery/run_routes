# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **refactored Python geospatial analysis project** focused on running route optimization using OpenStreetMap data and elevation information. The project uses a **shared services architecture** to eliminate code duplication between CLI and web applications, providing optimized running routes for Christiansburg, Virginia.

**Key Architecture Change:** The project has been refactored from duplicated code to shared services, with zero code duplication between applications.

## Environment Setup

This project uses Python 3.12.3 with a virtual environment located in `venv/`. To activate the environment and run the code:

```bash
source venv/bin/activate
python route.py
```

## Key Dependencies

The project relies on several geospatial and data visualization libraries:
- `osmnx` (2.0.1) - OpenStreetMap network analysis
- `networkx` (3.4.2) - Graph analysis and manipulation
- `matplotlib` (3.10.0) - Plotting and visualization
- `rasterio` (1.4.3) - Raster data handling (for SRTM elevation data)
- `geopandas` (1.0.1) - Geospatial data manipulation
- `numpy` (2.2.1) - Numerical computations

## Code Architecture

**Refactored Shared Services Architecture:**

### Shared Route Services (`route_services/`)
- **NetworkManager** - Graph loading, caching, node operations
- **RouteOptimizer** - TSP solving with automatic fast/standard solver fallback  
- **RouteAnalyzer** - Route analysis, statistics, turn-by-turn directions
- **ElevationProfiler** - Elevation profile generation and analysis
- **RouteFormatter** - Platform-agnostic output formatting

### Applications (Using Shared Services)
- **CLI Route Planner** (`cli_route_planner.py`) - Command-line interface
- **Streamlit Web App** (`running_route_app.py`) - Interactive web interface

### Core Utilities
- `route.py` - Core geospatial utility functions
- `tsp_solver_fast.py` - Optimized TSP solver without distance matrix
- `tsp_solver.py` - Standard TSP solver with distance matrix
- `graph_cache.py` - Network caching utilities

### Data Files
- `srtm_38_03.tif` - SRTM elevation data for the region
- `srtm_data/` - Contains additional SRTM data files and documentation
- `cache/` - Contains cached JSON data files (likely OSMnx cache)

The script currently focuses on analyzing a specific node (ID: 216507089) and visualizing the street network with node and edge information.

## First-Time Setup

### Cache Generation (Recommended)
For optimal performance, generate graph caches before first use:

```bash
# Generate common caches (one-time setup, ~2-5 minutes)
python setup_cache.py

# Or generate specific cache sizes
python generate_cached_graph.py --radius 800 --network-type all
python generate_cached_graph.py --radius 400 --network-type drive
```

This pre-processes street networks with elevation data, reducing application startup from 100+ seconds to <1 second.

## Development Commands

Since this is a refactored Python project with comprehensive testing and shared services:

```bash
# Activate virtual environment
source venv/bin/activate

# === MAIN APPLICATIONS ===
# Web Application (Streamlit) - Refactored with shared services
streamlit run running_route_app.py

# Command Line Interface - Refactored with shared services  
python cli_route_planner.py --interactive

# Generate optimized route directly (uses defaults: 5km, node 1529188403)
python cli_route_planner.py --start-node 1529188403 --distance 5.0 --objective elevation

# === TESTING (NEW) ===
# Run comprehensive test suite
python tests/run_tests.py                 # All tests
python tests/run_tests.py unit           # Unit tests only
python tests/run_tests.py integration    # Integration tests only

# === USING SHARED SERVICES DIRECTLY ===
# Example: Use services in Python
python -c "
from route_services import NetworkManager, RouteOptimizer, RouteFormatter
nm = NetworkManager()
graph = nm.load_network()
optimizer = RouteOptimizer(graph)
result = optimizer.optimize_route(1529188403, 5.0)
formatter = RouteFormatter()
print(formatter.format_route_summary(result))
"

# === UTILITIES AND ANALYSIS ===
# Generate visualizations
python plot_3d_streets.py --dist 800 --exaggeration 10

# Cache management
python setup_cache.py                    # Generate common caches
python graph_cache.py list              # List available caches  
python graph_cache.py clean             # Clean old caches

# Core analysis
python route.py                          # Basic network analysis

# === LEGACY/BACKUP ===
# Original applications (backed up)
python cli_route_planner_original.py    # Original CLI
streamlit run running_route_app_original.py  # Original Streamlit

# Check dependencies
source venv/bin/activate && pip list

# Install new packages (if needed)
source venv/bin/activate && pip install <package_name>
```

**Key Testing Features:**
- 80+ unit tests covering all shared services
- Integration tests for end-to-end workflows
- Comprehensive test runner with reporting
- Mocked dependencies for isolated testing

**Default Settings:**
- **Starting node:** 1529188403 (Christiansburg, VA)
- **Distance:** 5.0km
- **Solver:** Automatic fast/standard TSP fallback
- **Network area:** 0.8km radius around (37.1299, -80.4094)
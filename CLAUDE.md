# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Python geospatial analysis project** focused on running route optimization using OpenStreetMap data and elevation information. The project uses a **shared services architecture** to eliminate code duplication between CLI and web applications, providing optimized running routes for Christiansburg, Virginia.

**Key Architecture:**
- ✅ **Refactored to shared services**: Zero code duplication between applications
- ✅ **3DEP 1m Elevation Integration**: Complete integration with 1-meter precision elevation data
- ✅ **Genetic Algorithm Implementation**: Complete GA-based route optimization alongside existing TSP solvers
- ✅ **Enhanced Elevation Optimization**: Segment-based encoding for superior elevation gain routes with 1m precision
- ✅ **Development Visualizations**: OpenStreetMap-based route visualization during GA development
- ✅ **Comprehensive Refactoring (2026-03)**: 45% code reduction, clear module boundaries, simplified architecture

## Recent Refactoring (March 2026)

**Completed: 5-Phase Comprehensive Refactoring**

The codebase underwent a major refactoring to eliminate duplication, clarify module boundaries, and establish a clear layered architecture. Results:

**Code Metrics:**
- **Total lines reduced**: ~10,000 → ~5,500 lines (45% reduction)
- **Python files reduced**: 40+ → 25 files (38% reduction)
- **Root-level files**: 23 → 6 files (74% reduction)
- **Test coverage maintained**: >90% throughout refactoring
- **Test pass rate maintained**: 100% for integration/smoke tests

**Key Changes:**

**Phase 1: Elevation System Consolidation** ✅
- Consolidated 5 elevation files (3,600 lines) into 2 unified modules (1,424 lines)
- Created `utils/elevation.py` - unified elevation backend with automatic 3DEP → SRTM fallback
- Created `route_services/elevation_service.py` - unified elevation profiler
- Single source of truth for elevation data access

**Phase 2: Population Initializer Consolidation** ✅
- Consolidated 5 population initializer files (2,649 lines) into 1 file (583 lines, 78% reduction)
- Created unified `genetic_algorithm/population.py` with 8 configurable strategies
- Eliminated 80% code overlap across initializers

**Phase 3: Genetic Algorithm Cleanup** ✅
- Simplified `genetic_algorithm/__init__.py` exports from 24+ to 8 core classes
- Clarified module boundaries: Core vs Supporting vs Optional
- Removed circular import risks

**Phase 4: Root-Level Cleanup** ✅
- Created `utils/` directory for shared utilities (elevation, geometry, graph_utils, cache)
- Moved 7 utility scripts to `scripts/` directory
- Reduced root-level Python files from 23 to 6
- Updated both applications (CLI + web) to use refactored imports

**Phase 5: Documentation & Validation** ✅
- Updated all documentation to reflect new architecture
- Validated all applications work correctly
- Maintained backward compatibility during transition

**Architecture Benefits:**
- ✅ Clear dependency hierarchy: utils → genetic_algorithm → route_services → apps
- ✅ Zero circular imports
- ✅ Single source of truth for each concern
- ✅ Simplified imports (1 import instead of 5+ for common tasks)
- ✅ Faster application startup (reduced import overhead)
- ✅ Easier onboarding and maintenance

## Environment Setup

This project uses Python 3.12.3 with a virtual environment located in `venv/`. To activate the environment and run applications:

```bash
source venv/bin/activate

# Run CLI application
python cli_route_planner.py --interactive

# Run web application
streamlit run running_route_app.py
```

## Key Dependencies

The project relies on several geospatial and data visualization libraries (available in `venv/`):
- `osmnx` - OpenStreetMap network analysis
- `networkx` (3.4.2) - Graph analysis and manipulation
- `numpy` (2.2.1) - Numerical computations
- `matplotlib` - Plotting and visualization  
- `streamlit` - Web application framework
- `folium` - Interactive maps
- `plotly` - Interactive visualizations
- `pandas` - Data manipulation

## Code Architecture

**✅ Refactored Layered Architecture (45% code reduction achieved):**

```
┌─────────────────────────────────────────────┐
│        Applications (Root Level)            │
│  - cli_route_planner.py                     │
│  - running_route_app.py (Streamlit)         │
│  - setup_cache.py (utility script)          │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         route_services/                     │
│  (Public API - Clean Service Interface)     │
│  - NetworkManager                           │
│  - RouteOptimizer                           │
│  - RouteAnalyzer                            │
│  - ElevationService (unified profiler)      │
│  - RouteFormatter                           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         genetic_algorithm/                  │
│  (GA Implementation - 5 core files)         │
│  - optimizer.py (GA loop + adaptive logic)  │
│  - population.py (unified initializer)      │
│  - operators.py (all genetic operators)     │
│  - fitness.py (fitness evaluation)          │
│  - chromosome.py (route representation)     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              utils/                         │
│  (Shared Utilities - Bottom Layer)          │
│  - elevation.py (unified elevation access)  │
│  - geometry.py (distance, bearing calcs)    │
│  - graph_utils.py (graph operations)        │
│  - cache.py (graph caching)                 │
└─────────────────────────────────────────────┘
```

### Applications (Root Level - 6 files)
- **CLI Route Planner** (`cli_route_planner.py`) - Command-line interface
- **Streamlit Web App** (`running_route_app.py`) - Interactive web interface
- **Cache Setup** (`setup_cache.py`) - Graph cache generation utility
- **Graph Generator** (`generate_cached_graph.py`) - Individual cache generation
- **OSMnx Config** (`osmnx_config.py`) - OSMnx configuration
- **Route Objective** (`route_objective.py`) - Objective function definitions

### Shared Route Services (`route_services/`)
- **NetworkManager** - Graph loading, caching, node operations
- **RouteOptimizer** - Genetic Algorithm-based route optimization with segment-based encoding
- **RouteAnalyzer** - Route analysis, statistics, turn-by-turn directions
- **ElevationService** - Unified elevation profile generation with 3DEP 1m precision
- **RouteFormatter** - Platform-agnostic output formatting

### Genetic Algorithm (`genetic_algorithm/`)
**Core Implementation (5 files):**
- **optimizer.py** - Complete GA loop with adaptive mutation and restart mechanisms
- **population.py** - Unified population initializer with 8 configurable strategies
- **operators.py** - All genetic operators (crossover, mutation, selection)
- **fitness.py** - Fitness evaluation for all objectives
- **chromosome.py** - Route chromosome and segment representations

**Supporting Modules:**
- **performance.py** - GASegmentCache for performance optimization
- **visualization.py** - Development visualization tools (optional)

### Shared Utilities (`utils/`)
**Bottom layer - no dependencies on other project modules:**
- **elevation.py** - Unified elevation data access
  - `ElevationService` - Main API with automatic 3DEP → SRTM fallback
  - `LocalThreeDEPSource` - 3DEP 1m tile-based access with LRU caching
  - `SRTMElevationSource` - SRTM 90m data access
  - `HybridElevationSource` - Automatic fallback orchestration
- **geometry.py** - Geographic calculations
  - `haversine_distance()` - Great circle distance calculations
  - `calculate_bearing()` - Bearing between coordinates
  - `destination_point()` - Destination from bearing and distance
- **graph_utils.py** - Graph operations
  - `has_elevation_data()` - Validate elevation in graph
  - `add_elevation_to_edges()` - Calculate grade and elevation changes
  - `add_running_weights()` - Add running-specific edge weights
  - `get_nodes_within_distance()` - Find reachable nodes
- **cache.py** - Graph caching utilities
  - `load_cached_graph()` - Load graph from cache
  - `save_cached_graph()` - Save graph to cache
  - `list_cached_graphs()` - List available caches
  - `clean_cache()` - Remove old/invalid caches

### Data Files
- `elevation_data/srtm_90m/srtm_20_05.tif` - SRTM 90m elevation data for the region
- `elevation_data/3dep_1m/` - 3DEP 1-meter elevation tiles directory
- `elevation_data/3dep_1m/tiles/` - 3DEP tile storage
- `valid_3dep_coordinate.txt` - Validated 3DEP coverage coordinate (36.846651, -78.409308)
- `srtm_data/` - Contains additional SRTM data files and documentation
- `cache/` - Contains cached JSON data files (likely OSMnx cache)

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

## Genetic Algorithm Development

### **🚨 CRITICAL: Visualization Requirements**

During GA development, **visualizations are mandatory** at key verification points. All visualizations must:
- Use **OpenStreetMap background** for geographic context
- Show **detailed route overlays** with elevation coloring
- Include **population statistics** and **fitness metrics**
- Save as **PNG images** with descriptive filenames
- Be generated **automatically** during development phases

### **🧪 CRITICAL: Unit Testing Requirements**

**Unit tests are MANDATORY during GA development.** All GA components must have comprehensive unit test coverage:
- **Test-Driven Development**: Write tests before implementing features
- **Coverage requirement**: >90% test coverage for all GA components
- **Mock dependencies**: Use mocks for external dependencies (matplotlib, file I/O)
- **Fast execution**: Unit tests must complete in <1 second
- **Automatic execution**: Tests run via `python tests/run_tests.py ga`

### **Key Visualization Points:**
1. **Population Initialization**: Show initial population diversity
2. **Crossover Operations**: Verify parent combination correctness
3. **Mutation Effects**: Display mutation impact on routes
4. **Fitness Evolution**: Track population fitness over generations
5. **Final Results**: Compare GA vs TSP route quality

### **Visualization Standards:**
```python
# Example visualization call
visualizer = GAVisualizer(graph)
visualizer.save_population_map(
    population=current_population,
    generation=gen_num,
    filename=f"population_gen_{gen_num:03d}.png",
    show_fitness=True,
    show_elevation=True,
    osm_background=True
)
```

### **Development Verification Protocol:**
- Generate visualization **before and after** each major operation
- Save **fitness progression plots** every 25 generations
- Create **comparison images** between TSP and GA results
- Include **elevation profile overlays** for route analysis
- Document **unexpected behaviors** with annotated images

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

# Generate optimized route directly (genetic algorithm)
python cli_route_planner.py --start-node 1529188403 --distance 5.0 --objective elevation --algorithm genetic

# === TESTING ===
# Run comprehensive test suite (all tests use refactored architecture)
python tests/run_tests.py all            # All tests (unit + integration + smoke)
python tests/run_tests.py unit           # Unit tests (mocked, fast)
python tests/run_tests.py integration    # Integration tests (mocked dependencies)
python tests/run_tests.py smoke          # Smoke tests (real dependencies)

# Individual test files
python -m unittest tests.unit.test_network_manager -v
python -m unittest tests.unit.test_elevation_service -v  # NEW: Tests utils/elevation.py
python -m unittest tests.integration.test_route_services_integration -v

# === GA TESTING ===
# GA-specific unit testing (MANDATORY during development)
python tests/run_tests.py ga            # GA unit tests (235+ tests)
python tests/run_tests.py all           # All tests including GA (465+ tests)

# Individual GA test files
python -m unittest tests.unit.test_ga_chromosome -v         # Chromosome classes (32 tests)
python -m unittest tests.unit.test_ga_population -v         # Unified population initializer (45 tests)
python -m unittest tests.unit.test_genetic_optimizer -v     # Genetic optimizer (14 tests)
python -m unittest tests.unit.test_ga_operators -v          # All operators consolidated (60+ tests)
python -m unittest tests.unit.test_ga_fitness -v            # Fitness evaluation (16 tests)
python -m unittest tests.unit.test_ga_performance -v        # Performance optimization (60+ tests)

# === USING REFACTORED ARCHITECTURE DIRECTLY ===
# Example: Use services and utils in Python
python -c "
# Import route services
from route_services import NetworkManager, RouteOptimizer, RouteFormatter
from route_services.elevation_service import ElevationProfiler

# Import utils
from utils.elevation import get_elevation_service
from utils.geometry import haversine_distance
from utils.cache import load_cached_graph

# Use services
nm = NetworkManager()
graph = nm.load_network()
optimizer = RouteOptimizer(graph)
result = optimizer.optimize_route(1529188403, 5.0)
formatter = RouteFormatter()
print(formatter.format_route_summary(result))

# Use elevation service
elevation_svc = get_elevation_service()
elev = elevation_svc.get_elevation(37.1299, -80.4094)
print(f'Elevation at start: {elev}m')
"

# === GENETIC ALGORITHM DEVELOPMENT (✅ COMPLETED) ===
# Test GA implementation with visualization
python -c "
from route_services import NetworkManager, RouteOptimizer
from genetic_algorithm.visualization import GAVisualizer

nm = NetworkManager()
graph = nm.load_network()
optimizer = RouteOptimizer(graph)
viz = GAVisualizer(graph)

# Optimize with visualization
result = optimizer.optimize_route(1529188403, 5.0, 'maximize_elevation', algorithm='genetic')
print(f'Best route: {result[\"stats\"][\"total_elevation_gain_m\"]}m elevation gain')
"

# === UTILITIES AND ANALYSIS ===
# Generate visualizations (scripts in scripts/ directory)
python scripts/terrain_profile_plotter.py --dist 800 --exaggeration 10
python scripts/eval_k_shortest.py

# Cache management (uses utils.cache)
python setup_cache.py                    # Generate common caches
python -c "from utils.cache import list_cached_graphs; list_cached_graphs()"  # List available caches
python -c "from utils.cache import clean_cache; clean_cache()"  # Clean old caches

# Elevation cache generation with unified elevation service
python generate_cached_graph.py --radius 800   # Generate cache with automatic 3DEP → SRTM fallback
python generate_cached_graph.py --radius 1200 --force  # Force regenerate cache

# Check dependencies
source venv/bin/activate && pip list

# Install new packages (if needed)
source venv/bin/activate && pip install <package_name>

# === COVERAGE REPORTING ===
# Generate detailed test coverage reports with pytest-cov
python tests/run_tests.py coverage        # Run tests with coverage reporting
python generate_coverage_badge.py         # Generate coverage badges and summary

# Coverage artifacts generated:
# - htmlcov/index.html: Detailed HTML coverage report
# - coverage.xml: XML coverage data
# - coverage_badges.md: Badge markdown for documentation
# - coverage_data.json: JSON coverage data
```

**Default Settings:**
- **Starting node:** 1529188403 (Christiansburg, VA)
- **Distance:** 5.0km
- **Solver:** Genetic Algorithm-based optimization
  - GA: population-based search for superior route optimization
- **Network area:** 5.0km radius around (37.1299, -80.4094)

**GA Development Settings:**
- **Population size:** 100 (adaptive based on route distance)
- **Max generations:** 200 (adaptive based on route distance)
- **Visualization:** Required during development for verification
- **Output format:** PNG images with OpenStreetMap background
- **Key verification points:** Population initialization, crossover/mutation, fitness evolution

### **🚨 MANDATORY UNIT TESTING REQUIREMENTS**

**CRITICAL**: Unit tests MUST be created during development and MUST pass before proceeding to the next phase.

#### **Test Categories (Current: 235+ GA tests, 100% passing)**
```bash
# Run all GA tests
python tests/run_tests.py ga

# Specific test files
python -m unittest tests.unit.test_ga_chromosome -v       # 32 tests - RouteSegment & RouteChromosome
python -m unittest tests.unit.test_ga_population -v       # 45 tests - PopulationInitializer 
python -m unittest tests.unit.test_ga_visualizer -v       # 8 tests - GAVisualizer (mocked)
python -m unittest tests.unit.test_ga_operators -v        # 60+ tests - Crossover, mutation, selection operators
python -m unittest tests.unit.test_ga_fitness -v          # 16 tests - Fitness evaluation system
python -m unittest tests.unit.test_genetic_optimizer -v   # 14 tests - Complete genetic optimizer
python -m unittest tests.unit.test_ga_performance -v      # 60+ tests - Performance optimization components
```

#### **Test Requirements for Each Component:**
- **Chromosome classes**: ✅ Property calculation, connectivity validation, statistics (32 tests)
- **Population initialization**: ✅ All 4 strategies, diversity metrics, error handling (45 tests)
- **Genetic operators**: ✅ Crossover, mutation, selection with before/after validation (60+ tests)
- **Fitness evaluation**: ✅ All objectives, edge cases, plateau detection, statistics (16 tests)
- **Genetic optimizer**: ✅ Evolution loop, convergence, adaptive configuration, callbacks (14 tests)
- **Visualizer**: ✅ Mocked matplotlib calls, image generation, error handling (8 tests)
- **Performance optimization**: ✅ Caching, parallel evaluation, distance optimization, memory management (60+ tests)

#### **Quality Gates:**
- 🚨 **100% test pass rate** required before advancing phases
- 🚨 **Unit tests** must be created concurrently with code development
- 🚨 **Integration tests** required for multi-component interactions
- 🚨 **Error handling** must be tested for invalid inputs and edge cases


### **Image Naming Convention:**
```
ga_dev_PHASE_COMPONENT_TIMESTAMP.png

Examples:
ga_dev_init_population_gen000_20241204_143022.png
ga_dev_crossover_parents_vs_offspring_20241204_143045.png
ga_dev_evolution_fitness_progress_20241204_143112.png
ga_dev_comparison_tsp_vs_ga_elevation_20241204_143200.png
```

### **Quality Gates:**
Each phase requires **visual verification AND unit test coverage** before proceeding:
1. **Unit Test Coverage**: >90% test coverage for all GA components
2. **Test Execution**: All unit tests must pass (python tests/run_tests.py ga)
3. **Chromosome validity**: All routes must be connected and return to start
4. **Operator correctness**: Crossover/mutation must preserve route validity
5. **Evolution progress**: Fitness must improve over generations
6. **Objective optimization**: GA must exceed TSP for elevation objectives
7. **Performance acceptance**: GA runtime must be reasonable (<120s for 5km routes)
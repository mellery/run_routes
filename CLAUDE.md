# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **refactored Python geospatial analysis project** focused on running route optimization using OpenStreetMap data and elevation information. The project uses a **shared services architecture** to eliminate code duplication between CLI and web applications, providing optimized running routes for Christiansburg, Virginia.

**Key Architecture Changes:**
- ✅ **Refactored to shared services**: Zero code duplication between applications
- 🚧 **Genetic Algorithm Implementation**: New GA-based route optimization approach alongside existing TSP solvers
- 🎯 **Enhanced Elevation Optimization**: Segment-based encoding for superior elevation gain routes
- 📊 **Development Visualizations**: OpenStreetMap-based route visualization during GA development

## Environment Setup

This project uses Python 3.12.3 with a virtual environment located in `venv/`. To activate the environment and run the code:

```bash
source venv/bin/activate
python route.py
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

**Refactored Shared Services Architecture:**

### Shared Route Services (`route_services/`)
- **NetworkManager** - Graph loading, caching, node operations
- **RouteOptimizer** - Multi-algorithm optimization (TSP + Genetic Algorithm)
  - TSP solvers with automatic fast/standard fallback
  - **NEW**: Genetic Algorithm with segment-based encoding
  - Automatic algorithm selection based on objective
- **RouteAnalyzer** - Route analysis, statistics, turn-by-turn directions
- **ElevationProfiler** - Elevation profile generation and analysis
- **RouteFormatter** - Platform-agnostic output formatting
- **GAVisualizer** - 🚧 **IN DEVELOPMENT**: GA development visualizations

### Applications (Using Shared Services)
- **CLI Route Planner** (`cli_route_planner.py`) - Command-line interface
- **Streamlit Web App** (`running_route_app.py`) - Interactive web interface

### Core Utilities
- `route.py` - Core geospatial utility functions
- **TSP Solvers:**
  - `tsp_solver_fast.py` - Optimized TSP solver without distance matrix
  - `tsp_solver.py` - Standard TSP solver with distance matrix
- **Genetic Algorithm (🚧 IN DEVELOPMENT):**
  - `genetic_route_optimizer.py` - GA implementation with segment-based encoding
  - `ga_chromosome.py` - Route chromosome and segment representations
  - `ga_operators.py` - Crossover, mutation, and selection operators
  - `ga_visualizer.py` - Development visualization tools
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

## Genetic Algorithm Development

### **🚨 CRITICAL: Visualization Requirements**

During GA development, **visualizations are mandatory** at key verification points. All visualizations must:
- Use **OpenStreetMap background** for geographic context
- Show **detailed route overlays** with elevation coloring
- Include **population statistics** and **fitness metrics**
- Save as **PNG images** with descriptive filenames
- Be generated **automatically** during development phases

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

# Generate optimized route directly (uses defaults: 5km, node 1529188403)
python cli_route_planner.py --start-node 1529188403 --distance 5.0 --objective elevation

# === TESTING ===
# Run comprehensive test suite (90 tests, all passing)
python tests/run_tests.py all            # All tests (unit + integration + smoke)
python tests/run_tests.py unit           # Unit tests (76 tests, mocked, fast)
python tests/run_tests.py integration    # Integration tests (7 tests, mocked)
python tests/run_tests.py smoke          # Smoke tests (7 tests, real dependencies)

# Individual test files
python -m unittest tests.unit.test_network_manager -v
python -m unittest tests.integration.test_route_services_integration -v

# === GA TESTING (🚧 IN DEVELOPMENT) ===
# GA-specific testing with mandatory visualizations
python tests/run_tests.py ga            # GA unit tests
python tests/run_tests.py ga_integration # GA integration tests  
python tests/run_tests.py ga_visual      # GA visualization tests

# Individual GA test files
python -m unittest tests.unit.test_genetic_optimizer -v  # 🚧 IN DEVELOPMENT
python -m unittest tests.unit.test_ga_chromosome -v      # 🚧 IN DEVELOPMENT
python -m unittest tests.unit.test_ga_operators -v       # 🚧 IN DEVELOPMENT
python -m unittest tests.visual.test_ga_visualizer -v    # 🚧 IN DEVELOPMENT

# GA Development Testing with Visualizations
python tests/ga_development_test.py --phase initialization --save-images
python tests/ga_development_test.py --phase crossover --save-images
python tests/ga_development_test.py --phase mutation --save-images
python tests/ga_development_test.py --phase evolution --save-images
python tests/ga_development_test.py --phase comparison --save-images

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

# === GENETIC ALGORITHM DEVELOPMENT (🚧 IN DEVELOPMENT) ===
# Test GA implementation with visualization
python -c "
from route_services import NetworkManager
from genetic_route_optimizer import GeneticRouteOptimizer
from ga_visualizer import GAVisualizer

nm = NetworkManager()
graph = nm.load_network()
ga = GeneticRouteOptimizer(graph, population_size=50, max_generations=100)
viz = GAVisualizer(graph)

# Optimize with visualization
result = ga.optimize_route_with_viz(1529188403, 5.0, 'maximize_elevation', viz)
print(f'Best route: {result[\"stats\"][\"total_elevation_gain_m\"]}m elevation gain')
"

# Generate GA development visualizations
python ga_visualizer.py --test-population --save-images
python ga_visualizer.py --evolution-animation --generations 50
python ga_visualizer.py --compare-algorithms --distance 5.0

# === UTILITIES AND ANALYSIS ===
# Generate visualizations
python plot_3d_streets.py --dist 800 --exaggeration 10

# === GA DEVELOPMENT VISUALIZATIONS (🚧 IN DEVELOPMENT) ===
# IMPORTANT: Generate visualizations during GA development for verification
# All visualizations use OpenStreetMap background with detailed overlays

# Population visualization
python ga_visualizer.py --show-population --generation 0 --save population_gen0.png
python ga_visualizer.py --show-population --generation 50 --save population_gen50.png

# Evolution tracking
python ga_visualizer.py --evolution-gif --generations 100 --save evolution.gif
python ga_visualizer.py --fitness-history --save fitness_progress.png

# Chromosome analysis
python ga_visualizer.py --show-chromosome --route-id best --save best_route.png
python ga_visualizer.py --segment-analysis --route-id best --save segment_details.png

# Algorithm comparison
python ga_visualizer.py --compare-tsp-ga --objective elevation --save comparison.png
python ga_visualizer.py --elevation-heatmap --save elevation_analysis.png

# Operator testing
python ga_visualizer.py --test-crossover --parent1 id1 --parent2 id2 --save crossover_test.png
python ga_visualizer.py --test-mutation --chromosome id1 --save mutation_test.png

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

**Key Features Completed:**
- ✅ **Zero Code Duplication**: ~800 lines eliminated via shared services
- ✅ **Comprehensive Testing**: 90 tests (100% passing)
  - 76 unit tests (mocked, fast ~0.02s)
  - 7 integration tests (mocked workflows) 
  - 7 smoke tests (real dependencies ~1.2s)
- ✅ **Robust Test Suite**: Hybrid approach with mocked + real dependency testing
- ✅ **Production Ready**: Fully refactored applications using shared services

**Features In Development:**
- 🚧 **Genetic Algorithm Route Optimization**: Segment-based encoding for superior elevation routes
- 🚧 **GA Development Visualizations**: OpenStreetMap-based verification images
- 🚧 **Multi-Algorithm Selection**: Automatic TSP vs GA selection based on objective
- 🚧 **Enhanced Elevation Optimization**: Population-based search for creative route discovery

**Default Settings:**
- **Starting node:** 1529188403 (Christiansburg, VA)
- **Distance:** 5.0km
- **Solver:** Automatic algorithm selection (TSP/GA based on objective)
  - TSP: fast/standard fallback for distance objectives
  - GA: population-based search for elevation objectives
- **Network area:** 5.0km radius around (37.1299, -80.4094)

**GA Development Settings:**
- **Population size:** 100 (adaptive based on route distance)
- **Max generations:** 200 (adaptive based on route distance)
- **Visualization:** Required during development for verification
- **Output format:** PNG images with OpenStreetMap background
- **Key verification points:** Population initialization, crossover/mutation, fitness evolution

## GA Development Workflow

### **Phase 1: Foundation (Week 1)**
```bash
# Implement core classes
python -c "from ga_chromosome import RouteChromosome, RouteSegment; print('Classes implemented')"

# Test chromosome creation with visualization
python ga_development_test.py --phase chromosome --save-images
python ga_visualizer.py --test-chromosome --save chromosome_test.png

# Verify population initialization
python ga_development_test.py --phase initialization --save-images
```

### **Phase 2: Genetic Operators (Week 2)**
```bash
# Test crossover operators with before/after visualization
python ga_development_test.py --phase crossover --save-images
python ga_visualizer.py --test-crossover --save crossover_results.png

# Test mutation operators with impact visualization
python ga_development_test.py --phase mutation --save-images
python ga_visualizer.py --test-mutation --save mutation_results.png

# Verify selection strategies
python ga_development_test.py --phase selection --save-images
```

### **Phase 3: Evolution & Optimization (Week 3)**
```bash
# Full evolution test with generation-by-generation visualization
python ga_development_test.py --phase evolution --generations 50 --save-images
python ga_visualizer.py --evolution-gif --save evolution_50gen.gif

# Performance comparison with TSP
python ga_development_test.py --phase comparison --save-images
python ga_visualizer.py --compare-algorithms --save tsp_vs_ga.png
```

### **🚨 MANDATORY VISUALIZATION CHECKLIST**

Before each development milestone, generate these verification images:

**✅ Population Diversity Check:**
- [ ] Initial population routes overlaid on OpenStreetMap
- [ ] Population fitness distribution histogram
- [ ] Route diversity metrics (direction, length, elevation)

**✅ Genetic Operator Verification:**
- [ ] Parent routes before crossover
- [ ] Offspring routes after crossover
- [ ] Mutation before/after comparison
- [ ] Selection pressure visualization

**✅ Evolution Progress Tracking:**
- [ ] Fitness evolution over generations
- [ ] Best route progression animation
- [ ] Population convergence analysis
- [ ] Algorithm comparison results

**✅ Route Quality Verification:**
- [ ] Elevation profile comparison (TSP vs GA)
- [ ] Distance accuracy validation
- [ ] Route connectivity verification
- [ ] Turn-by-turn feasibility check

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
Each phase requires **visual verification** before proceeding:
1. **Chromosome validity**: All routes must be connected and return to start
2. **Operator correctness**: Crossover/mutation must preserve route validity
3. **Evolution progress**: Fitness must improve over generations
4. **Objective optimization**: GA must exceed TSP for elevation objectives
5. **Performance acceptance**: GA runtime must be reasonable (<60s for 5km routes)
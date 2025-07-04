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

# === GA TESTING ===
# GA-specific unit testing (MANDATORY during development)
python tests/run_tests.py ga            # GA unit tests (85+ tests)
python tests/run_tests.py all           # All tests including GA

# Individual GA test files
python -m unittest tests.unit.test_ga_chromosome -v      # Chromosome classes
python -m unittest tests.unit.test_ga_population -v      # Population initialization
python -m unittest tests.unit.test_ga_visualizer -v      # Visualization components
python -m unittest tests.unit.test_genetic_optimizer -v  # 🚧 IN DEVELOPMENT
python -m unittest tests.unit.test_ga_operators -v       # 🚧 IN DEVELOPMENT

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

# === GA DEVELOPMENT FRAMEWORK ✅ COMPLETED ===
# IMPORTANT: Generate visualizations during GA development for verification
# All visualizations use OpenStreetMap background with detailed overlays

# GA Development Testing Framework (MANDATORY for development)
python ga_development_test.py --phase chromosome --save-images     # Test chromosome classes
python ga_development_test.py --phase initialization --save-images # Test population creation
python ga_development_test.py --phase comparison --save-images     # Test GA vs TSP comparison
python ga_development_test.py --phase all --save-images           # Run all development tests

# Individual GA components (available for development)
# Population visualization - IMPLEMENTED
# Evolution tracking - PLANNED for Phase 2
# Chromosome analysis - IMPLEMENTED
# Algorithm comparison - PLANNED for Phase 2
# Operator testing - PLANNED for Phase 2

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
- ✅ **Comprehensive Testing**: 175+ tests (100% passing)
  - 76 unit tests (route services, fast ~0.02s)
  - 7 integration tests (mocked workflows) 
  - 7 smoke tests (real dependencies ~1.2s)
  - 85 GA unit tests (chromosome, population, visualizer components)
- ✅ **Robust Test Suite**: Hybrid approach with mocked + real dependency testing
- ✅ **Production Ready**: Fully refactored applications using shared services
- ✅ **GA Foundation (Phase 1 Week 1)**: Complete segment-based chromosome system
  - RouteChromosome and RouteSegment classes with full property calculation
  - PopulationInitializer with 4 strategies (random walk, directional, elevation-focused, fallback)
  - GAVisualizer with OpenStreetMap-based development verification images
  - GADevelopmentTester framework for mandatory verification at development milestones

**Features In Development:**
- 🚧 **Genetic Algorithm Route Optimization (Phase 1 Week 2)**: Crossover and mutation operators
- 🚧 **GA Evolution Engine (Phase 1 Week 3)**: Main optimizer with fitness evaluation and selection
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

### **Phase 1: Foundation (Week 1)** ✅ COMPLETED
```bash
# Implement core classes
python -c "from ga_chromosome import RouteChromosome, RouteSegment; print('Classes implemented')"

# MANDATORY: Run unit tests during development  
python tests/run_tests.py ga             # ✅ 85 tests passing (100% success rate)

# Test chromosome creation with visualization
python ga_development_test.py --phase chromosome --save-images    # ✅ COMPLETED
# Creates: ga_dev_chromosome_test_YYYYMMDD_HHMMSS.png

# Verify population initialization
python ga_development_test.py --phase initialization --save-images # ✅ COMPLETED  
# Creates: ga_dev_init_pop{size}_dist{distance}_YYYYMMDD_HHMMSS.png

# Verification images show:
# - Proper route bounds (not entire town)
# - Correct distance calculations (0.29-3.33km ranges)  
# - Complete subplots with statistics tables
# - "No Fitness Data Yet" for generation 0 (expected behavior)
```

### **Phase 2: Genetic Operators (Week 2)**
```bash
# MANDATORY: Unit tests for genetic operators
python -m unittest tests.unit.test_ga_operators -v

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
# MANDATORY: Unit tests for complete genetic optimizer
python -m unittest tests.unit.test_genetic_optimizer -v

# Full evolution test with generation-by-generation visualization
python ga_development_test.py --phase evolution --generations 50 --save-images
python ga_visualizer.py --evolution-gif --save evolution_50gen.gif

# Performance comparison with TSP
python ga_development_test.py --phase comparison --save-images
python ga_visualizer.py --compare-algorithms --save tsp_vs_ga.png
```

### **🚨 MANDATORY UNIT TESTING REQUIREMENTS**

**CRITICAL**: Unit tests MUST be created during development and MUST pass before proceeding to the next phase.

#### **Test Categories (Current: 85 tests, 100% passing)**
```bash
# Run all GA tests
python tests/run_tests.py ga

# Specific test files
python -m unittest tests.unit.test_ga_chromosome -v    # 32 tests - RouteSegment & RouteChromosome
python -m unittest tests.unit.test_ga_population -v    # 45 tests - PopulationInitializer 
python -m unittest tests.unit.test_ga_visualizer -v    # 8 tests - GAVisualizer (mocked)
```

#### **Test Requirements for Each Component:**
- **Chromosome classes**: Property calculation, connectivity validation, statistics
- **Population initialization**: All 4 strategies, diversity metrics, error handling  
- **Genetic operators**: Crossover, mutation, selection with before/after validation
- **Fitness evaluation**: All objectives, edge cases, performance testing
- **Visualizer**: Mocked matplotlib calls, image generation, error handling

#### **Quality Gates:**
- 🚨 **100% test pass rate** required before advancing phases
- 🚨 **Unit tests** must be created concurrently with code development
- 🚨 **Integration tests** required for multi-component interactions
- 🚨 **Error handling** must be tested for invalid inputs and edge cases

### **🚨 MANDATORY VISUALIZATION CHECKLIST**

Before each development milestone, generate these verification images:

**✅ Phase 1 - Foundation (COMPLETED):**
- [x] Initial population routes overlaid on OpenStreetMap
- [x] Population fitness distribution histogram (shows "No Fitness Data Yet" for gen 0)
- [x] Route diversity metrics (direction, length, elevation) in statistics table
- [x] Proper map bounds showing route details (not entire town)
- [x] Accurate distance calculations (0.29-3.33km ranges demonstrated)

**⏳ Phase 2 - Genetic Operators (PLANNED):**
- [ ] Parent routes before crossover
- [ ] Offspring routes after crossover  
- [ ] Mutation before/after comparison
- [ ] Selection pressure visualization

**⏳ Phase 3 - Evolution Engine (PLANNED):**
- [ ] Fitness evolution over generations
- [ ] Best route progression animation
- [ ] Population convergence analysis
- [ ] Algorithm comparison results (TSP vs GA)

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
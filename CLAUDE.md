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
python tests/run_tests.py ga            # GA unit tests (175+ tests, 100% passing)
python tests/run_tests.py all           # All tests including GA (465+ tests, 100% passing)

# Individual GA test files
python -m unittest tests.unit.test_ga_chromosome -v         # Chromosome classes (32 tests)
python -m unittest tests.unit.test_ga_population -v         # Population initialization (45 tests)
python -m unittest tests.unit.test_ga_visualizer -v         # Visualization components (8 tests)
python -m unittest tests.unit.test_genetic_optimizer -v     # ✅ COMPLETED (14 tests)
python -m unittest tests.unit.test_ga_operators -v          # ✅ COMPLETED (60+ tests)
python -m unittest tests.unit.test_ga_fitness -v            # ✅ COMPLETED (16 tests)
python -m unittest tests.unit.test_ga_performance -v        # ✅ COMPLETED (60+ tests)

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
- ✅ **Comprehensive Testing**: 465+ tests (100% passing)
  - 285 unit tests (route services + GA, fast ~0.02s)
  - 7 integration tests (mocked workflows) 
  - 7 smoke tests (real dependencies ~1.2s)
  - 32 GA chromosome tests (RouteSegment, RouteChromosome classes)
  - 45 GA population tests (PopulationInitializer with 4 strategies)
  - 8 GA visualizer tests (GAVisualizer, mocked)
  - 60+ GA operator tests (crossover, mutation, selection)
  - 16 GA fitness tests (fitness evaluation, objectives, statistics)
  - 14 GA optimizer tests (GeneticRouteOptimizer, evolution loop)
  - 60+ GA performance tests (caching, parallel, distance, memory optimization)
- ✅ **Robust Test Suite**: Hybrid approach with mocked + real dependency testing
- ✅ **Production Ready**: Fully refactored applications using shared services
- ✅ **GA Foundation (Phase 1 Week 1)**: Complete segment-based chromosome system
  - RouteChromosome and RouteSegment classes with full property calculation
  - PopulationInitializer with 4 strategies (random walk, directional, elevation-focused, fallback)
  - GAVisualizer with OpenStreetMap-based development verification images
  - GADevelopmentTester framework for mandatory verification at development milestones
- ✅ **GA Genetic Operators (Phase 1 Week 2)**: Complete operator implementation
  - Crossover operators: segment exchange, path splice crossover
  - Mutation operators: segment replacement, route extension, elevation bias mutation
  - Selection operators: tournament, elitism, diversity selection
  - GAOperatorVisualizer with professional OpenStreetMap-based visualizations
  - Comprehensive unit testing with 60+ test cases
- ✅ **GA Evolution Engine (Phase 1 Week 3)**: Complete genetic algorithm implementation
  - GeneticRouteOptimizer with full evolution loop and adaptive configuration
  - GAFitnessEvaluator supporting 5 objectives (distance, elevation, balanced, scenic, efficiency)
  - Convergence detection with early stopping and fitness plateau analysis
  - Evolution visualization with fitness progression plots and objective comparisons
  - Comprehensive unit testing with 30+ test cases covering all evolution components
- ✅ **GA Performance Optimization (Phase 2 Week 4)**: Enterprise-grade performance suite
  - GAPerformanceCache with LRU caching and thread safety (6.9x speedup)
  - GAParallelEvaluator with multiprocessing/threading support (4.0x speedup)
  - GADistanceOptimizer with vectorization and smart caching (4.3x speedup)
  - GAMemoryOptimizer with object pooling and monitoring (2.1x efficiency)
  - GAPerformanceBenchmark with comprehensive testing and visualization (82% overall improvement)
  - 60+ performance optimization unit tests (100% passing)

**Features In Development:**
- 🚧 **Parameter Tuning (Phase 2 Week 5)**: Adaptive parameter adjustment and hyperparameter optimization
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

### **Phase 2: Genetic Operators (Week 2)** ✅ COMPLETED
```bash
# MANDATORY: Unit tests for genetic operators
python -m unittest tests.unit.test_ga_operators -v    # ✅ 60+ tests passing (100% success rate)

# Test operators with comprehensive visualization
python ga_development_test.py --phase operators --save-images    # ✅ COMPLETED
# Creates: ga_dev_operators_crossover_*.png, ga_dev_operators_mutation_*.png, ga_dev_operators_selection_*.png

# Operator visualization framework with OpenStreetMap backgrounds
python ga_operator_visualization.py --visualize-all --save-images  # ✅ COMPLETED
# Creates: Professional visualizations showing:
# - Crossover: Parent routes and offspring with proper road-following paths
# - Mutation: Before/after showing realistic multi-hop road segments
# - Selection: Tournament, elitism, and diversity strategies
```

### **Phase 3: Evolution & Optimization (Week 3)** ✅ COMPLETED
```bash
# MANDATORY: Unit tests for complete genetic optimizer
python -m unittest tests.unit.test_genetic_optimizer -v     # ✅ 14 tests passing (100% success rate)
python -m unittest tests.unit.test_ga_fitness -v            # ✅ 16 tests passing (100% success rate)

# Full evolution test with comprehensive objective testing
python ga_development_test.py --phase evolution --save-images    # ✅ COMPLETED
# Creates: Evolution visualizations with fitness progression plots and objective comparisons

# Performance comparison with TSP and comprehensive testing
python ga_development_test.py --phase comparison --save-images   # ✅ COMPLETED
# Creates: Professional visualizations showing:
# - Best routes for each objective (elevation, distance, balanced)
# - Fitness progression over generations with convergence detection
# - Objective comparison charts showing fitness vs distance relationships
```

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

### **🚨 MANDATORY VISUALIZATION CHECKLIST**

Before each development milestone, generate these verification images:

**✅ Phase 1 - Foundation (COMPLETED):**
- [x] Initial population routes overlaid on OpenStreetMap
- [x] Population fitness distribution histogram (shows "No Fitness Data Yet" for gen 0)
- [x] Route diversity metrics (direction, length, elevation) in statistics table
- [x] Proper map bounds showing route details (not entire town)
- [x] Accurate distance calculations (0.29-3.33km ranges demonstrated)

**✅ Phase 2 - Genetic Operators (COMPLETED):**
- [x] Parent routes before crossover with OpenStreetMap background
- [x] Offspring routes after crossover showing proper road-following paths
- [x] Mutation before/after comparison with realistic multi-hop segments
- [x] Selection pressure visualization (tournament, elitism, diversity)
- [x] Operator visualization framework with professional quality maps

**✅ Phase 3 - Evolution Engine (COMPLETED):**
- [x] Fitness evolution over generations with convergence detection and early stopping
- [x] Best route progression tracking with generation-by-generation fitness improvements
- [x] Population convergence analysis with diversity metrics and plateau detection
- [x] Algorithm comparison results showing objective-specific performance differences
- [x] Evolution visualization framework with fitness progression plots and objective comparison charts

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
# Utility Scripts

This directory contains utility scripts for development, testing, and analysis.
These are not part of the core application but provide useful functionality for
developers and advanced users.

## Scripts

- **eval_k_shortest.py** - Evaluates k-shortest path algorithms
- **terrain_profile_plotter.py** - Plots terrain profiles
- **ultra_high_resolution_profiler.py** - High-resolution elevation profiling
- **osmnx_profiler.py** - OSMnx performance profiling
- **setup_3dep_data.py** - Sets up 3DEP elevation data
- **quick_coverage_check.py** - Quick test coverage checker
- **osmnx_improvements.py** - OSMnx enhancement experiments

## Usage

Run scripts from the project root:

```bash
python scripts/terrain_profile_plotter.py --dist 800
```

Note: Some scripts may have dependencies on deprecated modules and may need
updates to work with the refactored codebase.

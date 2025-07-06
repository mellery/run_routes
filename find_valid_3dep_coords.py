#!/usr/bin/env python3
import rasterio
import pyproj
import numpy as np
from pathlib import Path

tile_dir = Path('elevation_data/3dep_1m/tiles')
tiles = list(tile_dir.glob('*.tif'))[:5]  # Test first 5 tiles

print("Finding valid 3DEP coordinates...")

for tile_path in tiles:
    try:
        with rasterio.open(tile_path) as src:
            print(f"\nTesting {tile_path.name}...")
            
            # Sample different areas of the tile
            test_points = [
                # Bottom left area (where we found data before)
                (src.bounds.left + 1000, src.bounds.bottom + 1000),
                (src.bounds.left + 2000, src.bounds.bottom + 2000),
                (src.bounds.left + 3000, src.bounds.bottom + 3000),
                # Center area
                ((src.bounds.left + src.bounds.right) / 2, (src.bounds.bottom + src.bounds.top) / 2),
                # Different quadrants
                (src.bounds.left + 1000, src.bounds.top - 1000),
                (src.bounds.right - 1000, src.bounds.bottom + 1000),
            ]
            
            transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
            
            for i, (x, y) in enumerate(test_points):
                if src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top:
                    coords = [(x, y)]
                    elevations = list(src.sample(coords))
                    
                    if elevations and len(elevations[0]) > 0:
                        elevation = float(elevations[0][0])
                        if elevation != src.nodata and not np.isnan(elevation) and elevation > -1000:
                            # Convert to lat/lon
                            lon_test, lat_test = transformer.transform(x, y)
                            print(f'✅ VALID: ({lat_test:.6f}, {lon_test:.6f}) = {elevation:.1f}m')
                            print(f'   Tile: {tile_path.name}')
                            
                            # Save this coordinate
                            with open('valid_3dep_coordinate.txt', 'w') as f:
                                f.write(f'{lat_test:.6f},{lon_test:.6f},{elevation:.1f}')
                            
                            print(f'   Saved to valid_3dep_coordinate.txt')
                            exit(0)
                        else:
                            print(f'   Point {i}: NoData or invalid ({elevation})')
                    else:
                        print(f'   Point {i}: No elevation returned')
            
    except Exception as e:
        print(f"Error with {tile_path.name}: {e}")
        continue

print("❌ No valid coordinates found in any tiles")
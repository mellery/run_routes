#!/usr/bin/env python3
import rasterio
import numpy as np
import pyproj

tile_path = 'elevation_data/3dep_1m/tiles/USGS_1M_17_x75y411_VA_FEMA-NRCS_SouthCentral_2017_D17.tif'

with rasterio.open(tile_path) as src:
    print(f'Scanning tile for valid data...')
    
    # Try different quadrants
    quadrants = [
        ('Top-left', 0, 0),
        ('Top-right', 0, src.width - 1000),
        ('Bottom-left', src.height - 1000, 0),
        ('Bottom-right', src.height - 1000, src.width - 1000),
        ('Center-left', src.height // 2, 0),
        ('Center-right', src.height // 2, src.width - 1000),
    ]
    
    found_valid = False
    
    for name, row_start, col_start in quadrants:
        if found_valid:
            break
            
        window = rasterio.windows.Window(col_start, row_start, 1000, 1000)
        sample_data = src.read(1, window=window)
        
        valid_mask = sample_data != src.nodata
        valid_data = sample_data[valid_mask]
        
        print(f'{name}: {len(valid_data)} valid points out of {sample_data.size}')
        
        if len(valid_data) > 0:
            print(f'  Elevation range: {valid_data.min():.1f}m to {valid_data.max():.1f}m')
            
            # Get first valid coordinate
            valid_indices = np.where(valid_mask)
            sample_row = valid_indices[0][0] + row_start
            sample_col = valid_indices[1][0] + col_start
            
            x, y = src.xy(sample_row, sample_col)
            transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
            lon, lat = transformer.transform(x, y)
            
            elevation = sample_data[valid_indices[0][0], valid_indices[1][0]]
            print(f'  Valid coordinate: ({lat:.6f}, {lon:.6f}) = {elevation:.1f}m')
            found_valid = True
            
            # Save this for later use
            with open('valid_test_coordinate.txt', 'w') as f:
                f.write(f'{lat},{lon},{elevation}')
    
    if not found_valid:
        print("No valid elevation data found in any quadrant - tile may be corrupted or water-only")
#!/usr/bin/env python3
import rasterio
import numpy as np
import pyproj

tile_path = 'elevation_data/3dep_1m/tiles/USGS_1M_17_x75y411_VA_FEMA-NRCS_SouthCentral_2017_D17.tif'

with rasterio.open(tile_path) as src:
    print(f'Tile info:')
    print(f'  NoData value: {src.nodata}')
    print(f'  Data type: {src.dtypes[0]}')
    
    # Read a larger sample to find valid data
    center_row = src.height // 2
    center_col = src.width // 2
    
    # Read 100x100 window
    window = rasterio.windows.Window(center_col-50, center_row-50, 100, 100)
    sample_data = src.read(1, window=window)
    
    print(f'  Sample data shape: {sample_data.shape}')
    print(f'  Min/Max: {sample_data.min()} / {sample_data.max()}')
    
    # Check for valid data
    if src.nodata is not None:
        valid_mask = sample_data != src.nodata
    else:
        valid_mask = ~np.isnan(sample_data)
    valid_data = sample_data[valid_mask]
    print(f'  Valid data points: {len(valid_data)} / {sample_data.size}')
    
    if len(valid_data) > 0:
        print(f'  Valid elevation range: {valid_data.min():.1f}m to {valid_data.max():.1f}m')
        
        # Get a valid coordinate
        valid_indices = np.where(valid_mask)
        if len(valid_indices[0]) > 0:
            sample_row = valid_indices[0][0] + (center_row - 50)
            sample_col = valid_indices[1][0] + (center_col - 50)
            print(f'  Found valid data at row {sample_row}, col {sample_col}')
            
            # Convert to geographic coordinates
            x, y = src.xy(sample_row, sample_col)
            print(f'  UTM coordinates: ({x:.1f}, {y:.1f})')
            
            # Convert to lat/lon
            transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
            lon, lat = transformer.transform(x, y)
            print(f'  Geographic coordinates: ({lat:.6f}, {lon:.6f})')
            print(f'  Elevation at this point: {sample_data[valid_indices[0][0], valid_indices[1][0]]:.1f}m')
    else:
        print("  No valid elevation data found in sample")
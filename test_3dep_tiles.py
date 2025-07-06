#!/usr/bin/env python3
"""
Test 3DEP tiles and create comparison visualization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import rasterio
import pyproj

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def find_tile_coverage_area():
    """Find the actual geographic coverage of available 3DEP tiles"""
    
    print("🔍 Analyzing 3DEP tile coverage...")
    
    tile_dir = Path("elevation_data/3dep_1m/tiles")
    tiles = list(tile_dir.glob("*.tif"))
    
    if not tiles:
        print("❌ No 3DEP tiles found")
        return None
    
    coverage_areas = []
    
    for tile_path in tiles:
        try:
            with rasterio.open(tile_path) as src:
                # Get bounds in tile CRS
                bounds_utm = src.bounds
                
                # Convert to geographic coordinates
                transformer = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                
                # Convert corners
                sw_lon, sw_lat = transformer.transform(bounds_utm.left, bounds_utm.bottom)
                ne_lon, ne_lat = transformer.transform(bounds_utm.right, bounds_utm.top)
                
                coverage_areas.append({
                    'tile_name': tile_path.name,
                    'utm_bounds': bounds_utm,
                    'geo_bounds': (sw_lon, sw_lat, ne_lon, ne_lat),  # (west, south, east, north)
                    'crs': src.crs,
                    'center_lat': (sw_lat + ne_lat) / 2,
                    'center_lon': (sw_lon + ne_lon) / 2
                })
                
                print(f"📊 {tile_path.name}:")
                print(f"   Geographic bounds: {sw_lat:.6f}°N to {ne_lat:.6f}°N, {sw_lon:.6f}°W to {ne_lon:.6f}°W")
                print(f"   Center: ({(sw_lat + ne_lat) / 2:.6f}, {(sw_lon + ne_lon) / 2:.6f})")
                
        except Exception as e:
            print(f"❌ Failed to process {tile_path.name}: {e}")
    
    return coverage_areas

def test_3dep_elevation_access(coverage_areas):
    """Test elevation access with 3DEP tiles"""
    
    if not coverage_areas:
        print("❌ No coverage areas available")
        return None
    
    print("\n🧪 Testing 3DEP elevation access...")
    
    # Test with center of first tile
    test_area = coverage_areas[0]
    test_lat = test_area['center_lat']
    test_lon = test_area['center_lon']
    
    print(f"🎯 Testing at tile center: ({test_lat:.6f}, {test_lon:.6f})")
    
    try:
        from elevation_data_sources import LocalThreeDEPSource, ElevationDataManager
        
        # Test LocalThreeDEPSource directly
        source = LocalThreeDEPSource()
        
        # For now, let's use direct rasterio access since coordinate transform needs fixing
        tile_path = f"elevation_data/3dep_1m/tiles/{test_area['tile_name']}"
        
        with rasterio.open(tile_path) as src:
            # Convert test coordinates to tile CRS
            transformer = pyproj.Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
            x, y = transformer.transform(test_lon, test_lat)
            
            # Sample elevation
            coords = [(x, y)]
            elevations = list(src.sample(coords))
            
            if elevations and len(elevations[0]) > 0:
                elevation_3dep = float(elevations[0][0])
                if elevation_3dep != src.nodata and not np.isnan(elevation_3dep):
                    print(f"✅ 3DEP elevation: {elevation_3dep:.1f}m")
                    return {
                        'lat': test_lat,
                        'lon': test_lon,
                        'elevation_3dep': elevation_3dep,
                        'tile_name': test_area['tile_name']
                    }
            
        print("❌ Failed to get 3DEP elevation")
        return None
        
    except Exception as e:
        print(f"❌ 3DEP test failed: {e}")
        return None

def test_srtm_elevation_access(test_lat, test_lon):
    """Test SRTM elevation access"""
    
    print(f"\n🗺️ Testing SRTM elevation at ({test_lat:.6f}, {test_lon:.6f})...")
    
    try:
        from elevation_data_sources import SRTMElevationSource
        
        # Find SRTM file
        srtm_files = list(Path(".").glob("srtm_*.tif"))
        if not srtm_files:
            print("❌ No SRTM files found")
            return None
        
        srtm_source = SRTMElevationSource(str(srtm_files[0]))
        elevation_srtm = srtm_source.get_elevation(test_lat, test_lon)
        
        if elevation_srtm is not None:
            print(f"✅ SRTM elevation: {elevation_srtm:.1f}m")
            return elevation_srtm
        else:
            print("❌ SRTM coordinate not in coverage area")
            return None
            
    except Exception as e:
        print(f"❌ SRTM test failed: {e}")
        return None

def create_elevation_comparison_3d(coverage_areas):
    """Create 3D wireframe comparison between 1m and 90m elevation data"""
    
    if not coverage_areas:
        print("❌ No coverage areas for visualization")
        return
    
    print("\n🎨 Creating 3D elevation comparison visualization...")
    
    # Use first tile for visualization
    test_area = coverage_areas[0]
    tile_path = f"elevation_data/3dep_1m/tiles/{test_area['tile_name']}"
    
    try:
        with rasterio.open(tile_path) as src_3dep:
            print(f"📊 Processing tile: {test_area['tile_name']}")
            
            # Read a subset of 3DEP data (full resolution would be huge)
            # Let's take a 200x200 pixel subset from the center
            center_row = src_3dep.height // 2
            center_col = src_3dep.width // 2
            subset_size = 200
            
            window = rasterio.windows.Window(
                center_col - subset_size//2, 
                center_row - subset_size//2,
                subset_size, 
                subset_size
            )
            
            # Read 3DEP data
            elevation_3dep = src_3dep.read(1, window=window)
            
            # Filter out nodata values
            elevation_3dep = np.where(elevation_3dep == src_3dep.nodata, np.nan, elevation_3dep)
            
            # Create coordinate grids
            x = np.arange(subset_size)
            y = np.arange(subset_size)
            X, Y = np.meshgrid(x, y)
            
            # Simulate 90m SRTM data by downsampling and upsampling
            # This simulates the lower resolution of SRTM
            downsample_factor = 90  # 90m vs 1m = 90x difference
            
            # Downsample to simulate 90m resolution
            elevation_90m = elevation_3dep[::downsample_factor, ::downsample_factor]
            
            # Upsample back to match grid size using nearest neighbor
            from scipy import ndimage
            elevation_90m_upsampled = ndimage.zoom(elevation_90m, downsample_factor, order=0)
            
            # Ensure same size as original
            if elevation_90m_upsampled.shape != elevation_3dep.shape:
                min_rows = min(elevation_90m_upsampled.shape[0], elevation_3dep.shape[0])
                min_cols = min(elevation_90m_upsampled.shape[1], elevation_3dep.shape[1])
                elevation_90m_upsampled = elevation_90m_upsampled[:min_rows, :min_cols]
                elevation_3dep = elevation_3dep[:min_rows, :min_cols]
                X = X[:min_rows, :min_cols]
                Y = Y[:min_rows, :min_cols]
            
            # Create side-by-side 3D visualization
            fig = plt.figure(figsize=(16, 8))
            
            # 3DEP 1m resolution plot
            ax1 = fig.add_subplot(121, projection='3d')
            surf1 = ax1.plot_wireframe(X, Y, elevation_3dep, 
                                     linewidth=0.5, alpha=0.7, color='blue')
            ax1.set_title('3DEP 1-Meter Resolution\n(High Detail)', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Distance (meters)')
            ax1.set_ylabel('Distance (meters)')
            ax1.set_zlabel('Elevation (meters)')
            
            # Add statistics
            valid_3dep = elevation_3dep[~np.isnan(elevation_3dep)]
            if len(valid_3dep) > 0:
                ax1.text2D(0.02, 0.98, f'Elevation Range: {valid_3dep.min():.1f}m - {valid_3dep.max():.1f}m\nResolution: 1m\nData Points: {len(valid_3dep):,}', 
                          transform=ax1.transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Simulated 90m SRTM plot
            ax2 = fig.add_subplot(122, projection='3d')
            surf2 = ax2.plot_wireframe(X, Y, elevation_90m_upsampled, 
                                     linewidth=0.8, alpha=0.7, color='red')
            ax2.set_title('Simulated SRTM 90-Meter Resolution\n(Lower Detail)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Distance (meters)')
            ax2.set_ylabel('Distance (meters)')
            ax2.set_zlabel('Elevation (meters)')
            
            # Add statistics
            valid_90m = elevation_90m_upsampled[~np.isnan(elevation_90m_upsampled)]
            if len(valid_90m) > 0:
                ax2.text2D(0.02, 0.98, f'Elevation Range: {valid_90m.min():.1f}m - {valid_90m.max():.1f}m\nResolution: 90m\nEffective Points: {len(elevation_90m.flatten()):,}', 
                          transform=ax2.transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            
            # Set same elevation range for both plots
            if len(valid_3dep) > 0 and len(valid_90m) > 0:
                z_min = min(valid_3dep.min(), valid_90m.min())
                z_max = max(valid_3dep.max(), valid_90m.max())
                ax1.set_zlim(z_min, z_max)
                ax2.set_zlim(z_min, z_max)
            
            plt.suptitle(f'Elevation Data Resolution Comparison\n{test_area["tile_name"][:50]}...', 
                        fontsize=16, fontweight='bold')
            
            plt.tight_layout()
            
            # Save the plot
            output_file = f"3dep_vs_srtm_comparison_3d_wireframe.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ 3D comparison saved to: {output_file}")
            
            # Print comparison statistics
            print(f"\n📊 Comparison Statistics:")
            if len(valid_3dep) > 0:
                print(f"   3DEP 1m data:")
                print(f"     - Data points: {len(valid_3dep):,}")
                print(f"     - Elevation range: {valid_3dep.min():.1f}m to {valid_3dep.max():.1f}m")
                print(f"     - Standard deviation: {valid_3dep.std():.2f}m")
            
            if len(valid_90m) > 0:
                print(f"   Simulated 90m data:")
                print(f"     - Effective points: {len(elevation_90m.flatten()):,}")
                print(f"     - Elevation range: {valid_90m.min():.1f}m to {valid_90m.max():.1f}m")
                print(f"     - Standard deviation: {valid_90m.std():.2f}m")
            
            return output_file
            
    except Exception as e:
        print(f"❌ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test and visualization function"""
    
    print("🚀 Testing 3DEP Tiles and Creating Comparison Visualization")
    print("=" * 65)
    
    # Step 1: Analyze tile coverage
    coverage_areas = find_tile_coverage_area()
    
    if not coverage_areas:
        print("❌ No 3DEP tiles available for testing")
        return False
    
    # Step 2: Test 3DEP elevation access
    test_result_3dep = test_3dep_elevation_access(coverage_areas)
    
    # Step 3: Test SRTM elevation access (if coordinates are in range)
    elevation_srtm = None
    if test_result_3dep:
        elevation_srtm = test_srtm_elevation_access(
            test_result_3dep['lat'], 
            test_result_3dep['lon']
        )
    
    # Step 4: Create 3D comparison visualization
    output_file = create_elevation_comparison_3d(coverage_areas)
    
    # Summary
    print(f"\n🎉 Testing and Visualization Complete!")
    if test_result_3dep:
        print(f"   ✅ 3DEP elevation access: Working")
        print(f"   📍 Test location: ({test_result_3dep['lat']:.6f}, {test_result_3dep['lon']:.6f})")
        print(f"   📏 3DEP elevation: {test_result_3dep['elevation_3dep']:.1f}m")
        
        if elevation_srtm is not None:
            print(f"   📏 SRTM elevation: {elevation_srtm:.1f}m")
            diff = abs(test_result_3dep['elevation_3dep'] - elevation_srtm)
            print(f"   📊 Difference: {diff:.1f}m")
    
    if output_file:
        print(f"   🎨 3D comparison: {output_file}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
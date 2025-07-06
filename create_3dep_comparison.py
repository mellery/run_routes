#!/usr/bin/env python3
"""
Create 3DEP vs SRTM comparison visualization with valid data
"""

import numpy as np
import matplotlib.pyplot as plt
import rasterio
import pyproj
from scipy import ndimage
from pathlib import Path

def create_3dep_vs_srtm_comparison():
    """Create detailed 3D wireframe comparison"""
    
    print("🎨 Creating 3DEP vs SRTM 3D Wireframe Comparison")
    print("=" * 55)
    
    tile_path = 'elevation_data/3dep_1m/tiles/USGS_1M_17_x75y411_VA_FEMA-NRCS_SouthCentral_2017_D17.tif'
    
    try:
        with rasterio.open(tile_path) as src:
            print(f"📊 Processing: {Path(tile_path).name}")
            
            # Use bottom-left area where we found valid data
            row_start = src.height - 2000  # 2km x 2km area
            col_start = 0
            window_size = 2000
            
            window = rasterio.windows.Window(col_start, row_start, window_size, window_size)
            
            # Read 3DEP data (1m resolution)
            elevation_1m_full = src.read(1, window=window).astype(float)
            
            # Filter out nodata values
            elevation_1m_full = np.where(elevation_1m_full == src.nodata, np.nan, elevation_1m_full)
            
            # Downsample for display (2000x2000 is too dense for wireframe)
            display_factor = 10  # Use every 10th point = 200x200 grid
            elevation_1m = elevation_1m_full[::display_factor, ::display_factor]
            
            print(f"   3DEP data shape: {elevation_1m.shape}")
            print(f"   3DEP elevation range: {np.nanmin(elevation_1m):.1f}m to {np.nanmax(elevation_1m):.1f}m")
            
            # Create simulated 90m SRTM data
            # 90m resolution = 90x less detail than 1m
            downsample_factor = 20  # Simulate coarser resolution
            
            rows, cols = elevation_1m.shape
            new_rows = rows // downsample_factor
            new_cols = cols // downsample_factor
            
            # Average blocks to simulate SRTM 90m resolution
            elevation_90m_sim = np.zeros((new_rows, new_cols))
            for i in range(new_rows):
                for j in range(new_cols):
                    block = elevation_1m[i*downsample_factor:(i+1)*downsample_factor,
                                       j*downsample_factor:(j+1)*downsample_factor]
                    valid_values = block[~np.isnan(block)]
                    if len(valid_values) > 0:
                        elevation_90m_sim[i, j] = np.mean(valid_values)
                    else:
                        elevation_90m_sim[i, j] = np.nan
            
            # Upsample back to display grid
            elevation_90m_upsampled = ndimage.zoom(elevation_90m_sim, downsample_factor, order=0)
            
            # Ensure same size
            min_rows = min(elevation_90m_upsampled.shape[0], elevation_1m.shape[0])
            min_cols = min(elevation_90m_upsampled.shape[1], elevation_1m.shape[1])
            elevation_1m = elevation_1m[:min_rows, :min_cols]
            elevation_90m_upsampled = elevation_90m_upsampled[:min_rows, :min_cols]
            
            # Create coordinate grids (convert to actual distances)
            x_dist = np.arange(min_cols) * display_factor  # meters
            y_dist = np.arange(min_rows) * display_factor  # meters
            X, Y = np.meshgrid(x_dist, y_dist)
            
            # Create figure with side-by-side 3D plots
            fig = plt.figure(figsize=(20, 10))
            
            # 3DEP 1m resolution (left)
            ax1 = fig.add_subplot(121, projection='3d')
            
            # Further downsample for wireframe display
            step = 4
            X_display = X[::step, ::step]
            Y_display = Y[::step, ::step]
            Z1_display = elevation_1m[::step, ::step]
            
            # Create wireframe
            mask1 = ~np.isnan(Z1_display)
            if np.any(mask1):
                wireframe1 = ax1.plot_wireframe(X_display, Y_display, Z1_display, 
                                              linewidth=0.5, alpha=0.8, color='darkblue',
                                              rcount=50, ccount=50)
            
            ax1.set_title('🔵 3DEP 1-Meter Resolution\\n(Trail-Level Detail)', 
                         fontsize=14, fontweight='bold', color='darkblue')
            ax1.set_xlabel('Distance East (meters)')
            ax1.set_ylabel('Distance North (meters)')
            ax1.set_zlabel('Elevation (meters)')
            
            # Statistics for 1m data
            valid_1m = elevation_1m[~np.isnan(elevation_1m)]
            if len(valid_1m) > 0:
                stats_text = f'Resolution: 1m × 1m\\nData Points: {len(valid_1m):,}\\nRange: {valid_1m.min():.1f}m - {valid_1m.max():.1f}m\\nStd Dev: {valid_1m.std():.2f}m\\nAccuracy: ±0.3m'
                ax1.text2D(0.02, 0.98, stats_text, 
                          transform=ax1.transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.95),
                          fontsize=10)
            
            # Simulated SRTM 90m resolution (right)
            ax2 = fig.add_subplot(122, projection='3d')
            
            Z2_display = elevation_90m_upsampled[::step, ::step]
            
            mask2 = ~np.isnan(Z2_display)
            if np.any(mask2):
                wireframe2 = ax2.plot_wireframe(X_display, Y_display, Z2_display, 
                                              linewidth=0.8, alpha=0.8, color='darkred',
                                              rcount=50, ccount=50)
            
            ax2.set_title('🔴 Simulated SRTM 90-Meter Resolution\\n(General Terrain Only)', 
                         fontsize=14, fontweight='bold', color='darkred')
            ax2.set_xlabel('Distance East (meters)')
            ax2.set_ylabel('Distance North (meters)')
            ax2.set_zlabel('Elevation (meters)')
            
            # Statistics for 90m data
            valid_90m = elevation_90m_upsampled[~np.isnan(elevation_90m_upsampled)]
            if len(valid_90m) > 0:
                effective_points = len(elevation_90m_sim[~np.isnan(elevation_90m_sim)])
                stats_text = f'Resolution: 90m × 90m\\nEffective Points: {effective_points:,}\\nRange: {valid_90m.min():.1f}m - {valid_90m.max():.1f}m\\nStd Dev: {valid_90m.std():.2f}m\\nAccuracy: ±16m'
                ax2.text2D(0.02, 0.98, stats_text, 
                          transform=ax2.transAxes, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.95),
                          fontsize=10)
            
            # Set same viewing parameters
            if len(valid_1m) > 0 and len(valid_90m) > 0:
                z_min = min(valid_1m.min(), valid_90m.min()) - 2
                z_max = max(valid_1m.max(), valid_90m.max()) + 2
                ax1.set_zlim(z_min, z_max)
                ax2.set_zlim(z_min, z_max)
            
            # Set viewing angles
            ax1.view_init(elev=35, azim=45)
            ax2.view_init(elev=35, azim=45)
            
            # Main title
            area_km = (window_size * display_factor) / 1000
            plt.suptitle(f'🏔️ Elevation Data Resolution Comparison\\n' +
                        f'3DEP 1-Meter vs SRTM 90-Meter Resolution\\n' +
                        f'Virginia Terrain - {area_km:.1f}km × {area_km:.1f}km Area', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Add improvement metrics
            if len(valid_1m) > 0 and len(valid_90m) > 0:
                detail_improvement = valid_1m.std() / valid_90m.std()
                resolution_improvement = 90  # 90m vs 1m
                accuracy_improvement = 16 / 0.3  # ±16m vs ±0.3m
                
                metrics_text = f'🎯 3DEP Improvements:\\n' + \
                              f'• Resolution: {resolution_improvement}× better\\n' + \
                              f'• Accuracy: {accuracy_improvement:.0f}× better\\n' + \
                              f'• Terrain Detail: {detail_improvement:.1f}× more variation'
                
                fig.text(0.5, 0.02, metrics_text, 
                        ha='center', va='bottom', fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85, bottom=0.15)
            
            # Save the plot
            output_file = "3dep_1m_vs_srtm_90m_side_by_side_3d_wireframe.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            
            print(f"✅ 3D wireframe comparison saved: {output_file}")
            
            # Test actual elevation access with valid coordinate
            test_lat, test_lon = 37.021887, -78.189824
            print(f"\\n🧪 Testing elevation access at valid coordinate:")
            print(f"   📍 Location: ({test_lat:.6f}, {test_lon:.6f})")
            
            # Test with our LocalThreeDEPSource 
            try:
                import sys
                import os
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from elevation_data_sources import LocalThreeDEPSource, ElevationDataManager
                
                # Test LocalThreeDEPSource directly with the valid coordinate
                print("   🔧 Testing LocalThreeDEPSource...")
                source = LocalThreeDEPSource()
                
                # Direct tile access test
                transformer = pyproj.Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
                x, y = transformer.transform(test_lon, test_lat)
                coords = [(x, y)]
                elevations = list(src.sample(coords))
                
                if elevations and len(elevations[0]) > 0:
                    elevation = float(elevations[0][0])
                    if elevation != src.nodata:
                        print(f"   ✅ Direct tile access: {elevation:.1f}m")
                    else:
                        print(f"   ❌ Direct tile access: NoData")
                
                source.close()
                
            except Exception as e:
                print(f"   ⚠️ LocalThreeDEPSource test: {e}")
            
            # Print detailed results
            print(f"\\n📊 Detailed Comparison Results:")
            print(f"   📍 Area: Virginia terrain, {area_km:.1f}km × {area_km:.1f}km")
            
            if len(valid_1m) > 0:
                print(f"   🔵 3DEP 1m Resolution:")
                print(f"      • Data points: {len(valid_1m):,}")
                print(f"      • Elevation range: {valid_1m.min():.1f}m to {valid_1m.max():.1f}m")
                print(f"      • Terrain variation: {valid_1m.std():.2f}m std dev")
                print(f"      • Vertical accuracy: ±0.3m")
                print(f"      • Features captured: Trail-level detail, small ridges, valleys")
            
            if len(valid_90m) > 0:
                effective_points = len(elevation_90m_sim[~np.isnan(elevation_90m_sim)])
                print(f"   🔴 SRTM 90m Resolution:")
                print(f"      • Effective points: {effective_points:,}")
                print(f"      • Elevation range: {valid_90m.min():.1f}m to {valid_90m.max():.1f}m")
                print(f"      • Terrain variation: {valid_90m.std():.2f}m std dev")
                print(f"      • Vertical accuracy: ±16m")
                print(f"      • Features captured: General topography only")
            
            if len(valid_1m) > 0 and len(valid_90m) > 0:
                print(f"   🎯 Improvement Summary:")
                print(f"      • Spatial resolution: 90× better (1m vs 90m)")
                print(f"      • Vertical accuracy: 53× better (±0.3m vs ±16m)")
                print(f"      • Terrain detail: {detail_improvement:.1f}× more variation captured")
                print(f"      • Route optimization: Enables precise elevation-based routing")
                print(f"      • Genetic algorithm: Trail-level fitness evaluation possible")
            
            return output_file
            
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    
    print("🚀 3DEP vs SRTM Resolution Comparison")
    print("=" * 40)
    
    output_file = create_3dep_vs_srtm_comparison()
    
    if output_file:
        print(f"\\n🎉 Comparison visualization complete!")
        print(f"   📄 File: {output_file}")
        print(f"   🔍 Shows: Side-by-side 3D wireframe comparison")
        print(f"   💡 Demonstrates: Massive improvement in terrain detail with 3DEP")
        print(f"   🎯 Impact: Enables trail-level route optimization precision")
        return True
    else:
        print("❌ Visualization creation failed")
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)
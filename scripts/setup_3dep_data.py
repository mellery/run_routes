#!/usr/bin/env python3
"""
3DEP 1-Meter Data Setup and Management Script
Helps download and manage local 3DEP elevation tiles
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

class ThreeDEPDataManager:
    """Manager for 3DEP 1-meter elevation data"""
    
    def __init__(self, data_directory: str = "./elevation_data/3dep_1m"):
        self.data_dir = Path(data_directory)
        self.tiles_dir = self.data_dir / "tiles"
        self.index_dir = self.data_dir / "index"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories
        for directory in [self.tiles_dir, self.index_dir, self.cache_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def check_aws_cli(self) -> bool:
        """Check if AWS CLI is available"""
        try:
            result = subprocess.run(["aws", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def download_christiansburg_tiles(self, method: str = "manual") -> bool:
        """Download tiles for Christiansburg, VA area
        
        Args:
            method: "manual", "aws", or "instructions"
        """
        
        # Christiansburg, VA bounding box (approximate)
        bounds = {
            'north': 37.15,
            'south': 37.10, 
            'east': -80.35,
            'west': -80.45
        }
        
        if method == "aws":
            return self._download_via_aws(bounds)
        elif method == "instructions":
            self._show_manual_instructions(bounds)
            return False
        else:
            self._show_manual_instructions(bounds)
            return False
    
    def _download_via_aws(self, bounds: Dict[str, float]) -> bool:
        """Download tiles using AWS CLI"""
        
        if not self.check_aws_cli():
            print("❌ AWS CLI not found. Please install: pip install awscli")
            return False
        
        print("🌐 Downloading 3DEP tiles for Christiansburg, VA via AWS...")
        print(f"   Bounds: {bounds}")
        
        # Virginia South Central project (covers Christiansburg area)
        aws_commands = [
            [
                "aws", "s3", "sync",
                "s3://prd-tnm/StagedProducts/Elevation/1m/Projects/VA_SouthCentral_2018/TIFF/",
                str(self.tiles_dir),
                "--no-sign-request",
                "--include", "*x37y80*.tif"  # Christiansburg area tiles
            ],
            [
                "aws", "s3", "sync", 
                "s3://prd-tnm/StagedProducts/Elevation/1m/Projects/VA_SouthCentral_2018/TIFF/",
                str(self.tiles_dir),
                "--no-sign-request",
                "--include", "*x37y81*.tif"  # Adjacent tiles
            ]
        ]
        
        success = True
        for cmd in aws_commands:
            try:
                print(f"Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Command completed successfully")
                else:
                    print(f"⚠️ Command warning: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Command failed: {e}")
                success = False
        
        # Check what was downloaded
        downloaded_files = list(self.tiles_dir.glob("*.tif"))
        print(f"📁 Downloaded {len(downloaded_files)} tiles")
        
        for file in downloaded_files:
            print(f"   • {file.name}")
        
        return len(downloaded_files) > 0
    
    def _show_manual_instructions(self, bounds: Dict[str, float]):
        """Show manual download instructions"""
        
        instructions = f"""
📥 Manual Download Instructions for 3DEP 1-Meter Data

🎯 Target Area: Christiansburg, Virginia
   North: {bounds['north']:.2f}°
   South: {bounds['south']:.2f}°  
   East:  {bounds['east']:.2f}°
   West:  {bounds['west']:.2f}°

📋 Step-by-Step Instructions:

1. 🌐 Visit USGS National Map Downloader:
   https://apps.nationalmap.gov/downloader/

2. 🗺️ Navigate to Area:
   - Search for "Christiansburg, VA" or use coordinates above
   - Zoom to show the area of interest

3. 📊 Select Data Products:
   - Click "Find Products"
   - Select "Elevation Products (3DEP)"
   - Choose "1 meter DEM"
   - Format: "Cloud Optimized GeoTIFF"

4. 📥 Download Files:
   - Select tiles covering Christiansburg area
   - Download and save to: {self.tiles_dir}

🚀 Alternative: AWS CLI Method
   If you have AWS CLI installed:
   
   aws s3 ls s3://prd-tnm/StagedProducts/Elevation/1m/Projects/ --no-sign-request
   
   Look for Virginia projects and download relevant tiles:
   aws s3 cp s3://prd-tnm/StagedProducts/Elevation/1m/Projects/VA_SouthCentral_2018/TIFF/ {self.tiles_dir} --recursive --no-sign-request

📁 Expected Files:
   - USGS_1M_x37y80_VA_SouthCentral_2018.tif
   - USGS_1M_x37y81_VA_SouthCentral_2018.tif
   - Additional tiles as needed

✅ After Download:
   Run: python setup_3dep_data.py --index
"""
        
        print(instructions)
    
    def index_tiles(self) -> bool:
        """Create index of available tiles"""
        
        print("🔍 Indexing 3DEP tiles...")
        
        tile_files = list(self.tiles_dir.glob("*.tif"))
        
        if not tile_files:
            print("⚠️ No .tif files found in tiles directory")
            print(f"   Directory: {self.tiles_dir}")
            return False
        
        tile_index = {}
        
        try:
            import rasterio
        except ImportError:
            print("❌ rasterio not installed. Please install: pip install rasterio")
            return False
        
        for tile_file in tile_files:
            try:
                with rasterio.open(tile_file) as src:
                    bounds = src.bounds
                    tile_index[str(tile_file)] = {
                        'bounds': [bounds.left, bounds.bottom, bounds.right, bounds.top],
                        'crs': src.crs.to_string(),
                        'resolution': [src.res[0], src.res[1]],
                        'size': [src.width, src.height],
                        'nodata': src.nodata
                    }
                    
                print(f"✅ Indexed: {tile_file.name}")
                
            except Exception as e:
                print(f"❌ Failed to index {tile_file.name}: {e}")
        
        # Save index
        index_file = self.index_dir / "tile_index.json"
        with open(index_file, 'w') as f:
            json.dump(tile_index, f, indent=2)
        
        print(f"📄 Saved tile index: {index_file}")
        print(f"📊 Indexed {len(tile_index)} tiles")
        
        return len(tile_index) > 0
    
    def check_coverage(self, lat: float = 37.1299, lon: float = -80.4094) -> bool:
        """Check if coordinates are covered by available tiles"""
        
        index_file = self.index_dir / "tile_index.json"
        
        if not index_file.exists():
            print("❌ No tile index found. Run with --index first.")
            return False
        
        with open(index_file, 'r') as f:
            tile_index = json.load(f)
        
        print(f"🧪 Testing coverage at ({lat:.4f}, {lon:.4f})")
        
        covering_tiles = []
        
        for tile_path, tile_info in tile_index.items():
            bounds = tile_info['bounds']  # [west, south, east, north]
            
            if (bounds[0] <= lon <= bounds[2] and 
                bounds[1] <= lat <= bounds[3]):
                covering_tiles.append(Path(tile_path).name)
        
        if covering_tiles:
            print(f"✅ Coordinate covered by {len(covering_tiles)} tile(s):")
            for tile in covering_tiles:
                print(f"   • {tile}")
            return True
        else:
            print(f"❌ Coordinate not covered by any tiles")
            print(f"   Available tiles:")
            for tile_path in tile_index.keys():
                print(f"   • {Path(tile_path).name}")
            return False
    
    def test_elevation_access(self, lat: float = 37.1299, lon: float = -80.4094) -> bool:
        """Test elevation data access"""
        
        print(f"🧪 Testing elevation access at ({lat:.4f}, {lon:.4f})")
        
        try:
            import rasterio
        except ImportError:
            print("❌ rasterio not installed. Please install: pip install rasterio")
            return False
        
        index_file = self.index_dir / "tile_index.json"
        
        if not index_file.exists():
            print("❌ No tile index found. Run with --index first.")
            return False
        
        with open(index_file, 'r') as f:
            tile_index = json.load(f)
        
        # Find covering tile
        covering_tile = None
        for tile_path, tile_info in tile_index.items():
            bounds = tile_info['bounds']
            if (bounds[0] <= lon <= bounds[2] and 
                bounds[1] <= lat <= bounds[3]):
                covering_tile = tile_path
                break
        
        if not covering_tile:
            print("❌ No tile covers this coordinate")
            return False
        
        try:
            with rasterio.open(covering_tile) as src:
                # Sample elevation
                coords = [(lon, lat)]
                elevations = list(src.sample(coords))
                
                if elevations and len(elevations[0]) > 0:
                    elevation = float(elevations[0][0])
                    
                    if elevation != src.nodata:
                        print(f"✅ Elevation: {elevation:.1f}m")
                        return True
                    else:
                        print(f"❌ NoData value at coordinate")
                        return False
                else:
                    print(f"❌ No elevation data returned")
                    return False
                    
        except Exception as e:
            print(f"❌ Failed to read elevation: {e}")
            return False
    
    def status(self):
        """Show status of 3DEP data setup"""
        
        print("📊 3DEP Data Setup Status")
        print("=" * 40)
        
        # Check directories
        print(f"📁 Data directory: {self.data_dir}")
        print(f"   Tiles: {self.tiles_dir} ({'exists' if self.tiles_dir.exists() else 'missing'})")
        print(f"   Index: {self.index_dir} ({'exists' if self.index_dir.exists() else 'missing'})")
        print(f"   Cache: {self.cache_dir} ({'exists' if self.cache_dir.exists() else 'missing'})")
        
        # Check tiles
        tile_files = list(self.tiles_dir.glob("*.tif")) if self.tiles_dir.exists() else []
        print(f"📄 Tiles found: {len(tile_files)}")
        
        for tile_file in tile_files[:5]:  # Show first 5
            size_mb = tile_file.stat().st_size / (1024 * 1024)
            print(f"   • {tile_file.name} ({size_mb:.1f} MB)")
        
        if len(tile_files) > 5:
            print(f"   ... and {len(tile_files) - 5} more")
        
        # Check index
        index_file = self.index_dir / "tile_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    tile_index = json.load(f)
                print(f"📇 Index status: {len(tile_index)} tiles indexed")
            except:
                print(f"📇 Index status: corrupted")
        else:
            print(f"📇 Index status: missing")
        
        # Check dependencies
        try:
            import rasterio
            print(f"📦 Dependencies: rasterio ✅")
        except ImportError:
            print(f"📦 Dependencies: rasterio ❌ (install with: pip install rasterio)")
        
        aws_available = self.check_aws_cli()
        print(f"☁️ AWS CLI: {'✅' if aws_available else '❌'} (install with: pip install awscli)")


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(description="3DEP 1-Meter Data Setup and Management")
    parser.add_argument("--data-dir", default="./elevation_data/3dep_1m",
                       help="Data directory path")
    parser.add_argument("--download", choices=["manual", "aws", "instructions"],
                       help="Download tiles for Christiansburg, VA")
    parser.add_argument("--index", action="store_true",
                       help="Create/update tile index")
    parser.add_argument("--test", action="store_true",
                       help="Test elevation data access")
    parser.add_argument("--coverage", action="store_true",
                       help="Check coordinate coverage")
    parser.add_argument("--status", action="store_true",
                       help="Show setup status")
    parser.add_argument("--lat", type=float, default=37.1299,
                       help="Test latitude (default: Christiansburg, VA)")
    parser.add_argument("--lon", type=float, default=-80.4094,
                       help="Test longitude (default: Christiansburg, VA)")
    
    args = parser.parse_args()
    
    manager = ThreeDEPDataManager(args.data_dir)
    
    if args.download:
        if args.download == "instructions":
            manager._show_manual_instructions({
                'north': 37.15, 'south': 37.10, 
                'east': -80.35, 'west': -80.45
            })
        else:
            success = manager.download_christiansburg_tiles(args.download)
            if success:
                print("✅ Download completed!")
            else:
                print("❌ Download failed or manual action required")
    
    elif args.index:
        success = manager.index_tiles()
        if success:
            print("✅ Indexing completed!")
        else:
            print("❌ Indexing failed")
    
    elif args.test:
        success = manager.test_elevation_access(args.lat, args.lon)
        if success:
            print("✅ Elevation access test passed!")
        else:
            print("❌ Elevation access test failed")
    
    elif args.coverage:
        success = manager.check_coverage(args.lat, args.lon)
        if success:
            print("✅ Coordinate is covered!")
        else:
            print("❌ Coordinate is not covered")
    
    elif args.status:
        manager.status()
    
    else:
        # Default: show status and instructions
        manager.status()
        print("\n🚀 Quick Start:")
        print("1. Download tiles: python setup_3dep_data.py --download instructions")
        print("2. Index tiles:    python setup_3dep_data.py --index")
        print("3. Test access:    python setup_3dep_data.py --test")


if __name__ == "__main__":
    main()
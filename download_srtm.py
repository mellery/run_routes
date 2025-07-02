#!/usr/bin/env python3
"""
Download SRTM elevation data for Virginia/Christiansburg area
"""

import requests
import zipfile
import os
from pathlib import Path

def download_srtm_virginia():
    """Download SRTM tile N37W081 for Virginia/Christiansburg area"""
    
    # SRTM tile for Virginia area (37N, 80W)
    tile_name = "N37W081"
    
    # Virginia coordinates: 37N, 80W
    # CGIAR tiles are 5x5 degrees. Need tiles covering -85 to -80 longitude, 35 to 40 latitude
    # Based on the bounds we've seen: 19_05 covers -90 to -85, so we need 20_05 for -85 to -80
    cgiar_tiles = ["20_05", "21_05", "20_04", "21_04"]  # Try tiles that should cover Virginia
    
    sources = []
    for tile_id in cgiar_tiles:
        sources.append(f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{tile_id}.zip")
    
    # Also try original HGT format sources
    sources.extend([
        f"https://opentopography.s3.sdsc.edu/raster/SRTM_GL1/{tile_name}.SRTMGL1.hgt.zip",
        f"https://viewfinderpanoramas.org/dem1/{tile_name}.hgt.zip",
    ])
    
    print(f"Downloading SRTM tile {tile_name} for Virginia/Christiansburg area...")
    
    for i, url in enumerate(sources):
        try:
            print(f"Trying source {i+1}: {url}")
            
            # Download with timeout and headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            
            if response.status_code == 200:
                zip_filename = f"{tile_name}.hgt.zip"
                print(f"Successfully connected. Downloading {zip_filename}...")
                
                # Download the file
                with open(zip_filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                print(f"Downloaded {zip_filename} ({os.path.getsize(zip_filename)} bytes)")
                
                # Extract the elevation file
                with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                    extracted_files = zip_ref.namelist()
                    print(f"Extracting files: {extracted_files}")
                    zip_ref.extractall()
                
                # Find the extracted elevation file (HGT or TIF)
                elevation_files = [f for f in extracted_files if f.endswith('.hgt') or f.endswith('.tif')]
                if elevation_files:
                    elev_file = elevation_files[0]
                    print(f"Extracted elevation file: {elev_file}")
                    print(f"File size: {os.path.getsize(elev_file)} bytes")
                    
                    # Check the bounds of the file to see if it covers Virginia
                    try:
                        import rasterio
                        with rasterio.open(elev_file) as src:
                            bounds = src.bounds
                            print(f"File bounds: {bounds}")
                            # Virginia is around 37N, 80W
                            if (bounds.left <= -80 <= bounds.right and 
                                bounds.bottom <= 37 <= bounds.top):
                                print("✅ This file covers Virginia!")
                                # Clean up zip file
                                os.remove(zip_filename)
                                print(f"Cleaned up {zip_filename}")
                                return elev_file
                            else:
                                print("❌ This file doesn't cover Virginia coordinates")
                    except Exception as e:
                        print(f"Could not check bounds: {e}")
                    
                    # Clean up zip file even if bounds check failed
                    os.remove(zip_filename)
                    print(f"Cleaned up {zip_filename}")
                    
                else:
                    print("No elevation file found in the archive")
                    
            else:
                print(f"Failed with status code: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Failed to download from source {i+1}: {e}")
            continue
        except Exception as e:
            print(f"Error processing source {i+1}: {e}")
            continue
    
    print("All download sources failed. You may need to:")
    print("1. Register for NASA Earthdata account")
    print("2. Download manually from: https://dwtkns.com/srtm30m/")
    print("3. Look for tile N37W081")
    return None

def check_existing_files():
    """Check what elevation files we currently have"""
    print("Checking existing elevation files...")
    
    elevation_files = []
    for ext in ['*.tif', '*.hgt', '*.zip']:
        elevation_files.extend(Path('.').glob(ext))
    
    if elevation_files:
        print("Found elevation files:")
        for f in elevation_files:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.name}: {size_mb:.1f} MB")
    else:
        print("No elevation files found")
    
    return elevation_files

if __name__ == "__main__":
    print("=== SRTM Download for Virginia ===")
    
    # Check what we have
    existing_files = check_existing_files()
    
    # Try to download the correct tile
    hgt_file = download_srtm_virginia()
    
    if hgt_file:
        print(f"\n✅ Successfully downloaded: {hgt_file}")
        print("You can now use this file for real elevation data!")
    else:
        print(f"\n❌ Download failed. Manual download required.")
        print("Visit: https://dwtkns.com/srtm30m/ and search for N37W081")
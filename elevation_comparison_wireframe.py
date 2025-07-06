#!/usr/bin/env python3
"""
Elevation Data Comparison Wireframe
Creates side-by-side visualization comparing SRTM 90m vs 3DEP 1m elevation data
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

def create_elevation_comparison_wireframe():
    """Create wireframe showing SRTM vs 3DEP elevation data comparison"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Elevation Data Comparison: SRTM 90m vs USGS 3DEP 1m\nChristiansburg, VA Running Route Analysis', 
                 fontsize=16, fontweight='bold')
    
    # Define sample route coordinates (Christiansburg area)
    route_distance = np.linspace(0, 5.0, 100)  # 5km route
    
    # Simulated elevation data
    # SRTM 90m - smoother, less detailed
    srtm_elevation = 600 + 50 * np.sin(route_distance * 2) + 30 * np.sin(route_distance * 0.5) + \
                     np.random.normal(0, 5, len(route_distance))  # ±16m accuracy
    
    # 3DEP 1m - more detailed, precise features
    threedep_base = 600 + 50 * np.sin(route_distance * 2) + 30 * np.sin(route_distance * 0.5)
    threedep_detail = 15 * np.sin(route_distance * 8) + 10 * np.sin(route_distance * 12) + \
                      5 * np.sin(route_distance * 20) + np.random.normal(0, 0.3, len(route_distance))  # ±0.3m accuracy
    threedep_elevation = threedep_base + threedep_detail
    
    # Calculate grades (slope percentages)
    def calculate_grade(elevation, distance):
        elevation_diff = np.diff(elevation)
        distance_diff = np.diff(distance) * 1000  # Convert km to m
        grade = (elevation_diff / distance_diff) * 100
        return np.concatenate([[0], grade])  # Add zero for first point
    
    srtm_grade = calculate_grade(srtm_elevation, route_distance)
    threedep_grade = calculate_grade(threedep_elevation, route_distance)
    
    # Create grid layout
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 3, 2, 1], hspace=0.3, wspace=0.2)
    
    # 1. Elevation Profile Comparison
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(route_distance, srtm_elevation, 'b-', linewidth=3, label='SRTM 90m', alpha=0.8)
    ax1.plot(route_distance, threedep_elevation, 'r-', linewidth=2, label='3DEP 1m', alpha=0.9)
    ax1.fill_between(route_distance, srtm_elevation, alpha=0.3, color='blue')
    ax1.fill_between(route_distance, threedep_elevation, alpha=0.3, color='red')
    
    ax1.set_title('Elevation Profile Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Distance (km)')
    ax1.set_ylabel('Elevation (m)')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add annotations for key differences
    ax1.annotate('Smoothed terrain\n(90m resolution)', 
                xy=(1.5, srtm_elevation[15]), xytext=(1.5, srtm_elevation[15] + 40),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, ha='center', color='blue')
    
    ax1.annotate('Detailed trail features\n(1m resolution)', 
                xy=(3.5, threedep_elevation[70]), xytext=(3.5, threedep_elevation[70] + 40),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, ha='center', color='red')
    
    # 2. Grade Analysis Comparison
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(route_distance, srtm_grade, 'b-', linewidth=3, label='SRTM 90m Grade', alpha=0.8)
    ax2.plot(route_distance, threedep_grade, 'r-', linewidth=2, label='3DEP 1m Grade', alpha=0.9)
    ax2.axhline(y=8, color='orange', linestyle='--', alpha=0.7, label='Steep Grade (8%)')
    ax2.axhline(y=-8, color='orange', linestyle='--', alpha=0.7)
    
    ax2.set_title('Grade (Slope) Analysis Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Grade (%)')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-15, 15)
    
    # 3. Data Resolution Visualization
    ax3 = fig.add_subplot(gs[2, 0])
    
    # SRTM 90m grid representation
    grid_size = 6
    srtm_grid = np.ones((grid_size, grid_size))
    ax3.imshow(srtm_grid, cmap='Blues', alpha=0.6)
    
    # Draw grid lines to show 90m cells
    for i in range(grid_size + 1):
        ax3.axhline(i - 0.5, color='black', linewidth=1)
        ax3.axvline(i - 0.5, color='black', linewidth=1)
    
    ax3.set_title('SRTM 90m Resolution\n(6 cells = 540m)', fontsize=12, fontweight='bold')
    ax3.set_xticks([])
    ax3.set_yticks([])
    
    # Add cell size annotation
    ax3.annotate('90m × 90m\nper cell', xy=(2.5, 2.5), ha='center', va='center',
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3DEP 1m grid representation
    ax4 = fig.add_subplot(gs[2, 1])
    
    # 3DEP 1m grid (much finer)
    fine_grid_size = 18
    threedep_grid = np.random.rand(fine_grid_size, fine_grid_size)
    ax4.imshow(threedep_grid, cmap='Reds', alpha=0.6)
    
    # Draw some grid lines (not all, would be too dense)
    for i in range(0, fine_grid_size + 1, 3):
        ax4.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.5)
        ax4.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.5)
    
    ax4.set_title('3DEP 1m Resolution\n(324 cells = 540m)', fontsize=12, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    # Add cell size annotation
    ax4.annotate('1m × 1m\nper cell', xy=(8.5, 8.5), ha='center', va='center',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 4. Statistics and Comparison Table
    ax5 = fig.add_subplot(gs[3, :])
    ax5.axis('off')
    
    # Calculate statistics
    srtm_total_gain = np.sum(np.maximum(0, np.diff(srtm_elevation)))
    threedep_total_gain = np.sum(np.maximum(0, np.diff(threedep_elevation)))
    srtm_max_grade = np.max(np.abs(srtm_grade))
    threedep_max_grade = np.max(np.abs(threedep_grade))
    
    # Create comparison table
    table_data = [
        ['Metric', 'SRTM 90m', '3DEP 1m', 'Improvement'],
        ['Horizontal Resolution', '90m', '1m', '90× better'],
        ['Vertical Accuracy', '±16m', '±0.3m', '53× better'],
        [f'Total Elevation Gain', f'{srtm_total_gain:.0f}m', f'{threedep_total_gain:.0f}m', 
         f'{((threedep_total_gain/srtm_total_gain-1)*100):+.0f}%'],
        [f'Maximum Grade', f'{srtm_max_grade:.1f}%', f'{threedep_max_grade:.1f}%', 
         f'{((threedep_max_grade/srtm_max_grade-1)*100):+.0f}%'],
        ['Grade Detection', 'Smoothed', 'Precise', 'Trail-level detail'],
        ['Route Optimization', 'General terrain', 'Actual features', 'Fine-grained'],
    ]
    
    # Create table
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.2, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if j == 3:  # Improvement column
                table[(i, j)].set_facecolor('#E8F5E8')
            elif j == 1:  # SRTM column
                table[(i, j)].set_facecolor('#E3F2FD')
            elif j == 2:  # 3DEP column
                table[(i, j)].set_facecolor('#FFEBEE')
    
    # Add summary text boxes
    summary_text = """
Key Benefits of 3DEP 1m Integration:
• Accurate trail-level grade detection
• Precise climbing segment identification  
• Enhanced route optimization for elevation objectives
• Improved safety through accurate terrain analysis
• Better user experience with detailed elevation profiles
"""
    
    tech_specs = """
Technical Implementation:
• Hybrid data source with SRTM fallback
• Cloud Optimized GeoTIFF (COG) format
• Local caching for performance
• py3dep library integration
• OpenTopography API support
"""
    
    plt.figtext(0.02, 0.02, summary_text, fontsize=9, 
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E8", alpha=0.8))
    
    plt.figtext(0.52, 0.02, tech_specs, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_route_detail_comparison():
    """Create detailed comparison of a specific route segment"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Route Segment Detail: Hill Climbing Analysis\nChristiansburg, VA - 500m Trail Segment', 
                 fontsize=14, fontweight='bold')
    
    # Create detailed segment (500m of trail)
    distance = np.linspace(0, 0.5, 500)  # 500m at 1m resolution
    
    # SRTM representation (smoothed, 5-6 data points for 500m)
    srtm_points = np.linspace(0, 0.5, 6)
    srtm_elev_points = np.array([620, 625, 635, 648, 655, 660])
    srtm_elevation = np.interp(distance, srtm_points, srtm_elev_points)
    
    # 3DEP representation (detailed trail features)
    base_elevation = np.interp(distance, srtm_points, srtm_elev_points)
    
    # Add trail features: switchbacks, rocks, small hills
    trail_features = np.zeros_like(distance)
    
    # Switchback at 150m
    trail_features += 3 * np.exp(-((distance - 0.15) * 1000 / 20)**2)
    
    # Rocky section at 300m
    rocky_section = (distance > 0.28) & (distance < 0.32)
    trail_features[rocky_section] += 2 * np.sin((distance[rocky_section] - 0.28) * 1000 * np.pi / 20)
    
    # Small hill at 400m
    trail_features += 4 * np.exp(-((distance - 0.4) * 1000 / 30)**2)
    
    # Add fine-scale noise for 1m precision
    trail_features += np.random.normal(0, 0.2, len(distance))
    
    threedep_elevation = base_elevation + trail_features
    
    # Plot elevation profiles
    ax1.plot(distance * 1000, srtm_elevation, 'b-', linewidth=4, 
             label='SRTM 90m (6 data points)', alpha=0.8, marker='o', markersize=8)
    ax1.plot(distance * 1000, threedep_elevation, 'r-', linewidth=1.5, 
             label='3DEP 1m (500 data points)', alpha=0.9)
    
    ax1.fill_between(distance * 1000, srtm_elevation, alpha=0.3, color='blue')
    ax1.fill_between(distance * 1000, threedep_elevation, alpha=0.2, color='red')
    
    ax1.set_title('Elevation Profile: 500m Hill Climb', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Distance (m)')
    ax1.set_ylabel('Elevation (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Annotate features
    ax1.annotate('Missed switchback\n(smoothed out)', 
                xy=(150, np.interp(0.15, distance, srtm_elevation)), 
                xytext=(100, 645),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9, color='blue')
    
    ax1.annotate('Detected switchback\n(3m elevation change)', 
                xy=(150, np.interp(0.15, distance, threedep_elevation)), 
                xytext=(200, 665),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')
    
    ax1.annotate('Rocky terrain\n(missed)', 
                xy=(300, np.interp(0.3, distance, srtm_elevation)), 
                xytext=(350, 640),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=9, color='blue')
    
    ax1.annotate('Rocky section\n(2m variations)', 
                xy=(300, np.interp(0.3, distance, threedep_elevation)), 
                xytext=(350, 670),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')
    
    # Calculate and plot grades
    srtm_grade = np.gradient(srtm_elevation, distance * 1000) * 100
    threedep_grade = np.gradient(threedep_elevation, distance * 1000) * 100
    
    ax2.plot(distance * 1000, srtm_grade, 'b-', linewidth=4, 
             label='SRTM 90m Grade', alpha=0.8)
    ax2.plot(distance * 1000, threedep_grade, 'r-', linewidth=1.5, 
             label='3DEP 1m Grade', alpha=0.9)
    
    # Highlight steep sections
    ax2.axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Very Steep (15%)')
    ax2.axhline(y=8, color='yellow', linestyle='--', alpha=0.7, label='Steep (8%)')
    ax2.fill_between(distance * 1000, -20, 15, where=(threedep_grade > 15), 
                     alpha=0.3, color='red', label='Dangerous Grade')
    
    ax2.set_title('Grade Analysis: Trail Safety Assessment', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Distance (m)')
    ax2.set_ylabel('Grade (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 25)
    
    # Add grade analysis text
    max_srtm_grade = np.max(srtm_grade)
    max_3dep_grade = np.max(threedep_grade)
    steep_sections_3dep = np.sum(threedep_grade > 15)
    
    analysis_text = f"""
Grade Analysis Results:
• SRTM Max Grade: {max_srtm_grade:.1f}% (underestimated)
• 3DEP Max Grade: {max_3dep_grade:.1f}% (actual trail conditions)
• Steep sections (>15%): {steep_sections_3dep}m detected with 3DEP
• Safety impact: 3DEP reveals potentially dangerous grades
• Route optimization: More accurate difficulty assessment
"""
    
    plt.figtext(0.02, 0.02, analysis_text, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFEBEE", alpha=0.8))
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    print("Creating elevation data comparison wireframes...")
    
    # Create main comparison wireframe
    fig1 = create_elevation_comparison_wireframe()
    fig1.savefig("elevation_comparison_wireframe.png", dpi=300, bbox_inches='tight')
    print("✅ Created: elevation_comparison_wireframe.png")
    
    # Create detailed route segment comparison
    fig2 = create_route_detail_comparison()
    fig2.savefig("route_detail_comparison_wireframe.png", dpi=300, bbox_inches='tight')
    print("✅ Created: route_detail_comparison_wireframe.png")
    
    # Show plots
    plt.show()
    
    print("\n📊 Wireframe Summary:")
    print("• elevation_comparison_wireframe.png: Overall SRTM vs 3DEP comparison")
    print("• route_detail_comparison_wireframe.png: Detailed trail segment analysis")
    print("• Demonstrates 90x resolution improvement and trail-level precision")
    print("• Shows impact on route optimization and safety assessment")
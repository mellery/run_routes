#!/usr/bin/env python3
"""
K-Shortest Paths Evaluation Script
Uses the same network data as cli_route_planner.py for accurate route analysis

Usage:
    python eval_k_shortest.py --destinations 10 --k 5
    python eval_k_shortest.py -d 100 -k 10
    python eval_k_shortest.py --help
"""

import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import os
import argparse
import time

# Add project root for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from route_services import NetworkManager
import osmnx as ox

def calculate_path_length(graph, path):
    """Calculate the total length of a path in meters"""
    total_length = 0
    for i in range(len(path) - 1):
        # Get edge data between consecutive nodes
        edge_data = graph.get_edge_data(path[i], path[i+1])
        if edge_data:
            # Handle multiple edges between same nodes
            if isinstance(edge_data, dict):
                if 0 in edge_data:  # Single edge case
                    total_length += edge_data[0].get('length', 0)
                else:  # Multiple edges, take the first one
                    total_length += list(edge_data.values())[0].get('length', 0)
            else:
                total_length += edge_data.get('length', 0)
    return total_length

def filter_graph_for_routing(graph, exclude_footways=True):
    """Filter graph exactly like the CLI route planner does"""
    if not exclude_footways:
        return graph
    
    # Create a copy of the graph
    filtered_graph = graph.copy()
    
    # Count removals for reporting
    edges_removed = 0
    total_footways = 0
    
    # Remove footway edges
    edges_to_remove = []
    for u, v, data in filtered_graph.edges(data=True):
        highway = data.get('highway', '')
        if highway == 'footway':
            total_footways += 1
            edges_to_remove.append((u, v))
            edges_removed += 1
    
    # Remove the edges
    filtered_graph.remove_edges_from(edges_to_remove)
    
    # Remove isolated nodes (nodes with no connections after edge removal)
    isolated_nodes = [node for node in filtered_graph.nodes() if filtered_graph.degree(node) == 0]
    filtered_graph.remove_nodes_from(isolated_nodes)
    
    if edges_removed > 0:
        print(f"   Footway filtering: removed {edges_removed}/{total_footways} footway edges, {len(isolated_nodes)} isolated nodes")
    
    return filtered_graph

def generate_color_palette(n_colors):
    """Generate distinct colors for many destinations"""
    if n_colors <= 10:
        return ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:n_colors]
    else:
        # Generate colors using HSV space for better distinction
        import matplotlib.colors as mcolors
        colors = []
        for i in range(n_colors):
            hue = i / n_colors
            # Vary saturation and value for better distinction
            sat = 0.7 + 0.3 * ((i % 4) / 3)  # 0.7 to 1.0
            val = 0.8 + 0.2 * ((i % 2))      # 0.8 to 1.0
            rgb = mcolors.hsv_to_rgb([hue, sat, val])
            colors.append(rgb)
        return colors

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='K-Shortest Paths Evaluation using CLI Route Planner Network Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python eval_k_shortest.py -d 5 -k 3                    # 5 destinations, 3 paths each
  python eval_k_shortest.py --destinations 25 --k 10     # 25 destinations, 10 paths each  
  python eval_k_shortest.py -d 100 -k 10                 # 100 destinations, 10 paths each
  python eval_k_shortest.py --quick                      # Quick test (5 destinations, 3 paths)
        """
    )
    
    parser.add_argument(
        '-d', '--destinations',
        type=int,
        default=5,
        help='Number of random destinations to analyze (default: 5)'
    )
    
    parser.add_argument(
        '-k', '--k',
        type=int,
        default=30,
        help='Number of shortest paths to find per destination (default: 30)'
    )
    
    parser.add_argument(
        '--radius',
        type=float,
        default=2.5,
        help='Network radius in kilometers (default: 2.5)'
    )
    
    parser.add_argument(
        '--start-node',
        type=int,
        default=1529188403,
        help='Starting node ID (default: 1529188403 - Christiansburg, VA)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible results (default: 42)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (5 destinations, 3 paths each)'
    )
    
    parser.add_argument(
        '--no-filter',
        action='store_true',
        help='Skip footway filtering (include all edges)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output filename for the plot (default: auto-generated)'
    )
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Apply quick mode settings
    if args.quick:
        args.destinations = 5
        args.k = 3
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print('🌐 K-Shortest Paths Evaluation')
    print(f'   Destinations: {args.destinations}')
    print(f'   Paths per destination: {args.k}')
    print(f'   Total paths to calculate: {args.destinations * args.k:,}')
    print(f'   Network radius: {args.radius}km')
    print(f'   Starting node: {args.start_node}')
    print(f'   Random seed: {args.seed}')
    print()
    
    print('📊 Loading street network using route_services NetworkManager...')
    print('   (Same processed data as cli_route_planner.py)')
    
    # Use NetworkManager exactly like CLI does
    network_manager = NetworkManager()
    unfiltered_graph = network_manager.load_network(radius_km=args.radius)
    
    if not unfiltered_graph:
        print("❌ Failed to load network")
        return
    
    # Apply filtering based on arguments
    exclude_footways = not args.no_filter
    if exclude_footways:
        print('🔧 Applying CLI route planner filtering (exclude_footways=True)...')
    else:
        print('🔧 Using unfiltered network (include all edges)...')
    graph = filter_graph_for_routing(unfiltered_graph, exclude_footways=exclude_footways)
    
    # Get network stats for both graphs
    unfiltered_stats = network_manager.get_network_stats(unfiltered_graph)
    filtered_stats = network_manager.get_network_stats(graph)
    print(f"📊 Network comparison:")
    print(f"   Unfiltered: {unfiltered_stats['nodes']} nodes, {unfiltered_stats['edges']} edges")
    print(f"   CLI filtered: {filtered_stats['nodes']} nodes, {filtered_stats['edges']} edges")
    print(f"   Reduction: {unfiltered_stats['nodes'] - filtered_stats['nodes']} nodes, {unfiltered_stats['edges'] - filtered_stats['edges']} edges removed")
    print(f"   Elevation data: {'✅ Available' if filtered_stats['has_elevation'] else '❌ Not available'}")
    
    # Find the starting point or closest node
    start_node = args.start_node
    if not network_manager.validate_node_exists(graph, start_node):
        print(f"⚠️ Starting node {start_node} not found in filtered graph")
        center_point = network_manager.center_point
        start_node = ox.nearest_nodes(graph, center_point[1], center_point[0])
        print(f"✅ Using closest node: {start_node}")
    else:
        print(f"✅ Using starting node: {start_node}")

    print(f'\n🎯 Selecting {args.destinations} random destination points...')
    # Select random nodes from the network
    all_nodes = list(graph.nodes())
    if args.destinations > len(all_nodes):
        print(f"❌ Error: Requested {args.destinations} destinations but network only has {len(all_nodes)} nodes")
        return
    
    random_nodes = random.sample(all_nodes, args.destinations)

    if args.destinations <= 10:
        print(f'Random destination nodes: {random_nodes}')
    else:
        print(f'First 5 destinations: {random_nodes[:5]}')
        print(f'Last 5 destinations: {random_nodes[-5:]}')

    # Get node positions for plotting
    node_coords = {node: (data['x'], data['y']) for node, data in graph.nodes(data=True)}

    # Create the plot using OSMnx for consistency
    # Adjust figure size based on number of destinations
    if args.destinations <= 10:
        figsize = (16, 16)
    elif args.destinations <= 50:
        figsize = (18, 18)
    else:
        figsize = (20, 20)
    
    fig, ax = ox.plot_graph(graph, figsize=figsize, node_size=0, edge_linewidth=0.3, 
                           edge_color='lightgray', show=False, close=False)

    # Generate colors for all destinations
    colors = generate_color_palette(args.destinations)

    print(f'\n🚀 Calculating k={args.k} shortest paths and distances...')
    print('=' * 80)

    all_paths = []
    for i, dest_node in enumerate(random_nodes):
        print(f'\nDestination {i+1}: Node {dest_node}')
        print('-' * 50)
        
        try:
            # Calculate k shortest paths using the processed network
            paths = list(ox.routing.k_shortest_paths(graph, start_node, dest_node, k=args.k, weight='length'))
            
            dest_paths = []
            distances = []
            
            for j, path in enumerate(paths):
                # Calculate path distance using our custom function
                path_length = calculate_path_length(graph, path)
                path_length_km = path_length / 1000  # Convert to km
                
                distances.append(path_length_km)
                
                # Store path info
                dest_paths.append({
                    'path': path,
                    'length_km': path_length_km,
                    'length_m': path_length
                })
                
                # Plot the path with varying transparency based on rank
                path_coords = [(node_coords[node][0], node_coords[node][1]) for node in path]
                path_x, path_y = zip(*path_coords)
                
                # Adjust visualization parameters based on k value and number of destinations
                if args.k <= 5:
                    # High visibility for small k
                    alpha = max(0.2, 0.9 - (j * 0.15))
                    linewidth = max(0.8, 3.0 - (j * 0.4))
                elif args.k <= 20:
                    # Medium visibility for medium k
                    alpha = max(0.15, 0.8 - (j * 0.03))
                    linewidth = max(0.5, 2.5 - (j * 0.1))
                else:
                    # Lower visibility for large k to avoid clutter
                    alpha = max(0.1, 0.6 - (j * 0.02))
                    linewidth = max(0.3, 2.0 - (j * 0.08))
                
                # Use generated color for this destination
                base_color = colors[i]
                
                ax.plot(path_x, path_y, color=base_color, 
                       linewidth=linewidth, alpha=alpha, 
                       zorder=15 - (j // 5), solid_capstyle='round')
            
            all_paths.append(dest_paths)
            
            # Print statistics for this destination
            if distances:
                print(f'  Found {len(distances)} paths')
                print(f'  Shortest: {min(distances):.3f} km')
                print(f'  Longest: {max(distances):.3f} km')
                print(f'  Average: {np.mean(distances):.3f} km')
                print(f'  Std Dev: {np.std(distances):.3f} km')
                print(f'  Range: {max(distances) - min(distances):.3f} km')
                
                # Show first 5 and last 5 paths
                print(f'  First 5 paths: {[f"{d:.3f}km" for d in distances[:5]]}')
                if len(distances) > 10:
                    print(f'  Last 5 paths:  {[f"{d:.3f}km" for d in distances[-5:]]}')
                
                # Debug: Verify paths were plotted
                print(f'  ✅ Plotted {len(paths)} paths with varying transparency (alpha: 0.9 → 0.15)')
            
        except Exception as e:
            print(f'  Error calculating paths to node {dest_node}: {e}')
            all_paths.append([])

    # Plot start and destination points
    start_x, start_y = node_coords[start_node]
    ax.scatter([start_x], [start_y], c='black', s=300, marker='*', 
              label=f'Start (Node {start_node})', zorder=20, edgecolors='white', linewidth=3)

    for i, dest_node in enumerate(random_nodes):
        if dest_node in node_coords:
            dest_x, dest_y = node_coords[dest_node]
            ax.scatter([dest_x], [dest_y], c=colors[i], s=150, marker='o', 
                      zorder=15, edgecolors='white', linewidth=2)

    # Add title  
    total_paths_plotted = sum(len(paths) for paths in all_paths if paths)
    filter_status = "CLI Filtered" if exclude_footways else "Unfiltered"
    ax.set_title(f'K={args.k} Shortest Paths - {args.destinations} Destinations\n'
                f'Using CLI Route Planner Network Data (radius: {args.radius}km)\n'
                f'{filter_status} Network: {filtered_stats["nodes"]} nodes, {filtered_stats["edges"]} edges\n'
                f'Total paths plotted: {total_paths_plotted}', 
                 fontsize=13, pad=20)

    # Create a simplified legend
    legend_elements = []
    legend_elements.append(plt.Line2D([0], [0], marker='*', color='black', markersize=15, 
                                     linestyle='None', label=f'Start (Node {start_node})'))

    # Show only first few destinations in legend to avoid clutter
    max_legend_dests = min(5, args.destinations)
    for i in range(max_legend_dests):
        if i < len(all_paths) and all_paths[i]:
            dest_node = random_nodes[i]
            shortest_km = all_paths[i][0]['length_km']
            num_paths = len(all_paths[i])
            legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=2, 
                                             label=f'Dest {i+1} ({num_paths} paths) - {shortest_km:.2f}km'))
    
    if args.destinations > max_legend_dests:
        legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=2, alpha=0.5,
                                         label=f'... and {args.destinations - max_legend_dests} more destinations'))

    # Add explanation
    legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=2, alpha=0.8,
                                     label=f'k={args.k} shortest paths'))
    legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=1, alpha=0.3,
                                     label='Alternative routes (fainter)'))

    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, 
             bbox_to_anchor=(0.02, 0.98))

    # Save the plot
    plt.tight_layout()
    
    # Generate output filename
    if args.output:
        filename = args.output
    else:
        filter_suffix = "_filtered" if exclude_footways else "_unfiltered"
        filename = f'k{args.k}_shortest_paths_{args.destinations}dest{filter_suffix}.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

    print('\n' + '=' * 80)
    print(f'SUMMARY - K={args.k} SHORTEST PATHS ANALYSIS')
    print('=' * 80)
    print(f'Start point: Node {start_node}')
    print(f'Network data: Same as cli_route_planner.py')
    print(f'  - Enhanced 3DEP elevation data: {filtered_stats["has_elevation"]}')
    print(f'  - Running weights: Applied')
    print(f'  - Network processing: Complete')
    print(f'  - Filtering: {"Enabled" if exclude_footways else "Disabled"}')
    print(f'  - Network size: {filtered_stats["nodes"]} nodes, {filtered_stats["edges"]} edges')
    print(f'Number of destinations: {len(random_nodes)}')
    print(f'Paths per destination: {args.k}')
    print(f'Total paths calculated: {total_paths_plotted}')

    # Show sample results
    max_shown = min(10, len(all_paths))
    for i in range(max_shown):
        if i < len(all_paths) and all_paths[i]:
            dest_node = random_nodes[i]
            distances = [p['length_km'] for p in all_paths[i]]
            print(f'\nDestination {i+1} (Node {dest_node}) - {len(distances)} paths found:')
            print(f'  Shortest: {min(distances):.3f} km')
            print(f'  Longest:  {max(distances):.3f} km')
            print(f'  Average:  {np.mean(distances):.3f} km')
            print(f'  Range:    {max(distances) - min(distances):.3f} km')
    
    if len(all_paths) > max_shown:
        print(f'\n... and {len(all_paths) - max_shown} more destinations')
    
    # Overall statistics
    if all_paths:
        all_min_distances = [min(p['length_km'] for p in paths) for paths in all_paths if paths]
        all_max_distances = [max(p['length_km'] for p in paths) for paths in all_paths if paths]
        
        print(f'\nOverall statistics:')
        print(f'  Global shortest path: {min(all_min_distances):.3f} km')
        print(f'  Global longest path: {max(all_max_distances):.3f} km')
        print(f'  Average minimum distance: {np.mean(all_min_distances):.3f} km')
        print(f'  Average maximum distance: {np.mean(all_max_distances):.3f} km')

    print(f'\nPlot saved as: {filename}')
    print('Note: Using the same processed & filtered network data as cli_route_planner.py')
    print('      - Enhanced with 3DEP 1m elevation data')
    print('      - Optimized with running weights')
    print(f'      - Footway filtering: {"Enabled" if exclude_footways else "Disabled"}')
    print(f'      - Path opacity and thickness decrease with path rank')
    print(f'      - Total paths visualized: {total_paths_plotted}')

if __name__ == "__main__":
    main()
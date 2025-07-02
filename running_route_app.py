#!/usr/bin/env python3
"""
Running Route Optimizer - Interactive Web Application
Streamlit-based UI for generating optimized running routes with elevation data
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import osmnx as ox
import numpy as np
import time
from route import add_elevation_to_graph, add_elevation_to_edges, add_running_weights
from tsp_solver import RunningRouteOptimizer, RouteObjective
import networkx as nx

# Configure page
st.set_page_config(
    page_title="Running Route Optimizer",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'graph' not in st.session_state:
    st.session_state.graph = None
if 'current_route' not in st.session_state:
    st.session_state.current_route = None
if 'route_result' not in st.session_state:
    st.session_state.route_result = None

def load_street_network():
    """Load and cache street network with elevation data"""
    if st.session_state.graph is None:
        with st.spinner("Loading street network and elevation data..."):
            try:
                # Use cached graph loader
                from graph_cache import load_or_generate_graph
                
                center_point = (37.1299, -80.4094)
                graph = load_or_generate_graph(
                    center_point=center_point,
                    radius_m=800,  # Use cached 800m network
                    network_type='all'
                )
                
                if graph:
                    st.session_state.graph = graph
                    st.success(f"Loaded {len(graph.nodes)} intersections and {len(graph.edges)} road segments")
                else:
                    st.error("Failed to load street network")
                    return None
                    
            except Exception as e:
                st.error(f"Failed to load street network: {e}")
                return None
    
    return st.session_state.graph

def create_base_map(graph):
    """Create base Folium map with all intersections"""
    # Get center coordinates
    center_lat = np.mean([data['y'] for _, data in graph.nodes(data=True)])
    center_lon = np.mean([data['x'] for _, data in graph.nodes(data=True)])
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles='OpenStreetMap'
    )
    
    # Add intersection markers
    for node_id, data in list(graph.nodes(data=True))[:200]:  # Limit for performance
        elevation = data.get('elevation', 0)
        
        # Color based on elevation
        if elevation < 630:
            color = 'blue'
        elif elevation < 650:
            color = 'green'
        elif elevation < 670:
            color = 'orange'
        else:
            color = 'red'
        
        folium.CircleMarker(
            location=[data['y'], data['x']],
            radius=3,
            popup=f"Node: {node_id}<br>Elevation: {elevation:.0f}m",
            color=color,
            fill=True,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def get_nearby_nodes(graph, lat, lon, radius_km=0.5):
    """Get nodes near a clicked location"""
    from route import haversine_distance
    
    nearby = []
    for node_id, data in graph.nodes(data=True):
        distance = haversine_distance(lat, lon, data['y'], data['x'])
        if distance <= radius_km * 1000:  # Convert to meters
            nearby.append((node_id, distance, data))
    
    # Sort by distance
    nearby.sort(key=lambda x: x[1])
    return nearby[:10]  # Return closest 10

def create_route_map(graph, route_result):
    """Create map showing the optimized route"""
    if not route_result or not route_result.get('route'):
        return None
    
    route = route_result['route']
    
    # Get center coordinates
    center_lat = np.mean([data['y'] for _, data in graph.nodes(data=True)])
    center_lon = np.mean([data['x'] for _, data in graph.nodes(data=True)])
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Get route coordinates
    route_coords = []
    route_elevations = []
    
    for i, node in enumerate(route):
        if node in graph.nodes:
            node_data = graph.nodes[node]
            coord = [node_data['y'], node_data['x']]
            route_coords.append(coord)
            route_elevations.append(node_data.get('elevation', 0))
            
            # Add markers for route points
            color = 'green' if i == 0 else 'red' if i == len(route) - 1 else 'blue'
            icon = 'play' if i == 0 else 'stop' if i == len(route) - 1 else 'record'
            
            folium.Marker(
                location=coord,
                popup=f"{'Start' if i == 0 else 'Finish' if i == len(route) - 1 else f'Point {i+1}'}<br>"
                      f"Elevation: {route_elevations[-1]:.0f}m",
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)
    
    # Add return to start
    if route_coords:
        route_coords.append(route_coords[0])
    
    # Draw route path
    if len(route_coords) > 1:
        folium.PolyLine(
            locations=route_coords,
            weight=4,
            color='red',
            opacity=0.8,
            popup="Running Route"
        ).add_to(m)
    
    return m

def create_elevation_profile(graph, route_result):
    """Create elevation profile chart"""
    if not route_result or not route_result.get('route'):
        return None
    
    route = route_result['route']
    stats = route_result.get('stats', {})
    
    # Get elevation data for route
    elevations = []
    distances = [0]  # Start at 0
    cumulative_distance = 0
    
    for i, node in enumerate(route):
        if node in graph.nodes:
            elevations.append(graph.nodes[node].get('elevation', 0))
            
            if i > 0:
                # Calculate distance from previous node
                prev_node = route[i-1]
                if prev_node in graph.nodes:
                    from route import haversine_distance
                    prev_data = graph.nodes[prev_node]
                    curr_data = graph.nodes[node]
                    segment_dist = haversine_distance(
                        prev_data['y'], prev_data['x'],
                        curr_data['y'], curr_data['x']
                    )
                    cumulative_distance += segment_dist
                    distances.append(cumulative_distance)
    
    # Add return to start
    if len(route) > 1 and route[0] in graph.nodes and route[-1] in graph.nodes:
        from route import haversine_distance
        start_data = graph.nodes[route[0]]
        end_data = graph.nodes[route[-1]]
        return_dist = haversine_distance(
            end_data['y'], end_data['x'],
            start_data['y'], start_data['x']
        )
        cumulative_distance += return_dist
        distances.append(cumulative_distance)
        elevations.append(elevations[0])  # Back to start elevation
    
    # Convert distances to km
    distances_km = [d / 1000 for d in distances]
    
    # Create plotly figure
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=distances_km,
        y=elevations,
        mode='lines+markers',
        name='Elevation Profile',
        line=dict(color='green', width=3),
        marker=dict(size=6),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title=f"Elevation Profile - {stats.get('total_distance_km', 0):.2f}km Route",
        xaxis_title="Distance (km)",
        yaxis_title="Elevation (m)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def generate_directions(graph, route_result):
    """Generate turn-by-turn directions"""
    if not route_result or not route_result.get('route'):
        return []
    
    route = route_result['route']
    directions = []
    
    # Start instruction
    if route and route[0] in graph.nodes:
        start_data = graph.nodes[route[0]]
        directions.append({
            'step': 1,
            'instruction': f"Start at intersection (Node {route[0]})",
            'elevation': f"{start_data.get('elevation', 0):.0f}m",
            'distance': "0.0 km"
        })
    
    # Route segments
    cumulative_distance = 0
    for i in range(1, len(route)):
        if route[i] in graph.nodes and route[i-1] in graph.nodes:
            curr_data = graph.nodes[route[i]]
            prev_data = graph.nodes[route[i-1]]
            
            # Calculate segment distance
            from route import haversine_distance
            segment_dist = haversine_distance(
                prev_data['y'], prev_data['x'],
                curr_data['y'], curr_data['x']
            )
            cumulative_distance += segment_dist
            
            # Direction instruction
            elevation_change = curr_data.get('elevation', 0) - prev_data.get('elevation', 0)
            if elevation_change > 5:
                terrain = "uphill"
            elif elevation_change < -5:
                terrain = "downhill"
            else:
                terrain = "level"
            
            directions.append({
                'step': i + 1,
                'instruction': f"Continue to intersection (Node {route[i]}) - {terrain}",
                'elevation': f"{curr_data.get('elevation', 0):.0f}m ({elevation_change:+.0f}m)",
                'distance': f"{cumulative_distance/1000:.2f} km"
            })
    
    # Return to start
    if len(route) > 1:
        directions.append({
            'step': len(route) + 1,
            'instruction': "Return to starting point to complete the loop",
            'elevation': f"{directions[0]['elevation']}",
            'distance': f"{route_result.get('stats', {}).get('total_distance_km', 0):.2f} km"
        })
    
    return directions

def main():
    """Main Streamlit app"""
    
    # Header
    st.title("🏃 Running Route Optimizer")
    st.markdown("Generate optimized running routes with elevation data for Christiansburg, VA")
    
    # Load network
    graph = load_street_network()
    
    # Sidebar controls
    st.sidebar.header("Route Parameters")
    
    # Target distance
    target_distance = st.sidebar.slider(
        "Target Distance (km)",
        min_value=0.5,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Desired route distance (±20% tolerance)"
    )
    
    # Route objective
    objective_options = {
        "Shortest Route": RouteObjective.MINIMIZE_DISTANCE,
        "Maximum Elevation Gain": RouteObjective.MAXIMIZE_ELEVATION,
        "Balanced Route": RouteObjective.BALANCED_ROUTE,
        "Easiest Route": RouteObjective.MINIMIZE_DIFFICULTY
    }
    
    selected_objective = st.sidebar.selectbox(
        "Route Objective",
        options=list(objective_options.keys()),
        index=0,
        help="Choose optimization strategy"
    )
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        options=["nearest_neighbor", "genetic"],
        index=0,
        help="Optimization algorithm (genetic is slower but better)"
    )
    
    # Difficulty level info
    st.sidebar.markdown("### Difficulty Levels")
    st.sidebar.markdown("""
    - **Shortest**: Focus on distance only
    - **Max Elevation**: Seek hills and climbs  
    - **Balanced**: Good mix of distance and elevation
    - **Easiest**: Avoid steep grades
    """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Select Starting Point")
        st.markdown("Click on the map to choose your starting intersection:")
        
        # Create and display map
        base_map = create_base_map(graph)
        map_data = st_folium(base_map, width=700, height=400)
        
        # Check for map click
        start_node = None
        if map_data['last_clicked']:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            
            # Find nearby nodes
            nearby_nodes = get_nearby_nodes(graph, clicked_lat, clicked_lon)
            
            if nearby_nodes:
                start_node = nearby_nodes[0][0]  # Closest node
                st.success(f"Selected starting point: Node {start_node}")
                st.info(f"Elevation: {graph.nodes[start_node].get('elevation', 0):.0f}m")
        
        # Generate route button
        if st.button("🚀 Generate Optimized Route", type="primary", disabled=start_node is None):
            if start_node:
                with st.spinner("Optimizing route..."):
                    optimizer = RunningRouteOptimizer(graph)
                    
                    try:
                        result = optimizer.find_optimal_route(
                            start_node=start_node,
                            target_distance_km=target_distance,
                            objective=objective_options[selected_objective],
                            algorithm=algorithm
                        )
                        
                        st.session_state.route_result = result
                        st.session_state.current_route = result['route']
                        
                        st.success("Route optimized successfully!")
                        
                    except Exception as e:
                        st.error(f"Route optimization failed: {e}")
    
    with col2:
        st.subheader("📊 Route Statistics")
        
        if st.session_state.route_result:
            result = st.session_state.route_result
            stats = result['stats']
            
            # Key metrics
            st.metric("Distance", f"{stats.get('total_distance_km', 0):.2f} km")
            st.metric("Elevation Gain", f"{stats.get('total_elevation_gain_m', 0):.0f} m")
            st.metric("Max Grade", f"{stats.get('max_grade_percent', 0):.1f}%")
            st.metric("Est. Time", f"{stats.get('estimated_time_min', 0):.0f} min")
            
            # Objective info
            st.info(f"**Objective:** {selected_objective}")
            st.info(f"**Algorithm:** {algorithm.title()}")
            st.info(f"**Solve Time:** {result.get('solve_time', 0):.2f}s")
            
        else:
            st.info("Select a starting point and generate a route to see statistics")
    
    # Route visualization and details
    if st.session_state.route_result:
        st.subheader("🗺️ Optimized Route")
        
        # Route map
        route_map = create_route_map(graph, st.session_state.route_result)
        if route_map:
            st_folium(route_map, width=700, height=400)
        
        # Elevation profile
        st.subheader("📈 Elevation Profile")
        elevation_fig = create_elevation_profile(graph, st.session_state.route_result)
        if elevation_fig:
            st.plotly_chart(elevation_fig, use_container_width=True)
        
        # Turn-by-turn directions
        st.subheader("📋 Turn-by-Turn Directions")
        directions = generate_directions(graph, st.session_state.route_result)
        
        if directions:
            directions_df = pd.DataFrame(directions)
            st.dataframe(
                directions_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "step": "Step",
                    "instruction": "Direction",
                    "elevation": "Elevation",
                    "distance": "Cumulative Distance"
                }
            )

if __name__ == "__main__":
    main()
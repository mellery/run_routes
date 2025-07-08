#!/usr/bin/env python3
"""
Refactored Running Route Optimizer - Streamlit Web Application
Uses shared route services for consistent functionality with CLI
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import time

from route_services import (
    NetworkManager, RouteOptimizer, RouteAnalyzer, 
    ElevationProfiler, RouteFormatter
)

# Configure page
st.set_page_config(
    page_title="Running Route Optimizer (Refactored)",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add refactored banner
st.markdown("""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
    ✨ <strong>Refactored Version</strong> - Now using shared route services for consistency with CLI
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_route_services():
    """Initialize and cache route services"""
    try:
        with st.spinner("Initializing route planning services..."):
            # Create network manager and load graph
            network_manager = NetworkManager()
            graph = network_manager.load_network(radius_km=2.5)  # Larger default for web interface
            
            if not graph:
                st.error("Failed to load street network")
                return None
            
            # Create all services
            services = {
                'network_manager': network_manager,
                'route_optimizer': RouteOptimizer(graph),
                'route_analyzer': RouteAnalyzer(graph),
                'elevation_profiler': ElevationProfiler(graph),
                'route_formatter': RouteFormatter(),
                'graph': graph
            }
            
            # Show success message
            stats = network_manager.get_network_stats(graph)
            st.success(f"✅ Loaded {stats['nodes']} intersections and {stats['edges']} road segments")
            
            return services
            
    except Exception as e:
        st.error(f"Failed to initialize services: {e}")
        return None

def create_base_map(services):
    """Create base map with street network nodes"""
    if not services:
        return None
    
    network_manager = services['network_manager']
    graph = services['graph']
    
    # Get center coordinates
    center_lat, center_lon = network_manager.center_point
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Add sample of nodes to map (to avoid overcrowding)
    node_sample = list(graph.nodes(data=True))[:50]  # Show first 50 nodes
    
    for node_id, data in node_sample:
        lat, lon = data['y'], data['x']
        elevation = data.get('elevation', 0)
        
        # Color code by elevation
        if elevation > 625:
            color = 'red'
        elif elevation > 615:
            color = 'orange'
        elif elevation > 605:
            color = 'yellow'
        else:
            color = 'green'
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            popup=f"Node: {node_id}<br>Elevation: {elevation:.0f}m",
            color=color,
            fill=True,
            fillOpacity=0.7
        ).add_to(m)
    
    return m

def create_route_map(services, route_result):
    """Create map showing the optimized route following actual roads"""
    if not services or not route_result or not route_result.get('route'):
        return None
    
    network_manager = services['network_manager']
    elevation_profiler = services['elevation_profiler']
    
    # Get center coordinates
    center_lat, center_lon = network_manager.center_point
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles='OpenStreetMap'
    )
    
    # Get detailed route path (follows actual roads)
    detailed_path = elevation_profiler.get_detailed_route_path(route_result)
    
    if detailed_path:
        # Convert to coordinates for folium
        route_coords = [[p['latitude'], p['longitude']] for p in detailed_path]
        
        # Add route path following roads
        folium.PolyLine(
            locations=route_coords,
            color='red',
            weight=4,
            opacity=0.8,
            popup=f"Running Route - {route_result['stats']['total_distance_km']:.2f}km"
        ).add_to(m)
        
        # Add start/finish marker
        start_point = detailed_path[0]
        folium.Marker(
            location=[start_point['latitude'], start_point['longitude']],
            popup=f"Start/Finish<br>Node: {start_point['node_id']}<br>Elevation: {start_point['elevation']:.0f}m",
            icon=folium.Icon(color='green', icon='play')
        ).add_to(m)
        
        # Add key intersection markers (every 20th intersection to avoid clutter)
        intersections = [p for p in detailed_path if p.get('node_type') == 'intersection']
        original_route = route_result['route']
        
        for i, intersection in enumerate(intersections):
            # Only show original route waypoints + a few intermediate ones
            if intersection['node_id'] in original_route or i % 20 == 0:
                if intersection['node_id'] != start_point['node_id']:  # Skip start point
                    marker_color = 'blue' if intersection['node_id'] in original_route else 'lightblue'
                    marker_size = 6 if intersection['node_id'] in original_route else 3
                    
                    folium.CircleMarker(
                        location=[intersection['latitude'], intersection['longitude']],
                        radius=marker_size,
                        popup=f"{'Key Waypoint' if intersection['node_id'] in original_route else 'Intersection'}<br>Node: {intersection['node_id']}<br>Elevation: {intersection['elevation']:.0f}m",
                        color=marker_color,
                        fillColor=marker_color,
                        fillOpacity=0.7
                    ).add_to(m)
        
        # Add route statistics
        stats = route_result.get('stats', {})
        folium.Marker(
            location=[center_lat + 0.002, center_lon + 0.002],  # Offset for visibility
            popup=f"""<b>Route Statistics</b><br>
            Distance: {stats.get('total_distance_km', 0):.2f} km<br>
            Elevation Gain: {stats.get('total_elevation_gain_m', 0):.0f} m<br>
            Time Estimate: {stats.get('estimated_time_min', 0):.0f} min<br>
            Total Path Nodes: {len(detailed_path)}<br>
            Key Intersections: {len(original_route)}""",
            icon=folium.Icon(color='gray', icon='info-sign')
        ).add_to(m)
    
    return m

def create_elevation_plot(services, route_result):
    """Create elevation profile plot using shared services"""
    if not services or not route_result:
        return None
    
    elevation_profiler = services['elevation_profiler']
    
    # Generate profile data
    profile_data = elevation_profiler.generate_profile_data(route_result)
    
    if not profile_data or not profile_data.get('elevations'):
        return None
    
    # Create plotly figure
    fig = go.Figure()
    
    distances_km = profile_data['distances_km']
    elevations = profile_data['elevations']
    
    fig.add_trace(go.Scatter(
        x=distances_km,
        y=elevations,
        mode='lines+markers',
        name='Elevation Profile',
        line=dict(color='green', width=3),
        marker=dict(size=6),
        fill='tonexty'
    ))
    
    stats = profile_data.get('elevation_stats', {})
    total_distance = profile_data.get('total_distance_km', 0)
    
    # Set Y-axis to start from lowest elevation with padding
    if elevations:
        min_elev = min(elevations)
        max_elev = max(elevations)
        elev_range = max_elev - min_elev
        padding = max(5, elev_range * 0.1)  # 10% padding or 5m minimum
        y_range = [min_elev - padding, max_elev + padding]
    else:
        y_range = None
    
    fig.update_layout(
        title=f"Elevation Profile - {total_distance:.2f}km Route",
        xaxis_title="Distance (km)",
        yaxis_title="Elevation (m)",
        yaxis=dict(range=y_range) if y_range else {},
        hovermode='x unified',
        height=400
    )
    
    return fig

def main():
    """Main Streamlit app using refactored services"""
    
    # Header
    st.title("🏃 Running Route Optimizer")
    st.markdown("Generate optimized running routes with elevation data for Christiansburg, VA")
    
    # Initialize services
    services = initialize_route_services()
    
    if not services:
        st.error("❌ Failed to initialize route services. Please refresh the page.")
        return
    
    # Get service instances
    network_manager = services['network_manager']
    route_optimizer = services['route_optimizer']
    route_analyzer = services['route_analyzer']
    elevation_profiler = services['elevation_profiler']
    route_formatter = services['route_formatter']
    graph = services['graph']
    
    # Sidebar controls
    st.sidebar.header("Route Parameters")
    
    # Show solver information
    solver_info = route_optimizer.get_solver_info()
    st.sidebar.success(f"✅ Using {solver_info['solver_type']} GA solver")
    
    # Target distance
    target_distance = st.sidebar.slider(
        "Target Distance (km)",
        min_value=0.5,
        max_value=25.0,
        value=5.0,
        step=0.1,
        help="Desired route distance (±20% tolerance). Routes >8km will automatically expand network coverage."
    )
    
    # Check if we need larger network for this distance
    if target_distance > 8.0 and graph and len(graph.nodes) < 1000:  # Small network indicator
        st.sidebar.warning(f"⚠️ Large route ({target_distance}km) may be limited by current network size. Consider using CLI for routes >8km.")
    elif target_distance > 25.0:
        st.sidebar.error("❌ Routes >25km are not supported in web interface. Use CLI instead.")
    
    # Route objective
    objectives = route_optimizer.get_available_objectives()
    
    selected_objective_name = st.sidebar.selectbox(
        "Route Objective",
        options=list(objectives.keys()),
        index=1,  # Default to "Maximum Elevation Gain"
        help="Choose optimization strategy"
    )
    selected_objective = objectives[selected_objective_name]
    
    # Algorithm selection
    algorithms = route_optimizer.get_available_algorithms()
    
    # Find genetic algorithm index
    default_algorithm_index = 0
    if "genetic" in algorithms:
        default_algorithm_index = algorithms.index("genetic")
    
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        options=algorithms,
        index=default_algorithm_index,  # Default to genetic if available
help="Genetic: Advanced genetic algorithm optimization"
    )
    
    # Show algorithm info
    if algorithm == "genetic":
        st.sidebar.info("🧬 Genetic Algorithm: Finds creative routes with optimized elevation profiles")
    
    # Footway filtering option
    exclude_footways = st.sidebar.checkbox(
        "Exclude footways/sidewalks",
        value=True,
        help="Prevents redundant back-and-forth routes on parallel sidewalks and roads. Recommended for most users."
    )
    
    # Elevation Data Source Selection
    st.sidebar.markdown("### Elevation Data")
    
    # Try to import elevation sources
    try:
        from elevation_data_sources import get_elevation_manager
        
        elevation_manager = get_elevation_manager()
        available_sources = elevation_manager.get_available_sources()
        
        if available_sources:
            # Add "auto" option
            source_options = ["auto"] + available_sources
            elevation_source = st.sidebar.selectbox(
                "Data Source",
                options=source_options,
                index=0,  # Default to auto
                help="Choose elevation data source (auto = best available)"
            )
            
            # Show source status
            active_source = elevation_manager.get_elevation_source()
            if active_source:
                source_info = active_source.get_source_info()
                resolution = active_source.get_resolution()
                st.sidebar.info(f"📊 Active: {source_info.get('type', 'Unknown')} ({resolution}m resolution)")
                
                # Show caching status if available
                if hasattr(active_source, 'get_cache_stats'):
                    if st.sidebar.button("📈 Show Cache Stats"):
                        cache_stats = active_source.get_cache_stats()
                        if cache_stats.get('enhanced_caching'):
                            perf = cache_stats.get('query_performance', {})
                            st.sidebar.text(f"Cache hits: {perf.get('cache_hit_rate_percent', 0):.1f}%")
            else:
                st.sidebar.warning("⚠️ No elevation source active")
        else:
            st.sidebar.warning("⚠️ No elevation sources available")
            elevation_source = "auto"
            
    except ImportError:
        st.sidebar.info("📊 Using basic elevation data")
        elevation_source = "auto"
    
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
        base_map = create_base_map(services)
        if base_map:
            map_data = st_folium(base_map, width=700, height=400)
        else:
            st.error("Failed to create map")
            return
        
        # Check for map click or use default
        start_node = 1529188403  # Default starting node
        
        if map_data['last_clicked']:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lon = map_data['last_clicked']['lng']
            
            # Find nearby nodes using network manager
            nearby_nodes = network_manager.get_nearby_nodes(
                graph, clicked_lat, clicked_lon, radius_km=0.1, max_nodes=1
            )
            
            if nearby_nodes:
                start_node = nearby_nodes[0][0]  # Closest node
                st.success(f"Selected starting point: Node {start_node}")
                node_info = network_manager.get_node_info(graph, start_node)
                st.info(f"Elevation: {node_info['elevation']:.0f}m")
        else:
            # Show default node info
            if network_manager.validate_node_exists(graph, start_node):
                node_info = network_manager.get_node_info(graph, start_node)
                st.info(f"Default starting point: Node {start_node}")
                st.info(f"Elevation: {node_info['elevation']:.0f}m")
                st.markdown("💡 *Click on the map to select a different starting point*")
            else:
                st.warning("Default starting point not found in current network")
        
        # Generate route button
        if st.button("🚀 Generate Optimized Route", type="primary"):
            if start_node and network_manager.validate_node_exists(graph, start_node):
                with st.spinner("Optimizing route..."):
                    
                    # Generate route using route optimizer
                    result = route_optimizer.optimize_route(
                        start_node=start_node,
                        target_distance_km=target_distance,
                        objective=selected_objective,
                        algorithm=algorithm,
                        exclude_footways=exclude_footways
                    )
                    
                    if result:
                        # Store result in session state
                        st.session_state.route_result = result
                        
                        # Show success message
                        solver_info = result.get('solver_info', {})
                        st.success(f"✅ Route generated in {solver_info.get('solve_time', 0):.2f} seconds")
                        
                        # Display route summary
                        summary = route_formatter.format_route_summary(result, 'web')
                        st.markdown(f"**Route Summary:** {summary}")
                        
                    else:
                        st.error("❌ Failed to generate route")
            else:
                st.error("❌ Invalid starting point selected")
    
    with col2:
        st.subheader("📊 Route Statistics")
        
        # Display route statistics if available
        if 'route_result' in st.session_state and st.session_state.route_result:
            result = st.session_state.route_result
            
            # Format statistics for web display
            analysis = route_analyzer.analyze_route(result)
            difficulty = route_analyzer.get_route_difficulty_rating(result)
            analysis['difficulty'] = difficulty
            
            web_stats = route_formatter.format_route_stats_web(result, analysis)
            
            # Display metrics
            for metric_name, metric_data in web_stats.items():
                if metric_name == 'difficulty':
                    # Special handling for difficulty badge
                    badge = route_formatter.create_difficulty_badge(analysis)
                    st.markdown(f"**Difficulty:** <span style='color: {badge['color']}'>{badge['text']}</span> {badge['score']}", unsafe_allow_html=True)
                else:
                    st.metric(
                        metric_name.replace('_', ' ').title(),
                        metric_data['value']
                    )
        else:
            st.info("Generate a route to see statistics")
    
    # Route visualization section
    if 'route_result' in st.session_state and st.session_state.route_result:
        result = st.session_state.route_result
        
        st.subheader("🗺️ Route Visualization")
        
        # Create route map
        route_map = create_route_map(services, result)
        if route_map:
            st_folium(route_map, width=700, height=400)
        
        st.subheader("📈 Elevation Profile")
        
        # Create elevation plot
        elevation_plot = create_elevation_plot(services, result)
        if elevation_plot:
            st.plotly_chart(elevation_plot, use_container_width=True)
        
        # Turn-by-turn directions
        st.subheader("📋 Turn-by-Turn Directions")
        
        directions = route_analyzer.generate_directions(result)
        formatted_directions = route_formatter.format_directions_web(directions)
        
        # Display directions in a table
        if formatted_directions:
            df = pd.DataFrame(formatted_directions)
            # Reorder columns for better display
            column_order = ['step', 'instruction', 'elevation', 'elevation_change', 'distance', 'terrain']
            df = df[column_order]
            st.dataframe(df, use_container_width=True)
        
        # Additional analysis
        st.subheader("🔍 Route Analysis")
        
        # Elevation zones
        zones = elevation_profiler.get_elevation_zones(result, zone_count=3)
        if zones:
            st.markdown("**Elevation Zones:**")
            for zone in zones:
                st.markdown(f"- Zone {zone['zone_number']}: {zone['start_km']:.2f}-{zone['end_km']:.2f}km, "
                           f"Avg elevation: {zone['avg_elevation']:.0f}m")
        
        # Climbing segments
        climbing_segments = elevation_profiler.get_climbing_segments(result, min_gain=10)
        if climbing_segments:
            st.markdown("**Climbing Segments:**")
            for i, segment in enumerate(climbing_segments, 1):
                st.markdown(f"- Climb {i}: {segment['start_km']:.2f}-{segment['end_km']:.2f}km, "
                           f"Gain: {segment['elevation_gain']:.0f}m, "
                           f"Grade: {segment['avg_grade']:.1f}%")
        
        # Export options
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # JSON export
            if st.button("📄 Export as JSON"):
                profile_data = elevation_profiler.generate_profile_data(result)
                json_data = route_formatter.export_route_json(
                    result, analysis, directions, profile_data
                )
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name=f"route_{start_node}_{target_distance}km.json",
                    mime="application/json"
                )
        
        with col2:
            # Route summary
            if st.button("📋 Generate Summary"):
                summary = route_formatter.format_route_summary(result, 'cli')
                st.code(summary)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Route Formatter
Formats route data for different output formats (CLI, web, etc.)
"""

from typing import Dict, List, Any, Optional
import json


class RouteFormatter:
    """Formats route data for different presentation contexts"""
    
    def __init__(self):
        """Initialize route formatter"""
        pass
    
    def format_route_stats_cli(self, route_result: Dict[str, Any], analysis: Dict[str, Any] = None) -> str:
        """Format route statistics for CLI output
        
        Args:
            route_result: Route result from optimizer
            analysis: Optional route analysis data
            
        Returns:
            Formatted string for CLI display
        """
        if not route_result:
            return "❌ No route data available"
        
        stats = route_result.get('stats', {})
        route = route_result.get('route', [])
        
        lines = []
        lines.append("📊 Route Statistics:")
        lines.append("=" * 50)
        lines.append(f"Distance:        {stats.get('total_distance_km', 0):.2f} km")
        lines.append(f"Elevation Gain:  {stats.get('total_elevation_gain_m', 0):.0f} m")
        lines.append(f"Elevation Loss:  {stats.get('total_elevation_loss_m', 0):.0f} m")
        lines.append(f"Net Elevation:   {stats.get('net_elevation_gain_m', 0):+.0f} m")
        lines.append(f"Max Grade:       {stats.get('max_grade_percent', 0):.1f}%")
        lines.append(f"Est. Time:       {stats.get('estimated_time_min', 0):.0f} minutes")
        lines.append(f"Route Points:    {len(route)} intersections")
        lines.append(f"Algorithm:       {route_result.get('algorithm', 'Unknown')}")
        lines.append(f"Objective:       {route_result.get('objective', 'Unknown')}")
        
        # Add solver info if available
        solver_info = route_result.get('solver_info', {})
        if solver_info:
            lines.append(f"Solver:          {solver_info.get('solver_type', 'Unknown')}")
            lines.append(f"Solve Time:      {solver_info.get('solve_time', 0):.2f} seconds")
        
        # Add analysis info if available
        if analysis:
            difficulty = analysis.get('difficulty', {})
            if difficulty:
                lines.append(f"Difficulty:      {difficulty.get('rating', 'Unknown')} ({difficulty.get('score', 0):.0f}/100)")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def format_route_stats_web(self, route_result: Dict[str, Any], analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format route statistics for web display
        
        Args:
            route_result: Route result from optimizer
            analysis: Optional route analysis data
            
        Returns:
            Dictionary with formatted metrics for web display
        """
        if not route_result:
            return {}
        
        stats = route_result.get('stats', {})
        route = route_result.get('route', [])
        solver_info = route_result.get('solver_info', {})
        
        metrics = {
            'distance': {
                'value': f"{stats.get('total_distance_km', 0):.2f} km",
                'raw_value': stats.get('total_distance_km', 0),
                'unit': 'km'
            },
            'elevation_gain': {
                'value': f"{stats.get('total_elevation_gain_m', 0):.0f} m",
                'raw_value': stats.get('total_elevation_gain_m', 0),
                'unit': 'm'
            },
            'elevation_loss': {
                'value': f"{stats.get('total_elevation_loss_m', 0):.0f} m",
                'raw_value': stats.get('total_elevation_loss_m', 0),
                'unit': 'm'
            },
            'net_elevation': {
                'value': f"{stats.get('net_elevation_gain_m', 0):+.0f} m",
                'raw_value': stats.get('net_elevation_gain_m', 0),
                'unit': 'm'
            },
            'max_grade': {
                'value': f"{stats.get('max_grade_percent', 0):.1f}%",
                'raw_value': stats.get('max_grade_percent', 0),
                'unit': '%'
            },
            'estimated_time': {
                'value': f"{stats.get('estimated_time_min', 0):.0f} min",
                'raw_value': stats.get('estimated_time_min', 0),
                'unit': 'min'
            },
            'route_points': {
                'value': f"{len(route)} intersections",
                'raw_value': len(route),
                'unit': 'intersections'
            }
        }
        
        # Add solver info
        if solver_info:
            metrics['solve_time'] = {
                'value': f"{solver_info.get('solve_time', 0):.2f} sec",
                'raw_value': solver_info.get('solve_time', 0),
                'unit': 'sec'
            }
        
        # Add difficulty rating if available
        if analysis and analysis.get('difficulty'):
            difficulty = analysis['difficulty']
            metrics['difficulty'] = {
                'value': difficulty.get('rating', 'Unknown'),
                'raw_value': difficulty.get('score', 0),
                'unit': f"({difficulty.get('score', 0):.0f}/100)"
            }
        
        return metrics
    
    def format_directions_cli(self, directions: List[Dict[str, Any]]) -> str:
        """Format turn-by-turn directions for CLI output
        
        Args:
            directions: List of direction dictionaries
            
        Returns:
            Formatted string for CLI display
        """
        if not directions:
            return "❌ No directions available"
        
        lines = []
        lines.append("📋 Turn-by-Turn Directions:")
        lines.append("=" * 60)
        
        for direction in directions:
            step = direction.get('step', 0)
            instruction = direction.get('instruction', '')
            elevation = direction.get('elevation', 0)
            elevation_change = direction.get('elevation_change', 0)
            cumulative_distance = direction.get('cumulative_distance_km', 0)
            
            lines.append(f"{step}. {instruction}")
            lines.append(f"   Elevation: {elevation:.0f}m ({elevation_change:+.0f}m)")
            lines.append(f"   Distance: {cumulative_distance:.2f} km")
            lines.append()
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def format_directions_web(self, directions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format directions for web display
        
        Args:
            directions: List of direction dictionaries
            
        Returns:
            List of formatted direction dictionaries
        """
        formatted_directions = []
        
        for direction in directions:
            formatted_direction = {
                'step': direction.get('step', 0),
                'instruction': direction.get('instruction', ''),
                'elevation': f"{direction.get('elevation', 0):.0f}m",
                'elevation_change': f"{direction.get('elevation_change', 0):+.0f}m",
                'distance': f"{direction.get('cumulative_distance_km', 0):.2f} km",
                'terrain': direction.get('terrain', 'level'),
                'type': direction.get('type', 'continue')
            }
            formatted_directions.append(formatted_direction)
        
        return formatted_directions
    
    def format_elevation_profile_data(self, profile_data: Dict[str, Any], 
                                    format_type: str = "web") -> Dict[str, Any]:
        """Format elevation profile data for visualization
        
        Args:
            profile_data: Elevation profile data
            format_type: Output format ('web', 'cli', 'json')
            
        Returns:
            Formatted profile data
        """
        if not profile_data:
            return {}
        
        if format_type == "web":
            return {
                'distances': profile_data.get('distances_km', []),
                'elevations': profile_data.get('elevations', []),
                'coordinates': profile_data.get('coordinates', []),
                'stats': profile_data.get('elevation_stats', {}),
                'total_distance': profile_data.get('total_distance_km', 0)
            }
        elif format_type == "cli":
            # Format for CLI text output
            stats = profile_data.get('elevation_stats', {})
            lines = []
            lines.append(f"📈 Elevation Profile:")
            lines.append(f"   Min Elevation: {stats.get('min_elevation', 0):.0f}m")
            lines.append(f"   Max Elevation: {stats.get('max_elevation', 0):.0f}m")
            lines.append(f"   Elevation Range: {stats.get('elevation_range', 0):.0f}m")
            lines.append(f"   Max Grade: {stats.get('max_grade', 0):.1f}%")
            lines.append(f"   Steep Sections: {stats.get('steep_section_count', 0)}")
            return "\n".join(lines)
        else:  # json
            return profile_data
    
    def export_route_json(self, route_result: Dict[str, Any], analysis: Dict[str, Any] = None,
                         directions: List[Dict[str, Any]] = None, 
                         profile_data: Dict[str, Any] = None) -> str:
        """Export complete route data as JSON
        
        Args:
            route_result: Route result from optimizer
            analysis: Optional route analysis
            directions: Optional directions list
            profile_data: Optional elevation profile data
            
        Returns:
            JSON string with complete route data
        """
        export_data = {
            'route_result': route_result,
            'analysis': analysis,
            'directions': directions,
            'elevation_profile': profile_data,
            'export_timestamp': self._get_timestamp(),
            'format_version': '1.0'
        }
        
        return json.dumps(export_data, indent=2, default=str)
    
    def format_route_summary(self, route_result: Dict[str, Any], 
                           format_type: str = "cli") -> str:
        """Format a brief route summary
        
        Args:
            route_result: Route result from optimizer
            format_type: Output format ('cli' or 'web')
            
        Returns:
            Formatted summary string
        """
        if not route_result:
            return "No route data"
        
        stats = route_result.get('stats', {})
        distance = stats.get('total_distance_km', 0)
        elevation_gain = stats.get('total_elevation_gain_m', 0)
        time_min = stats.get('estimated_time_min', 0)
        
        if format_type == "cli":
            return f"🏃 {distance:.1f}km route • {elevation_gain:.0f}m elevation gain • ~{time_min:.0f}min"
        else:  # web
            return f"{distance:.1f}km • {elevation_gain:.0f}m gain • ~{time_min:.0f}min"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def create_difficulty_badge(self, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Create difficulty badge data
        
        Args:
            analysis: Route analysis with difficulty rating
            
        Returns:
            Dictionary with badge information
        """
        difficulty = analysis.get('difficulty', {}) if analysis else {}
        rating = difficulty.get('rating', 'Unknown')
        score = difficulty.get('score', 0)
        
        # Color coding for difficulty
        color_map = {
            'Very Easy': '#4CAF50',    # Green
            'Easy': '#8BC34A',         # Light Green
            'Moderate': '#FF9800',     # Orange
            'Hard': '#FF5722',         # Red Orange
            'Very Hard': '#F44336',    # Red
            'Unknown': '#9E9E9E'       # Grey
        }
        
        return {
            'text': rating,
            'score': f"{score:.0f}/100",
            'color': color_map.get(rating, '#9E9E9E'),
            'description': f"Difficulty: {rating} ({score:.0f}/100)"
        }
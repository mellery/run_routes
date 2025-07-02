#!/usr/bin/env python3
"""
Route Analyzer
Analyzes routes and generates statistics and directions
"""

from typing import Dict, List, Any, Optional
import networkx as nx


class RouteAnalyzer:
    """Analyzes routes and generates statistics and directions"""
    
    def __init__(self, graph: nx.Graph):
        """Initialize route analyzer
        
        Args:
            graph: NetworkX graph
        """
        self.graph = graph
    
    def analyze_route(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a route and return comprehensive statistics
        
        Args:
            route_result: Route result from optimizer
            
        Returns:
            Dictionary with detailed route analysis
        """
        if not route_result or not route_result.get('route'):
            return {}
        
        route = route_result['route']
        
        # Get basic statistics from route result
        stats = route_result.get('stats', {})
        
        # Calculate additional statistics
        additional_stats = self._calculate_additional_stats(route)
        
        # Merge statistics
        analysis = {
            'basic_stats': stats,
            'additional_stats': additional_stats,
            'route_info': {
                'route_length': len(route),
                'start_node': route[0] if route else None,
                'end_node': route[-1] if len(route) > 1 else None,
                'is_loop': route[0] == route[-1] if len(route) > 1 else False
            }
        }
        
        return analysis
    
    def _calculate_additional_stats(self, route: List[int]) -> Dict[str, Any]:
        """Calculate additional route statistics
        
        Args:
            route: List of node IDs
            
        Returns:
            Dictionary with additional statistics
        """
        if not route or len(route) < 2:
            return {}
        
        from route import haversine_distance
        
        # Initialize counters
        total_segments = 0
        uphill_segments = 0
        downhill_segments = 0
        level_segments = 0
        steepest_uphill = 0
        steepest_downhill = 0
        elevation_changes = []
        
        # Analyze each segment
        for i in range(len(route)):
            current_node = route[i]
            if i < len(route) - 1:
                next_node = route[i + 1]
            else:
                # Return to start
                next_node = route[0]
            
            if current_node in self.graph.nodes and next_node in self.graph.nodes:
                current_elev = self.graph.nodes[current_node].get('elevation', 0)
                next_elev = self.graph.nodes[next_node].get('elevation', 0)
                
                elevation_change = next_elev - current_elev
                elevation_changes.append(elevation_change)
                
                # Calculate distance for grade
                current_data = self.graph.nodes[current_node]
                next_data = self.graph.nodes[next_node]
                distance = haversine_distance(
                    current_data['y'], current_data['x'],
                    next_data['y'], next_data['x']
                )
                
                # Calculate grade percentage
                if distance > 0:
                    grade = (elevation_change / distance) * 100
                    
                    if grade > 2:
                        uphill_segments += 1
                        steepest_uphill = max(steepest_uphill, grade)
                    elif grade < -2:
                        downhill_segments += 1
                        steepest_downhill = min(steepest_downhill, grade)
                    else:
                        level_segments += 1
                
                total_segments += 1
        
        # Calculate summary statistics
        avg_elevation_change = sum(elevation_changes) / len(elevation_changes) if elevation_changes else 0
        
        return {
            'total_segments': total_segments,
            'uphill_segments': uphill_segments,
            'downhill_segments': downhill_segments,
            'level_segments': level_segments,
            'uphill_percentage': (uphill_segments / total_segments * 100) if total_segments > 0 else 0,
            'downhill_percentage': (downhill_segments / total_segments * 100) if total_segments > 0 else 0,
            'steepest_uphill_grade': steepest_uphill,
            'steepest_downhill_grade': steepest_downhill,
            'avg_elevation_change': avg_elevation_change,
            'elevation_changes': elevation_changes
        }
    
    def generate_directions(self, route_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate turn-by-turn directions
        
        Args:
            route_result: Route result from optimizer
            
        Returns:
            List of direction dictionaries
        """
        if not route_result or not route_result.get('route'):
            return []
        
        route = route_result['route']
        directions = []
        
        from route import haversine_distance
        
        # Start instruction
        if route and route[0] in self.graph.nodes:
            start_data = self.graph.nodes[route[0]]
            directions.append({
                'step': 1,
                'type': 'start',
                'instruction': f"Start at intersection (Node {route[0]})",
                'node_id': route[0],
                'elevation': start_data.get('elevation', 0),
                'elevation_change': 0,
                'distance_km': 0.0,
                'cumulative_distance_km': 0.0,
                'terrain': 'start'
            })
        
        # Route segments
        cumulative_distance = 0
        for i in range(1, len(route)):
            if route[i] in self.graph.nodes and route[i-1] in self.graph.nodes:
                curr_data = self.graph.nodes[route[i]]
                prev_data = self.graph.nodes[route[i-1]]
                
                # Calculate segment distance
                segment_dist = haversine_distance(
                    prev_data['y'], prev_data['x'],
                    curr_data['y'], curr_data['x']
                )
                cumulative_distance += segment_dist
                
                # Determine terrain
                elevation_change = curr_data.get('elevation', 0) - prev_data.get('elevation', 0)
                if elevation_change > 5:
                    terrain = "uphill"
                elif elevation_change < -5:
                    terrain = "downhill"
                else:
                    terrain = "level"
                
                directions.append({
                    'step': i + 1,
                    'type': 'continue',
                    'instruction': f"Continue to intersection (Node {route[i]}) - {terrain}",
                    'node_id': route[i],
                    'elevation': curr_data.get('elevation', 0),
                    'elevation_change': elevation_change,
                    'distance_km': segment_dist / 1000,
                    'cumulative_distance_km': cumulative_distance / 1000,
                    'terrain': terrain
                })
        
        # Return to start
        if len(route) > 1:
            total_distance = route_result.get('stats', {}).get('total_distance_km', cumulative_distance / 1000)
            directions.append({
                'step': len(route) + 1,
                'type': 'finish',
                'instruction': "Return to starting point to complete the loop",
                'node_id': route[0],
                'elevation': directions[0]['elevation'],
                'elevation_change': 0,
                'distance_km': 0,
                'cumulative_distance_km': total_distance,
                'terrain': 'finish'
            })
        
        return directions
    
    def get_route_difficulty_rating(self, route_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate route difficulty rating
        
        Args:
            route_result: Route result from optimizer
            
        Returns:
            Dictionary with difficulty rating and explanation
        """
        if not route_result:
            return {'rating': 'unknown', 'score': 0, 'factors': []}
        
        stats = route_result.get('stats', {})
        analysis = self.analyze_route(route_result)
        additional_stats = analysis.get('additional_stats', {})
        
        # Difficulty factors
        distance_km = stats.get('total_distance_km', 0)
        elevation_gain = stats.get('total_elevation_gain_m', 0)
        uphill_percentage = additional_stats.get('uphill_percentage', 0)
        steepest_grade = additional_stats.get('steepest_uphill_grade', 0)
        
        # Calculate difficulty score (0-100)
        score = 0
        factors = []
        
        # Distance factor (0-25 points)
        if distance_km > 10:
            score += 25
            factors.append("Very long distance")
        elif distance_km > 5:
            score += 15
            factors.append("Long distance")
        elif distance_km > 2:
            score += 5
            factors.append("Moderate distance")
        
        # Elevation gain factor (0-30 points)
        elevation_per_km = elevation_gain / distance_km if distance_km > 0 else 0
        if elevation_per_km > 100:
            score += 30
            factors.append("Very high elevation gain")
        elif elevation_per_km > 50:
            score += 20
            factors.append("High elevation gain")
        elif elevation_per_km > 20:
            score += 10
            factors.append("Moderate elevation gain")
        
        # Uphill percentage factor (0-25 points)
        if uphill_percentage > 50:
            score += 25
            factors.append("Mostly uphill")
        elif uphill_percentage > 30:
            score += 15
            factors.append("Significant uphill")
        elif uphill_percentage > 15:
            score += 5
            factors.append("Some uphill")
        
        # Steepest grade factor (0-20 points)
        if steepest_grade > 15:
            score += 20
            factors.append("Very steep sections")
        elif steepest_grade > 10:
            score += 15
            factors.append("Steep sections")
        elif steepest_grade > 5:
            score += 5
            factors.append("Moderate inclines")
        
        # Determine rating
        if score >= 70:
            rating = "Very Hard"
        elif score >= 50:
            rating = "Hard"
        elif score >= 30:
            rating = "Moderate"
        elif score >= 15:
            rating = "Easy"
        else:
            rating = "Very Easy"
        
        return {
            'rating': rating,
            'score': score,
            'factors': factors,
            'distance_km': distance_km,
            'elevation_gain': elevation_gain,
            'elevation_per_km': elevation_per_km,
            'uphill_percentage': uphill_percentage,
            'steepest_grade': steepest_grade
        }
#!/usr/bin/env python3
"""
Genetic Algorithm Chromosome Classes
Implements segment-based route representation for GA optimization
"""

import math
import time
from typing import List, Optional, Tuple, Dict, Any
import networkx as nx
import numpy as np


class RouteSegment:
    """Represents a route segment between two nodes with complete path information"""
    
    def __init__(self, start_node: int, end_node: int, path_nodes: List[int]):
        """Initialize route segment
        
        Args:
            start_node: Starting intersection node ID
            end_node: Ending intersection node ID
            path_nodes: Complete path including intermediate nodes
        """
        self.start_node = start_node
        self.end_node = end_node
        self.path_nodes = path_nodes.copy() if path_nodes else [start_node, end_node]
        
        # Properties calculated by calculate_properties()
        self.length = 0.0                 # Segment length in meters
        self.elevation_gain = 0.0          # Elevation gain in meters
        self.elevation_loss = 0.0          # Elevation loss in meters
        self.net_elevation = 0.0           # Net elevation change
        self.max_grade = 0.0               # Maximum grade percentage
        self.avg_grade = 0.0               # Average grade percentage
        self.direction = None              # Cardinal direction (N,S,E,W,NE,etc.)
        self.grade_stats = {}              # Detailed grade statistics
        self.is_valid = True               # Segment connectivity flag
        
    def calculate_properties(self, graph: nx.Graph) -> None:
        """Calculate segment properties from graph data
        
        Args:
            graph: NetworkX graph with elevation and distance data
        """
        if len(self.path_nodes) < 2:
            self.is_valid = False
            return
            
        try:
            # Calculate total length
            total_length = 0.0
            elevations = []
            grades = []
            
            for i in range(len(self.path_nodes) - 1):
                node1 = self.path_nodes[i]
                node2 = self.path_nodes[i + 1]
                
                # Check if edge exists
                if not graph.has_edge(node1, node2):
                    self.is_valid = False
                    return
                
                # Get edge properties (handle MultiGraph format)
                edge_data = graph[node1][node2]
                # Check if it's a MultiGraph format with nested keys
                if 0 in edge_data:
                    edge_length = edge_data[0].get('length', 0.0)
                else:
                    edge_length = edge_data.get('length', 0.0)
                total_length += edge_length
                
                # Get node elevations
                elev1 = graph.nodes[node1].get('elevation', 0.0)
                elev2 = graph.nodes[node2].get('elevation', 0.0)
                elevations.extend([elev1, elev2])
                
                # Calculate grade for this edge
                if edge_length > 0:
                    elevation_change = elev2 - elev1
                    grade = (elevation_change / edge_length) * 100
                    grades.append(abs(grade))
            
            # Set basic properties
            self.length = total_length
            
            # Calculate elevation statistics
            if elevations:
                elevations = list(set(elevations))  # Remove duplicates
                start_elevation = graph.nodes[self.start_node].get('elevation', 0.0)
                end_elevation = graph.nodes[self.end_node].get('elevation', 0.0)
                
                self.net_elevation = end_elevation - start_elevation
                
                # Calculate cumulative elevation gain/loss
                gain = 0.0
                loss = 0.0
                for i in range(len(self.path_nodes) - 1):
                    elev1 = graph.nodes[self.path_nodes[i]].get('elevation', 0.0)
                    elev2 = graph.nodes[self.path_nodes[i + 1]].get('elevation', 0.0)
                    diff = elev2 - elev1
                    if diff > 0:
                        gain += diff
                    else:
                        loss += abs(diff)
                
                self.elevation_gain = gain
                self.elevation_loss = loss
            
            # Calculate grade statistics
            if grades:
                self.max_grade = max(grades)
                self.avg_grade = sum(grades) / len(grades)
                self.grade_stats = {
                    'max_grade_percent': self.max_grade,
                    'avg_grade_percent': self.avg_grade,
                    'steep_sections': len([g for g in grades if g > 8.0]),
                    'grade_distribution': grades
                }
            
            # Calculate direction
            self.direction = self._calculate_direction(graph)
            
        except Exception as e:
            print(f"Error calculating segment properties: {e}")
            self.is_valid = False
    
    def _calculate_direction(self, graph: nx.Graph) -> Optional[str]:
        """Calculate the primary direction of the segment"""
        try:
            start_lat = graph.nodes[self.start_node]['y']
            start_lon = graph.nodes[self.start_node]['x']
            end_lat = graph.nodes[self.end_node]['y']
            end_lon = graph.nodes[self.end_node]['x']
            
            # Calculate bearing
            lat1, lon1 = math.radians(start_lat), math.radians(start_lon)
            lat2, lon2 = math.radians(end_lat), math.radians(end_lon)
            
            dlon = lon2 - lon1
            y = math.sin(dlon) * math.cos(lat2)
            x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
            
            bearing = math.atan2(y, x)
            bearing = math.degrees(bearing)
            bearing = (bearing + 360) % 360
            
            # Convert to cardinal direction
            directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
            index = round(bearing / 45) % 8
            return directions[index]
            
        except Exception:
            return None
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for the segment"""
        return {
            'length_m': self.length,
            'elevation_gain_m': self.elevation_gain,
            'elevation_loss_m': self.elevation_loss,
            'net_elevation_m': self.net_elevation,
            'max_grade_percent': self.max_grade,
            'avg_grade_percent': self.avg_grade,
            'direction': self.direction,
            'is_valid': self.is_valid,
            'node_count': len(self.path_nodes)
        }
    
    def copy(self) -> 'RouteSegment':
        """Create a deep copy of the segment"""
        new_segment = RouteSegment(self.start_node, self.end_node, self.path_nodes)
        new_segment.length = self.length
        new_segment.elevation_gain = self.elevation_gain
        new_segment.elevation_loss = self.elevation_loss
        new_segment.net_elevation = self.net_elevation
        new_segment.max_grade = self.max_grade
        new_segment.avg_grade = self.avg_grade
        new_segment.direction = self.direction
        new_segment.grade_stats = self.grade_stats.copy()
        new_segment.is_valid = self.is_valid
        return new_segment


class RouteChromosome:
    """Represents a complete route as a sequence of segments (GA chromosome)"""
    
    def __init__(self, segments: Optional[List[RouteSegment]] = None):
        """Initialize route chromosome
        
        Args:
            segments: List of RouteSegment objects forming the route
        """
        self.segments = segments.copy() if segments else []
        
        # Cached fitness metrics
        self.fitness = None                # Fitness score (higher = better)
        self.distance = None               # Total distance in meters
        self.elevation_gain = None         # Total elevation gain in meters
        self.elevation_loss = None         # Total elevation loss in meters
        self.is_valid = True               # Route connectivity flag
        self.is_circular = False           # Whether route returns to start
        
        # Additional metrics
        self.diversity_score = None        # Route diversity measure
        self.difficulty_score = None       # Route difficulty rating
        self.objective_score = None        # Objective-specific score
        
        # Metadata
        self.generation = 0                # Generation when created
        self.parent_ids = []               # Parent chromosome IDs (for tracking)
        self.creation_method = "unknown"   # How chromosome was created
        
    def add_segment(self, segment: RouteSegment) -> None:
        """Add a segment to the route"""
        self.segments.append(segment)
        self._invalidate_cache()
    
    def insert_segment(self, index: int, segment: RouteSegment) -> None:
        """Insert a segment at specified index"""
        self.segments.insert(index, segment)
        self._invalidate_cache()
    
    def remove_segment(self, index: int) -> Optional[RouteSegment]:
        """Remove and return segment at specified index"""
        if 0 <= index < len(self.segments):
            segment = self.segments.pop(index)
            self._invalidate_cache()
            return segment
        return None
    
    def _invalidate_cache(self) -> None:
        """Invalidate cached calculations"""
        self.fitness = None
        self.distance = None
        self.elevation_gain = None
        self.elevation_loss = None
        self.diversity_score = None
        self.difficulty_score = None
        self.objective_score = None
    
    def validate_connectivity(self) -> bool:
        """Validate that all segments are properly connected"""
        if not self.segments:
            self.is_valid = False
            return False
        
        # Check each segment is valid
        for segment in self.segments:
            if not segment.is_valid:
                self.is_valid = False
                return False
        
        # Check segments are connected
        for i in range(len(self.segments) - 1):
            if self.segments[i].end_node != self.segments[i + 1].start_node:
                self.is_valid = False
                return False
        
        # Check if route is circular
        if (len(self.segments) > 0 and 
            self.segments[0].start_node == self.segments[-1].end_node):
            self.is_circular = True
        
        self.is_valid = True
        return True
    
    def get_total_distance(self) -> float:
        """Get total route distance in meters"""
        if self.distance is None:
            self.distance = sum(segment.length for segment in self.segments)
        return self.distance
    
    def get_elevation_gain(self) -> float:
        """Get total elevation gain in meters"""
        if self.elevation_gain is None:
            self.elevation_gain = sum(segment.elevation_gain for segment in self.segments)
        return self.elevation_gain
    
    def get_total_elevation_gain(self) -> float:
        """Alias for get_elevation_gain() for compatibility"""
        return self.get_elevation_gain()
    
    def get_elevation_loss(self) -> float:
        """Get total elevation loss in meters"""
        if self.elevation_loss is None:
            self.elevation_loss = sum(segment.elevation_loss for segment in self.segments)
        return self.elevation_loss
    
    def get_net_elevation(self) -> float:
        """Get net elevation change in meters"""
        return self.get_elevation_gain() - self.get_elevation_loss()
    
    def get_max_grade(self) -> float:
        """Get maximum grade across all segments"""
        if not self.segments:
            return 0.0
        return max(segment.max_grade for segment in self.segments)
    
    def get_route_nodes(self) -> List[int]:
        """Get list of all nodes in the route (intersection nodes only)"""
        if not self.segments:
            return []
        
        nodes = [self.segments[0].start_node]
        for segment in self.segments:
            nodes.append(segment.end_node)
        return nodes
    
    def get_nodes(self) -> List[int]:
        """Alias for get_route_nodes() for compatibility"""
        return self.get_route_nodes()
    
    def get_complete_path(self) -> List[int]:
        """Get complete path including all intermediate nodes"""
        if not self.segments:
            return []
        
        complete_path = []
        for i, segment in enumerate(self.segments):
            if i == 0:
                complete_path.extend(segment.path_nodes[:-1])  # Exclude last node
            else:
                complete_path.extend(segment.path_nodes[1:-1])  # Exclude first and last
            
            # Always add the end node
            if i == len(self.segments) - 1:
                complete_path.append(segment.end_node)
        
        return complete_path
    
    def calculate_diversity_score(self) -> float:
        """Calculate route diversity based on direction variety"""
        if not self.segments:
            self.diversity_score = 0.0
            return 0.0
        
        # Get directions
        directions = [segment.direction for segment in self.segments if segment.direction]
        if not directions:
            self.diversity_score = 0.0
            return 0.0
        
        # Calculate direction diversity
        unique_directions = set(directions)
        direction_diversity = len(unique_directions) / 8.0  # 8 possible directions
        
        # Penalize back-and-forth patterns
        pattern_penalty = 0.0
        opposite_directions = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E',
                             'NE': 'SW', 'SW': 'NE', 'NW': 'SE', 'SE': 'NW'}
        
        for i in range(len(directions) - 1):
            if directions[i + 1] == opposite_directions.get(directions[i]):
                pattern_penalty += 0.1
        
        self.diversity_score = max(0.0, direction_diversity - pattern_penalty)
        return self.diversity_score
    
    def get_route_stats(self) -> Dict[str, Any]:
        """Get comprehensive route statistics"""
        return {
            'total_distance_km': self.get_total_distance() / 1000,
            'total_distance_m': self.get_total_distance(),
            'total_elevation_gain_m': self.get_elevation_gain(),
            'total_elevation_loss_m': self.get_elevation_loss(),
            'net_elevation_gain_m': self.get_net_elevation(),
            'max_grade_percent': self.get_max_grade(),
            'segment_count': len(self.segments),
            'node_count': len(self.get_route_nodes()),
            'is_valid': self.is_valid,
            'is_circular': self.is_circular,
            'diversity_score': self.calculate_diversity_score(),
            'estimated_time_min': self.get_total_distance() / 1000 / 8.0 * 60,  # 8 km/h pace
        }
    
    def copy(self) -> 'RouteChromosome':
        """Create a deep copy of the chromosome"""
        new_chromosome = RouteChromosome([segment.copy() for segment in self.segments])
        new_chromosome.fitness = self.fitness
        new_chromosome.distance = self.distance
        new_chromosome.elevation_gain = self.elevation_gain
        new_chromosome.elevation_loss = self.elevation_loss
        new_chromosome.is_valid = self.is_valid
        new_chromosome.is_circular = self.is_circular
        new_chromosome.diversity_score = self.diversity_score
        new_chromosome.difficulty_score = self.difficulty_score
        new_chromosome.objective_score = self.objective_score
        new_chromosome.generation = self.generation
        new_chromosome.parent_ids = self.parent_ids.copy()
        new_chromosome.creation_method = self.creation_method
        return new_chromosome
    
    def to_route_result(self, objective: str, solve_time: float = 0.0) -> Dict[str, Any]:
        """Convert chromosome to standard route result format"""
        stats = self.get_route_stats()
        
        return {
            'route': self.get_route_nodes(),
            'cost': -self.fitness if self.fitness else float('inf'),
            'stats': stats,
            'solve_time': solve_time,
            'objective': objective,
            'algorithm': 'genetic',
            'target_distance_km': stats['total_distance_km'],
            'solver_info': {
                'solver_type': 'genetic',
                'solve_time': solve_time,
                'algorithm_used': 'genetic_algorithm',
                'objective_used': objective,
                'generation': self.generation,
                'creation_method': self.creation_method,
                'fitness': self.fitness,
                'diversity_score': self.diversity_score
            }
        }
    
    def __str__(self) -> str:
        """String representation of chromosome"""
        stats = self.get_route_stats()
        fitness_str = f"{self.fitness:.3f}" if self.fitness is not None else "None"
        return (f"RouteChromosome(segments={len(self.segments)}, "
                f"distance={stats['total_distance_km']:.2f}km, "
                f"elevation={stats['total_elevation_gain_m']:.1f}m, "
                f"fitness={fitness_str}, "
                f"valid={self.is_valid})")
    
    def __repr__(self) -> str:
        return self.__str__()
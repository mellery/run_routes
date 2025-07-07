#!/usr/bin/env python3
"""
GA Configuration Management System
Centralized configuration management for genetic algorithm parameters
"""

import json
import os
import time
import copy
import yaml
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from collections import defaultdict

from route_objective import RouteObjective
from ga_parameter_tuner import ParameterRange


class ConfigScope(Enum):
    """Configuration scope levels"""
    GLOBAL = "global"           # System-wide defaults
    PROFILE = "profile"         # Named configuration profiles
    SESSION = "session"         # Session-specific overrides
    RUNTIME = "runtime"         # Runtime dynamic adjustments


class ValidationLevel(Enum):
    """Configuration validation levels"""
    NONE = "none"               # No validation
    BASIC = "basic"             # Basic type and range checking
    STRICT = "strict"           # Comprehensive validation with dependencies
    CUSTOM = "custom"           # Custom validation functions


@dataclass
class ConfigValidationRule:
    """Configuration validation rule"""
    parameter_name: str
    validator: Callable[[Any], bool]
    error_message: str
    dependency_params: List[str] = field(default_factory=list)
    warning_only: bool = False


@dataclass
class ConfigProfile:
    """Named configuration profile"""
    name: str
    description: str
    parameters: Dict[str, Any]
    objective: Optional[RouteObjective] = None
    target_distance_range: Optional[tuple] = None
    created_timestamp: float = field(default_factory=time.time)
    last_used_timestamp: float = field(default_factory=time.time)
    usage_count: int = 0
    tags: List[str] = field(default_factory=list)


@dataclass
class ConfigChange:
    """Configuration change record"""
    parameter_name: str
    old_value: Any
    new_value: Any
    scope: ConfigScope
    timestamp: float
    reason: str
    source: str  # user, tuner, optimizer, etc.


@dataclass
class ConfigSnapshot:
    """Configuration state snapshot"""
    timestamp: float
    scope: ConfigScope
    parameters: Dict[str, Any]
    profile_name: Optional[str] = None
    generation: Optional[int] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class GAConfigManager:
    """Centralized configuration management for genetic algorithms"""
    
    def __init__(self, config_dir: str = "config", config: Optional[Dict[str, Any]] = None):
        """Initialize configuration manager
        
        Args:
            config_dir: Directory for configuration files
            config: Configuration options for manager
        """
        default_config = {
            'auto_save': True,                  # Auto-save configuration changes
            'save_interval_seconds': 300,       # Auto-save interval
            'max_history_size': 1000,          # Maximum change history size
            'validation_level': ValidationLevel.STRICT,  # Validation level
            'enable_profiles': True,            # Enable named profiles
            'enable_runtime_adaptation': True,  # Enable runtime parameter changes
            'config_file_format': 'json',       # json, yaml, or both
            'backup_count': 5,                  # Number of backup files to keep
            'allow_parameter_expansion': True,   # Allow adding new parameters
            'profile_auto_selection': True,     # Auto-select profiles based on context
            'parameter_inheritance': True       # Enable parameter inheritance from profiles
        }
        
        self.config = {**default_config, **(config or {})}
        self.config_dir = config_dir
        
        # Configuration storage
        self.configurations = {
            ConfigScope.GLOBAL: {},
            ConfigScope.PROFILE: {},
            ConfigScope.SESSION: {},
            ConfigScope.RUNTIME: {}
        }
        
        # Named profiles
        self.profiles = {}
        self.active_profile = None
        
        # Parameter definitions
        self.parameter_definitions = self._define_parameter_schema()
        self.validation_rules = self._define_validation_rules()
        
        # Change tracking
        self.change_history = []
        self.snapshots = []
        self.config_lock = threading.RLock()
        
        # Runtime state
        self.last_save_time = time.time()
        self.unsaved_changes = False
        
        # Initialize configuration directory and load existing configs
        self._initialize_config_directory()
        self._load_existing_configurations()
        
        print(f"🎛️ GA Config Manager initialized with {len(self.profiles)} profiles")
    
    def _define_parameter_schema(self) -> Dict[str, Dict[str, Any]]:
        """Define schema for all GA parameters"""
        return {
            # Population parameters
            'population_size': {
                'type': int,
                'range': (20, 1000),
                'default': 100,
                'description': 'Size of the genetic algorithm population',
                'category': 'population',
                'affects_performance': True
            },
            'elite_size_ratio': {
                'type': float,
                'range': (0.01, 0.5),
                'default': 0.1,
                'description': 'Ratio of elite individuals to preserve',
                'category': 'selection',
                'affects_convergence': True
            },
            
            # Genetic operators
            'mutation_rate': {
                'type': float,
                'range': (0.001, 0.5),
                'default': 0.1,
                'description': 'Probability of mutation for each individual',
                'category': 'operators',
                'adaptive': True
            },
            'crossover_rate': {
                'type': float,
                'range': (0.1, 0.95),
                'default': 0.8,
                'description': 'Probability of crossover between parents',
                'category': 'operators',
                'adaptive': True
            },
            'tournament_size': {
                'type': int,
                'range': (2, 50),
                'default': 5,
                'description': 'Size of tournament for selection',
                'category': 'selection',
                'affects_pressure': True
            },
            
            # Termination criteria
            'max_generations': {
                'type': int,
                'range': (10, 2000),
                'default': 200,
                'description': 'Maximum number of generations',
                'category': 'termination',
                'affects_runtime': True
            },
            'convergence_threshold': {
                'type': float,
                'range': (0.001, 0.1),
                'default': 0.01,
                'description': 'Fitness improvement threshold for convergence',
                'category': 'termination',
                'affects_convergence': True
            },
            'early_stopping_patience': {
                'type': int,
                'range': (5, 200),
                'default': 50,
                'description': 'Generations without improvement before stopping',
                'category': 'termination',
                'affects_runtime': True
            },
            
            # Route-specific parameters
            'distance_tolerance': {
                'type': float,
                'range': (0.01, 1.0),
                'default': 0.2,
                'description': 'Tolerance for target distance matching',
                'category': 'route',
                'objective_specific': True
            },
            'elevation_weight': {
                'type': float,
                'range': (0.0, 5.0),
                'default': 1.0,
                'description': 'Weight for elevation in fitness calculation',
                'category': 'fitness',
                'objective_specific': True
            },
            'diversity_weight': {
                'type': float,
                'range': (0.0, 2.0),
                'default': 0.1,
                'description': 'Weight for route diversity in fitness',
                'category': 'fitness',
                'affects_exploration': True
            },
            
            # Performance parameters
            'parallel_evaluation': {
                'type': bool,
                'default': True,
                'description': 'Enable parallel fitness evaluation',
                'category': 'performance',
                'affects_performance': True
            },
            'cache_enabled': {
                'type': bool,
                'default': True,
                'description': 'Enable fitness and distance caching',
                'category': 'performance',
                'affects_performance': True
            },
            'memory_limit_mb': {
                'type': int,
                'range': (128, 8192),
                'default': 1024,
                'description': 'Memory limit for GA operations',
                'category': 'performance',
                'affects_scalability': True
            }
        }
    
    def _define_validation_rules(self) -> List[ConfigValidationRule]:
        """Define validation rules for parameter combinations"""
        rules = []
        
        # Elite size should be reasonable relative to population size
        rules.append(ConfigValidationRule(
            parameter_name='elite_size_ratio',
            validator=lambda x: True,  # Complex validation in custom function
            error_message='Elite size ratio results in too few or too many elites',
            dependency_params=['population_size']
        ))
        
        # Tournament size should be reasonable relative to population size
        rules.append(ConfigValidationRule(
            parameter_name='tournament_size',
            validator=lambda x: True,  # Complex validation in custom function
            error_message='Tournament size should be smaller than population size',
            dependency_params=['population_size']
        ))
        
        # Memory limit should be sufficient for population size
        rules.append(ConfigValidationRule(
            parameter_name='memory_limit_mb',
            validator=lambda x: x >= 128,
            error_message='Memory limit too low for stable operation',
            dependency_params=['population_size']
        ))
        
        # Convergence threshold should be reasonable
        rules.append(ConfigValidationRule(
            parameter_name='convergence_threshold',
            validator=lambda x: 0.001 <= x <= 0.1,
            error_message='Convergence threshold should be between 0.001 and 0.1',
            warning_only=True
        ))
        
        return rules
    
    def _initialize_config_directory(self):
        """Initialize configuration directory structure"""
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, 'profiles'), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, 'backups'), exist_ok=True)
        os.makedirs(os.path.join(self.config_dir, 'snapshots'), exist_ok=True)
    
    def _load_existing_configurations(self):
        """Load existing configuration files"""
        try:
            # Load global configuration
            global_config_file = os.path.join(self.config_dir, 'global_config.json')
            if os.path.exists(global_config_file):
                with open(global_config_file, 'r') as f:
                    self.configurations[ConfigScope.GLOBAL] = json.load(f)
            else:
                # Initialize with defaults
                self.configurations[ConfigScope.GLOBAL] = self._get_default_parameters()
                self._save_global_configuration()
            
            # Load profiles
            profiles_dir = os.path.join(self.config_dir, 'profiles')
            for filename in os.listdir(profiles_dir):
                if filename.endswith('.json'):
                    profile_path = os.path.join(profiles_dir, filename)
                    with open(profile_path, 'r') as f:
                        profile_data = json.load(f)
                        profile = ConfigProfile(**profile_data)
                        self.profiles[profile.name] = profile
            
            print(f"✅ Loaded configuration: {len(self.profiles)} profiles")
            
        except Exception as e:
            print(f"⚠️ Error loading configurations: {e}")
            self.configurations[ConfigScope.GLOBAL] = self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameter values"""
        return {name: schema['default'] 
                for name, schema in self.parameter_definitions.items()
                if 'default' in schema}
    
    def get_parameter(self, name: str, scope: Optional[ConfigScope] = None) -> Any:
        """Get parameter value with scope precedence
        
        Args:
            name: Parameter name
            scope: Specific scope to check (optional)
            
        Returns:
            Parameter value
        """
        with self.config_lock:
            if scope:
                return self.configurations[scope].get(name)
            
            # Check scopes in order of precedence
            for check_scope in [ConfigScope.RUNTIME, ConfigScope.SESSION, 
                              ConfigScope.PROFILE, ConfigScope.GLOBAL]:
                if name in self.configurations[check_scope]:
                    return self.configurations[check_scope][name]
            
            # Return default if not found
            if name in self.parameter_definitions:
                return self.parameter_definitions[name].get('default')
            
            return None
    
    def set_parameter(self, name: str, value: Any, scope: ConfigScope = ConfigScope.SESSION,
                     reason: str = "manual", source: str = "user") -> bool:
        """Set parameter value in specified scope
        
        Args:
            name: Parameter name
            value: Parameter value
            scope: Configuration scope
            reason: Reason for change
            source: Source of change
            
        Returns:
            True if successful, False otherwise
        """
        with self.config_lock:
            # Validate parameter
            if not self._validate_parameter(name, value, scope):
                return False
            
            # Record old value for change tracking
            old_value = self.get_parameter(name, scope)
            
            # Set parameter
            self.configurations[scope][name] = value
            
            # Record change
            change = ConfigChange(
                parameter_name=name,
                old_value=old_value,
                new_value=value,
                scope=scope,
                timestamp=time.time(),
                reason=reason,
                source=source
            )
            
            self.change_history.append(change)
            if len(self.change_history) > self.config['max_history_size']:
                self.change_history.pop(0)
            
            self.unsaved_changes = True
            
            # Auto-save if enabled
            if self.config['auto_save'] and scope in [ConfigScope.GLOBAL, ConfigScope.PROFILE]:
                self._auto_save()
            
            print(f"🎛️ Parameter updated: {name}={value} (scope: {scope.value}, source: {source})")
            
            return True
    
    def _validate_parameter(self, name: str, value: Any, scope: ConfigScope) -> bool:
        """Validate parameter value"""
        if self.config['validation_level'] == ValidationLevel.NONE:
            return True
        
        # Check if parameter is defined
        if name not in self.parameter_definitions:
            if self.config['allow_parameter_expansion']:
                print(f"⚠️ Adding new parameter: {name}")
                return True
            else:
                print(f"❌ Unknown parameter: {name}")
                return False
        
        schema = self.parameter_definitions[name]
        
        # Type validation
        expected_type = schema.get('type')
        if expected_type and not isinstance(value, expected_type):
            try:
                value = expected_type(value)
            except (ValueError, TypeError):
                print(f"❌ Invalid type for {name}: expected {expected_type.__name__}")
                return False
        
        # Range validation
        if 'range' in schema:
            min_val, max_val = schema['range']
            if not (min_val <= value <= max_val):
                print(f"❌ Value out of range for {name}: {value} not in [{min_val}, {max_val}]")
                return False
        
        # Custom validation rules
        if self.config['validation_level'] in [ValidationLevel.STRICT, ValidationLevel.CUSTOM]:
            for rule in self.validation_rules:
                if rule.parameter_name == name:
                    if not self._validate_with_rule(rule, name, value):
                        if rule.warning_only:
                            print(f"⚠️ {rule.error_message}")
                        else:
                            print(f"❌ {rule.error_message}")
                            return False
        
        return True
    
    def _validate_with_rule(self, rule: ConfigValidationRule, name: str, value: Any) -> bool:
        """Apply custom validation rule"""
        try:
            # Get dependency values if needed
            if rule.dependency_params:
                dep_values = {dep: self.get_parameter(dep) for dep in rule.dependency_params}
                
                # Special validations based on dependencies
                if name == 'elite_size_ratio' and 'population_size' in dep_values:
                    pop_size = dep_values['population_size']
                    elite_count = int(value * pop_size)
                    return 1 <= elite_count <= pop_size // 2
                
                elif name == 'tournament_size' and 'population_size' in dep_values:
                    pop_size = dep_values['population_size']
                    return 2 <= value <= min(pop_size, 50)
            
            # Apply rule's validator function
            return rule.validator(value)
            
        except Exception as e:
            print(f"⚠️ Validation error for {name}: {e}")
            return False
    
    def create_profile(self, name: str, description: str, 
                      parameters: Optional[Dict[str, Any]] = None,
                      objective: Optional[RouteObjective] = None,
                      target_distance_range: Optional[tuple] = None,
                      tags: Optional[List[str]] = None) -> ConfigProfile:
        """Create new configuration profile
        
        Args:
            name: Profile name
            description: Profile description
            parameters: Profile parameters (uses current session if None)
            objective: Target optimization objective
            target_distance_range: Distance range for auto-selection
            tags: Profile tags
            
        Returns:
            Created profile
        """
        with self.config_lock:
            if parameters is None:
                # Use current effective configuration
                parameters = self.get_effective_configuration()
            
            profile = ConfigProfile(
                name=name,
                description=description,
                parameters=parameters.copy(),
                objective=objective,
                target_distance_range=target_distance_range,
                tags=tags or []
            )
            
            self.profiles[name] = profile
            
            # Save profile
            if self.config['enable_profiles']:
                self._save_profile(profile)
            
            print(f"🎛️ Created profile: {name}")
            
            return profile
    
    def activate_profile(self, name: str) -> bool:
        """Activate configuration profile
        
        Args:
            name: Profile name
            
        Returns:
            True if successful, False otherwise
        """
        with self.config_lock:
            if name not in self.profiles:
                print(f"❌ Profile not found: {name}")
                return False
            
            profile = self.profiles[name]
            
            # Clear current profile scope
            self.configurations[ConfigScope.PROFILE].clear()
            
            # Apply profile parameters
            if self.config['parameter_inheritance']:
                for param_name, param_value in profile.parameters.items():
                    self.configurations[ConfigScope.PROFILE][param_name] = param_value
            
            self.active_profile = name
            profile.last_used_timestamp = time.time()
            profile.usage_count += 1
            
            print(f"🎛️ Activated profile: {name}")
            
            return True
    
    def auto_select_profile(self, objective: RouteObjective, target_distance: float) -> Optional[str]:
        """Automatically select best profile for given criteria
        
        Args:
            objective: Optimization objective
            target_distance: Target route distance
            
        Returns:
            Selected profile name or None
        """
        if not self.config['profile_auto_selection']:
            return None
        
        best_profile = None
        best_score = -1
        
        for name, profile in self.profiles.items():
            score = 0
            
            # Objective match
            if profile.objective == objective:
                score += 10
            elif profile.objective is None:
                score += 2  # Generic profile
            
            # Distance range match
            if profile.target_distance_range:
                min_dist, max_dist = profile.target_distance_range
                if min_dist <= target_distance <= max_dist:
                    score += 5
                else:
                    # Penalty for distance mismatch
                    distance_diff = min(abs(target_distance - min_dist), 
                                      abs(target_distance - max_dist))
                    score -= distance_diff * 0.1
            
            # Usage frequency bonus
            score += min(profile.usage_count * 0.1, 2)
            
            # Recent usage bonus
            age_hours = (time.time() - profile.last_used_timestamp) / 3600
            if age_hours < 24:
                score += 1
            
            if score > best_score:
                best_score = score
                best_profile = name
        
        if best_profile and best_score > 5:  # Minimum confidence threshold
            print(f"🎛️ Auto-selected profile: {best_profile} (score: {best_score:.1f})")
            return best_profile
        
        return None
    
    def get_effective_configuration(self) -> Dict[str, Any]:
        """Get effective configuration with scope precedence"""
        with self.config_lock:
            effective_config = {}
            
            # Start with global defaults
            effective_config.update(self.configurations[ConfigScope.GLOBAL])
            
            # Apply profile overrides
            effective_config.update(self.configurations[ConfigScope.PROFILE])
            
            # Apply session overrides
            effective_config.update(self.configurations[ConfigScope.SESSION])
            
            # Apply runtime overrides
            effective_config.update(self.configurations[ConfigScope.RUNTIME])
            
            return effective_config
    
    def take_snapshot(self, generation: Optional[int] = None, 
                     performance_metrics: Optional[Dict[str, float]] = None) -> ConfigSnapshot:
        """Take configuration snapshot
        
        Args:
            generation: Current generation number
            performance_metrics: Performance metrics to include
            
        Returns:
            Configuration snapshot
        """
        with self.config_lock:
            snapshot = ConfigSnapshot(
                timestamp=time.time(),
                scope=ConfigScope.RUNTIME,
                parameters=self.get_effective_configuration(),
                profile_name=self.active_profile,
                generation=generation,
                performance_metrics=performance_metrics or {}
            )
            
            self.snapshots.append(snapshot)
            
            # Limit snapshot history
            if len(self.snapshots) > 100:
                self.snapshots.pop(0)
            
            return snapshot
    
    def get_configuration_recommendations(self) -> Dict[str, Any]:
        """Generate configuration recommendations based on usage patterns"""
        recommendations = {
            'parameter_adjustments': {},
            'profile_suggestions': [],
            'optimization_tips': [],
            'usage_insights': {}
        }
        
        if not self.change_history:
            recommendations['optimization_tips'].append("No configuration changes recorded yet")
            return recommendations
        
        # Analyze parameter change patterns
        param_changes = defaultdict(list)
        for change in self.change_history[-100:]:  # Recent changes
            param_changes[change.parameter_name].append(change)
        
        # Generate parameter adjustment recommendations
        for param_name, changes in param_changes.items():
            if len(changes) >= 3:  # Parameter changed frequently
                recent_values = [c.new_value for c in changes[-5:]]
                avg_value = sum(recent_values) / len(recent_values)
                current_value = self.get_parameter(param_name)
                
                if abs(avg_value - current_value) > 0.1:
                    recommendations['parameter_adjustments'][param_name] = {
                        'current': current_value,
                        'suggested': avg_value,
                        'reason': f'Parameter changed {len(changes)} times recently'
                    }
        
        # Profile usage insights
        if self.profiles:
            most_used = max(self.profiles.values(), key=lambda p: p.usage_count)
            recommendations['usage_insights']['most_used_profile'] = {
                'name': most_used.name,
                'usage_count': most_used.usage_count,
                'description': most_used.description
            }
        
        # Optimization tips based on change frequency
        high_change_params = [name for name, changes in param_changes.items() 
                             if len(changes) > 5]
        if high_change_params:
            recommendations['optimization_tips'].append(
                f"Consider creating a profile for frequently changed parameters: {', '.join(high_change_params)}"
            )
        
        return recommendations
    
    def _save_global_configuration(self):
        """Save global configuration to file"""
        try:
            config_file = os.path.join(self.config_dir, 'global_config.json')
            with open(config_file, 'w') as f:
                json.dump(self.configurations[ConfigScope.GLOBAL], f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving global configuration: {e}")
    
    def _save_profile(self, profile: ConfigProfile):
        """Save profile to file"""
        try:
            profile_file = os.path.join(self.config_dir, 'profiles', f'{profile.name}.json')
            with open(profile_file, 'w') as f:
                json.dump(asdict(profile), f, indent=2, default=str)
        except Exception as e:
            print(f"⚠️ Error saving profile {profile.name}: {e}")
    
    def _auto_save(self):
        """Auto-save unsaved changes"""
        current_time = time.time()
        if (self.unsaved_changes and 
            current_time - self.last_save_time > self.config['save_interval_seconds']):
            
            self._save_global_configuration()
            self.last_save_time = current_time
            self.unsaved_changes = False
    
    def export_configuration(self, filename: str, include_profiles: bool = True) -> str:
        """Export complete configuration to file
        
        Args:
            filename: Output filename
            include_profiles: Include all profiles in export
            
        Returns:
            Path to exported file
        """
        export_data = {
            'global_config': self.configurations[ConfigScope.GLOBAL],
            'active_profile': self.active_profile,
            'parameter_definitions': self.parameter_definitions,
            'export_timestamp': time.time()
        }
        
        if include_profiles:
            export_data['profiles'] = {name: asdict(profile) 
                                     for name, profile in self.profiles.items()}
        
        with open(filename, 'w') as f:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                yaml.dump(export_data, f, indent=2, default_flow_style=False)
            else:
                json.dump(export_data, f, indent=2, default=str)
        
        return filename
    
    def reset_scope(self, scope: ConfigScope):
        """Reset specific configuration scope"""
        with self.config_lock:
            self.configurations[scope].clear()
            if scope == ConfigScope.PROFILE:
                self.active_profile = None
            
            print(f"🎛️ Reset configuration scope: {scope.value}")


def test_config_manager():
    """Test function for configuration manager"""
    print("Testing GA Configuration Manager...")
    
    # Create config manager
    manager = GAConfigManager("test_config")
    
    # Test parameter operations
    manager.set_parameter('population_size', 150, ConfigScope.SESSION)
    pop_size = manager.get_parameter('population_size')
    print(f"✅ Parameter operations: population_size = {pop_size}")
    
    # Test profile creation
    profile = manager.create_profile(
        name="test_elevation_profile",
        description="Profile for elevation optimization",
        parameters={'mutation_rate': 0.15, 'elevation_weight': 2.0},
        objective=RouteObjective.MAXIMIZE_ELEVATION,
        target_distance_range=(3.0, 8.0)
    )
    print(f"✅ Profile created: {profile.name}")
    
    # Test profile activation
    success = manager.activate_profile("test_elevation_profile")
    print(f"✅ Profile activation: {success}")
    
    # Test auto-selection
    auto_profile = manager.auto_select_profile(RouteObjective.MAXIMIZE_ELEVATION, 5.0)
    print(f"✅ Auto-selected profile: {auto_profile}")
    
    # Test effective configuration
    effective_config = manager.get_effective_configuration()
    print(f"✅ Effective configuration: {len(effective_config)} parameters")
    
    # Test snapshot
    snapshot = manager.take_snapshot(generation=10, performance_metrics={'fitness': 0.85})
    print(f"✅ Snapshot taken: generation {snapshot.generation}")
    
    # Test recommendations
    recommendations = manager.get_configuration_recommendations()
    print(f"✅ Recommendations: {len(recommendations)} categories")
    
    print("✅ All configuration manager tests completed")


if __name__ == "__main__":
    test_config_manager()
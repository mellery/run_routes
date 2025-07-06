#!/usr/bin/env python3
"""
Demo script to showcase Week 3 Evolution Engine features
Generates visualizations showing the key components implemented
"""

import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from route_services.network_manager import NetworkManager
from genetic_route_optimizer import GeneticRouteOptimizer, GAConfig
from ga_fitness import GAFitnessEvaluator
from ga_visualizer import GAVisualizer

def demo_week3_features():
    """Demonstrate Week 3 evolution engine features with visualizations"""
    print("🧬 Week 3 Evolution Engine Demo")
    print("=" * 50)
    
    # Setup
    output_dir = "week3_demo_images"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load network
    print("🌐 Loading network...")
    network_manager = NetworkManager()
    graph = network_manager.load_network()
    
    if not graph:
        print("❌ Failed to load network")
        return
    
    print(f"✅ Network loaded: {len(graph.nodes)} nodes")
    
    # Initialize visualizer
    visualizer = GAVisualizer(graph, output_dir)
    
    # Demo 1: Multi-Objective Fitness Evaluation
    print("\n📊 Demo 1: Multi-Objective Fitness Evaluation")
    demo_fitness_objectives(graph, output_dir, timestamp)
    
    # Demo 2: Evolution with Different Objectives  
    print("\n🧬 Demo 2: Evolution Process with Different Objectives")
    demo_evolution_process(graph, visualizer, output_dir, timestamp)
    
    # Demo 3: Convergence Detection
    print("\n🎯 Demo 3: Convergence Detection and Early Stopping")
    demo_convergence_detection(graph, output_dir, timestamp)
    
    print(f"\n✅ All Week 3 demos completed!")
    print(f"📁 Visualizations saved to: {output_dir}")

def demo_fitness_objectives(graph, output_dir, timestamp):
    """Demo multi-objective fitness evaluation"""
    from ga_population import PopulationInitializer
    
    # Create test population
    start_node = NetworkManager.DEFAULT_START_NODE
    initializer = PopulationInitializer(graph, start_node)
    population = initializer.create_population(10, 3.0)
    
    if not population:
        print("  ❌ Failed to create test population")
        return
    
    # Test different objectives
    objectives = ["distance", "elevation", "balanced", "scenic", "efficiency"]
    objective_results = {}
    
    for objective in objectives:
        evaluator = GAFitnessEvaluator(objective, 3.0)
        fitness_scores = []
        
        for chromosome in population:
            fitness = evaluator.evaluate_chromosome(chromosome)
            fitness_scores.append(fitness)
        
        objective_results[objective] = {
            'fitness_scores': fitness_scores,
            'best_fitness': max(fitness_scores),
            'avg_fitness': sum(fitness_scores) / len(fitness_scores)
        }
        
        print(f"  ✅ {objective.title()}: Best={max(fitness_scores):.3f}, Avg={sum(fitness_scores)/len(fitness_scores):.3f}")
    
    # Create fitness comparison visualization
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Fitness distribution by objective
    plt.subplot(2, 2, 1)
    objectives_list = list(objective_results.keys())
    best_scores = [objective_results[obj]['best_fitness'] for obj in objectives_list]
    avg_scores = [objective_results[obj]['avg_fitness'] for obj in objectives_list]
    
    x = range(len(objectives_list))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], best_scores, width, label='Best Fitness', alpha=0.8)
    plt.bar([i + width/2 for i in x], avg_scores, width, label='Average Fitness', alpha=0.8)
    
    plt.xlabel('Optimization Objective')
    plt.ylabel('Fitness Score')
    plt.title('Fitness Scores by Objective')
    plt.xticks(x, [obj.title() for obj in objectives_list], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Fitness distribution histogram for elevation objective
    plt.subplot(2, 2, 2)
    elevation_scores = objective_results['elevation']['fitness_scores']
    plt.hist(elevation_scores, bins=8, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Fitness Score')
    plt.ylabel('Number of Routes')
    plt.title('Elevation Objective - Fitness Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Objective weights comparison
    plt.subplot(2, 2, 3)
    sample_evaluator = GAFitnessEvaluator("elevation", 3.0)
    weights = sample_evaluator.weights
    
    weight_names = list(weights.keys())
    weight_values = list(weights.values())
    
    plt.pie(weight_values, labels=[name.replace('_', ' ').title() for name in weight_names], 
            autopct='%1.1f%%', startangle=90)
    plt.title('Elevation Objective - Weight Distribution')
    
    # Plot 4: Multi-objective comparison radar chart (simplified)
    plt.subplot(2, 2, 4)
    for i, objective in enumerate(objectives_list):
        fitness_scores = objective_results[objective]['fitness_scores']
        plt.plot(range(len(fitness_scores)), sorted(fitness_scores, reverse=True), 
                'o-', label=objective.title(), alpha=0.7)
    
    plt.xlabel('Route Rank')
    plt.ylabel('Fitness Score')
    plt.title('Route Ranking by Objective')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"week3_fitness_objectives_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Saved fitness objectives demo: {filename}")

def demo_evolution_process(graph, visualizer, output_dir, timestamp):
    """Demo evolution process with visualization"""
    
    # Configure for quick demo
    config = GAConfig(
        population_size=15,
        max_generations=20,
        crossover_rate=0.8,
        mutation_rate=0.15,
        verbose=False
    )
    
    # Test two different objectives
    objectives = ["elevation", "distance"]
    evolution_results = {}
    
    for objective in objectives:
        print(f"  🔄 Testing {objective} objective...")
        
        optimizer = GeneticRouteOptimizer(graph, config)
        
        start_time = time.time()
        results = optimizer.optimize_route(
            NetworkManager.DEFAULT_START_NODE, 
            3.0, 
            objective
        )
        evolution_time = time.time() - start_time
        
        evolution_results[objective] = {
            'results': results,
            'time': evolution_time
        }
        
        print(f"    ✅ {objective.title()}: Best fitness={results.best_fitness:.3f}, "
              f"Generations={results.total_generations}, Time={evolution_time:.1f}s")
        
        # Save best route visualization
        best_route_filename = f"week3_best_route_{objective}_{timestamp}.png"
        visualizer.save_chromosome_map(
            results.best_chromosome, 
            best_route_filename,
            title=f"Best Route - {objective.title()} Objective (Fitness: {results.best_fitness:.3f})",
            show_elevation=True,
            show_segments=True
        )
        print(f"    📸 Saved best route: {best_route_filename}")
    
    # Create evolution comparison visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Fitness evolution over generations
    plt.subplot(2, 3, 1)
    for objective in objectives:
        results = evolution_results[objective]['results']
        generations = list(range(len(results.fitness_history)))
        best_fitness = [max(gen_fitness) for gen_fitness in results.fitness_history]
        
        plt.plot(generations, best_fitness, 'o-', label=f'{objective.title()} (Best)', linewidth=2, markersize=4)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Evolution Progress - Best Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Average fitness evolution
    plt.subplot(2, 3, 2)
    for objective in objectives:
        results = evolution_results[objective]['results']
        generations = list(range(len(results.fitness_history)))
        avg_fitness = [sum(gen_fitness)/len(gen_fitness) for gen_fitness in results.fitness_history]
        
        plt.plot(generations, avg_fitness, 's-', label=f'{objective.title()} (Avg)', linewidth=2, markersize=4)
    
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.title('Evolution Progress - Average Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Convergence comparison
    plt.subplot(2, 3, 3)
    convergence_data = []
    labels = []
    
    for objective in objectives:
        results = evolution_results[objective]['results']
        convergence_data.append(results.total_generations)
        labels.append(f"{objective.title()}\n({results.convergence_reason})")
    
    bars = plt.bar(labels, convergence_data, color=['green', 'blue'], alpha=0.7)
    plt.ylabel('Generations to Converge')
    plt.title('Convergence Speed Comparison')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, convergence_data):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(value), ha='center', va='bottom')
    
    # Plot 4: Performance statistics
    plt.subplot(2, 3, 4)
    metrics = ['Best Fitness', 'Final Generations', 'Time (s)']
    elevation_stats = [
        evolution_results['elevation']['results'].best_fitness,
        evolution_results['elevation']['results'].total_generations,
        evolution_results['elevation']['time']
    ]
    distance_stats = [
        evolution_results['distance']['results'].best_fitness,
        evolution_results['distance']['results'].total_generations,
        evolution_results['distance']['time']
    ]
    
    x = range(len(metrics))
    width = 0.35
    
    # Normalize values for comparison
    max_vals = [max(elevation_stats[i], distance_stats[i]) for i in range(len(metrics))]
    elevation_norm = [elevation_stats[i]/max_vals[i] for i in range(len(metrics))]
    distance_norm = [distance_stats[i]/max_vals[i] for i in range(len(metrics))]
    
    plt.bar([i - width/2 for i in x], elevation_norm, width, label='Elevation', alpha=0.8)
    plt.bar([i + width/2 for i in x], distance_norm, width, label='Distance', alpha=0.8)
    
    plt.xlabel('Metric')
    plt.ylabel('Normalized Value')
    plt.title('Performance Comparison (Normalized)')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Population diversity over time (for elevation objective)
    plt.subplot(2, 3, 5)
    elevation_results = evolution_results['elevation']['results']
    diversity_scores = []
    
    for gen_pop in elevation_results.population_history:
        distances = [chromo.get_total_distance() for chromo in gen_pop if chromo.segments]
        if distances:
            import numpy as np
            diversity = np.std(distances) / max(np.mean(distances), 1.0)
            diversity_scores.append(diversity)
        else:
            diversity_scores.append(0.0)
    
    generations = list(range(len(diversity_scores)))
    plt.plot(generations, diversity_scores, 'r-', linewidth=2, label='Population Diversity')
    plt.xlabel('Generation')
    plt.ylabel('Diversity Score')
    plt.title('Population Diversity Evolution')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot 6: Adaptive configuration showcase
    plt.subplot(2, 3, 6)
    config_info = [
        f"Population Size: {config.population_size}",
        f"Max Generations: {config.max_generations}",
        f"Crossover Rate: {config.crossover_rate}",
        f"Mutation Rate: {config.mutation_rate}",
        f"Elite Size: {config.elite_size}",
        f"Adaptive: {config.adaptive_sizing}"
    ]
    
    plt.text(0.05, 0.95, '\n'.join(config_info), transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.title('GA Configuration')
    plt.axis('off')
    
    plt.tight_layout()
    filename = f"week3_evolution_process_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Saved evolution process demo: {filename}")

def demo_convergence_detection(graph, output_dir, timestamp):
    """Demo convergence detection capabilities"""
    
    # Create fitness evaluator for testing convergence
    evaluator = GAFitnessEvaluator("elevation", 3.0)
    
    # Simulate different convergence scenarios
    scenarios = {
        'Fast Convergence': [0.3, 0.5, 0.7, 0.8, 0.85, 0.87, 0.87, 0.87, 0.87, 0.87],
        'Slow Improvement': [0.4, 0.45, 0.48, 0.5, 0.51, 0.52, 0.525, 0.53, 0.535, 0.54],
        'Oscillating': [0.3, 0.6, 0.4, 0.7, 0.45, 0.75, 0.5, 0.8, 0.55, 0.82],
        'Continuous Improvement': [0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    }
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Convergence scenarios
    plt.subplot(2, 2, 1)
    for scenario_name, fitness_values in scenarios.items():
        generations = list(range(len(fitness_values)))
        plt.plot(generations, fitness_values, 'o-', label=scenario_name, linewidth=2, markersize=4)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Different Convergence Patterns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Convergence detection analysis
    plt.subplot(2, 2, 2)
    convergence_detected = []
    scenario_names = []
    
    for scenario_name, fitness_values in scenarios.items():
        # Simulate convergence detection
        evaluator.fitness_history = fitness_values * 3  # Repeat to get enough history
        is_converged = evaluator.is_fitness_plateau(generations=8, threshold=0.01)
        
        convergence_detected.append(1 if is_converged else 0)
        scenario_names.append(scenario_name)
        
        print(f"  📊 {scenario_name}: {'Converged' if is_converged else 'Still Evolving'}")
    
    colors = ['red' if detected else 'green' for detected in convergence_detected]
    bars = plt.bar(scenario_names, convergence_detected, color=colors, alpha=0.7)
    plt.ylabel('Convergence Detected')
    plt.title('Convergence Detection Results')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.2)
    
    # Add labels
    for bar, detected in zip(bars, convergence_detected):
        label = 'Converged' if detected else 'Evolving'
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                label, ha='center', va='bottom')
    
    # Plot 3: Improvement rate analysis
    plt.subplot(2, 2, 3)
    for scenario_name, fitness_values in scenarios.items():
        improvements = [fitness_values[i] - fitness_values[i-1] for i in range(1, len(fitness_values))]
        generations = list(range(1, len(fitness_values)))
        plt.plot(generations, improvements, 'o-', label=scenario_name, linewidth=2, markersize=3)
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness Improvement')
    plt.title('Generation-to-Generation Improvement')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Convergence criteria visualization
    plt.subplot(2, 2, 4)
    criteria_info = [
        "Convergence Detection Criteria:",
        "",
        "• Plateau Threshold: 0.001",
        "• Analysis Window: 10 generations", 
        "• Recent Improvement < Threshold",
        "",
        "Early Stopping Benefits:",
        "• Saves computation time",
        "• Prevents overfitting",
        "• Identifies optimal solutions",
        "",
        "Adaptive Configuration:",
        "• Population size scales with problem",
        "• Generation limits adjust automatically",
        "• Tournament size adapts to population"
    ]
    
    plt.text(0.05, 0.95, '\n'.join(criteria_info), transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.title('Convergence & Adaptation Features')
    plt.axis('off')
    
    plt.tight_layout()
    filename = f"week3_convergence_detection_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 Saved convergence detection demo: {filename}")

if __name__ == "__main__":
    demo_week3_features()
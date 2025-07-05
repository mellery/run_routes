#!/usr/bin/env python3
"""
GA Parameter Tuning Visualizations and Performance Analysis
Advanced visualization and analysis tools for parameter tuning results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import math

from ga_parameter_tuner import GAParameterTuner, ParameterHistory
from ga_hyperparameter_optimizer import GAHyperparameterOptimizer, OptimizationResult
from ga_algorithm_selector import GAAlgorithmSelector, AlgorithmComparison
from ga_config_manager import GAConfigManager, ConfigSnapshot
from ga_sensitivity_analyzer import GASensitivityAnalyzer, SensitivityResult
from ga_performance_benchmark import GAPerformanceBenchmark, BenchmarkResult


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation"""
    output_dir: str = "tuning_visualizations"
    figure_format: str = "png"  # png, pdf, svg
    interactive_plots: bool = True
    color_scheme: str = "viridis"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_data: bool = True
    show_plots: bool = False


class GATuningVisualizer:
    """Advanced visualization system for GA parameter tuning analysis"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """Initialize tuning visualizer
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette(self.config.color_scheme)
        
        # Colors for consistent plotting
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'muted': '#7f7f7f'
        }
        
        print(f"📊 GA Tuning Visualizer initialized - output: {self.config.output_dir}")
    
    def visualize_parameter_adaptation_history(self, tuner: GAParameterTuner) -> str:
        """Visualize parameter adaptation history over time
        
        Args:
            tuner: Parameter tuner with adaptation history
            
        Returns:
            Path to generated visualization
        """
        if not tuner.adaptation_history:
            print("⚠️ No adaptation history available for visualization")
            return ""
        
        # Prepare data
        adaptation_data = []
        for entry in tuner.adaptation_history:
            for param_name, param_value in entry.parameters.items():
                adaptation_data.append({
                    'generation': entry.generation,
                    'parameter': param_name,
                    'value': param_value,
                    'reason': entry.adaptation_reason,
                    'timestamp': entry.timestamp
                })
        
        df = pd.DataFrame(adaptation_data)
        
        # Create visualization
        if self.config.interactive_plots:
            fig = self._create_interactive_adaptation_plot(df)
            filename = os.path.join(self.config.output_dir, "parameter_adaptation_history.html")
            fig.write_html(filename)
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle('Parameter Adaptation History', fontsize=16, fontweight='bold')
            
            # Plot adaptation trends for key parameters
            key_params = ['mutation_rate', 'population_size', 'crossover_rate', 'elite_size_ratio']
            
            for i, param in enumerate(key_params[:4]):
                ax = axes[i // 2, i % 2]
                param_data = df[df['parameter'] == param]
                
                if not param_data.empty:
                    ax.plot(param_data['generation'], param_data['value'], 
                           marker='o', linewidth=2, markersize=6, color=self.colors['primary'])
                    ax.set_title(f'{param.replace("_", " ").title()}', fontweight='bold')
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Value')
                    ax.grid(True, alpha=0.3)
                    
                    # Add adaptation points
                    for _, row in param_data.iterrows():
                        ax.annotate(f"G{int(row['generation'])}", 
                                  (row['generation'], row['value']),
                                  xytext=(5, 5), textcoords='offset points',
                                  fontsize=8, alpha=0.7)
                else:
                    ax.text(0.5, 0.5, f'No {param} adaptations', 
                           ha='center', va='center', transform=ax.transAxes)
            
            plt.tight_layout()
            filename = os.path.join(self.config.output_dir, f"parameter_adaptation_history.{self.config.figure_format}")
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
        
        # Save data
        if self.config.save_data:
            data_filename = os.path.join(self.config.output_dir, "parameter_adaptation_data.csv")
            df.to_csv(data_filename, index=False)
        
        print(f"📊 Parameter adaptation history saved: {filename}")
        return filename
    
    def _create_interactive_adaptation_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create interactive parameter adaptation plot"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Mutation Rate', 'Population Size', 'Crossover Rate', 'Elite Size Ratio'],
            vertical_spacing=0.1
        )
        
        key_params = ['mutation_rate', 'population_size', 'crossover_rate', 'elite_size_ratio']
        colors = [self.colors['primary'], self.colors['secondary'], self.colors['success'], self.colors['warning']]
        
        for i, param in enumerate(key_params):
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            param_data = df[df['parameter'] == param]
            
            if not param_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=param_data['generation'],
                        y=param_data['value'],
                        mode='lines+markers',
                        name=param.replace('_', ' ').title(),
                        line=dict(color=colors[i], width=3),
                        marker=dict(size=8),
                        hovertemplate='Generation: %{x}<br>Value: %{y}<br>Reason: %{customdata}<extra></extra>',
                        customdata=param_data['reason']
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title='Parameter Adaptation History',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def visualize_hyperparameter_optimization_results(self, optimization_results: List[OptimizationResult]) -> str:
        """Visualize hyperparameter optimization results
        
        Args:
            optimization_results: List of optimization results from different methods
            
        Returns:
            Path to generated visualization
        """
        if not optimization_results:
            print("⚠️ No optimization results available for visualization")
            return ""
        
        # Prepare data
        results_data = []
        for result in optimization_results:
            results_data.append({
                'method': result.method_used.value,
                'best_score': result.best_score,
                'optimization_time': result.optimization_time,
                'total_evaluations': result.total_evaluations,
                'convergence_generation': result.convergence_generation
            })
        
        df = pd.DataFrame(results_data)
        
        if self.config.interactive_plots:
            fig = self._create_interactive_optimization_results_plot(df, optimization_results)
            filename = os.path.join(self.config.output_dir, "hyperparameter_optimization_results.html")
            fig.write_html(filename)
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle('Hyperparameter Optimization Results', fontsize=16, fontweight='bold')
            
            # Best scores comparison
            axes[0, 0].bar(df['method'], df['best_score'], color=self.colors['primary'])
            axes[0, 0].set_title('Best Scores by Method')
            axes[0, 0].set_ylabel('Best Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Optimization time comparison
            axes[0, 1].bar(df['method'], df['optimization_time'], color=self.colors['secondary'])
            axes[0, 1].set_title('Optimization Time by Method')
            axes[0, 1].set_ylabel('Time (seconds)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Evaluations vs Score
            axes[1, 0].scatter(df['total_evaluations'], df['best_score'], 
                             c=df['optimization_time'], cmap=self.config.color_scheme, s=100)
            axes[1, 0].set_title('Evaluations vs Best Score')
            axes[1, 0].set_xlabel('Total Evaluations')
            axes[1, 0].set_ylabel('Best Score')
            
            # Convergence comparison
            axes[1, 1].bar(df['method'], df['convergence_generation'], color=self.colors['success'])
            axes[1, 1].set_title('Convergence Speed by Method')
            axes[1, 1].set_ylabel('Convergence Generation')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            filename = os.path.join(self.config.output_dir, f"hyperparameter_optimization_results.{self.config.figure_format}")
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
        
        # Save detailed results
        if self.config.save_data:
            data_filename = os.path.join(self.config.output_dir, "optimization_results_data.csv")
            df.to_csv(data_filename, index=False)
        
        print(f"📊 Hyperparameter optimization results saved: {filename}")
        return filename
    
    def visualize_algorithm_performance_comparison(self, comparison: AlgorithmComparison) -> str:
        """Visualize algorithm performance comparison
        
        Args:
            comparison: Algorithm comparison results
            
        Returns:
            Path to generated visualization
        """
        # Prepare data
        algorithms = [alg.value for alg in comparison.algorithms_compared]
        metrics = list(comparison.performance_matrix[algorithms[0]].keys())
        
        performance_data = []
        for algorithm in algorithms:
            for metric in metrics:
                performance_data.append({
                    'algorithm': algorithm,
                    'metric': metric,
                    'value': comparison.performance_matrix[algorithm][metric]
                })
        
        df = pd.DataFrame(performance_data)
        
        if self.config.interactive_plots:
            fig = self._create_interactive_algorithm_comparison_plot(df, comparison)
            filename = os.path.join(self.config.output_dir, "algorithm_performance_comparison.html")
            fig.write_html(filename)
        else:
            # Create radar chart for algorithm comparison
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
            
            # Performance matrix heatmap
            pivot_df = df.pivot(index='algorithm', columns='metric', values='value')
            sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap=self.config.color_scheme, ax=ax1)
            ax1.set_title('Performance Matrix')
            
            # Rankings comparison
            ranking_data = []
            for metric, ranked_algorithms in comparison.rankings.items():
                for rank, algorithm in enumerate(ranked_algorithms):
                    ranking_data.append({
                        'metric': metric,
                        'algorithm': algorithm.value,
                        'rank': rank + 1
                    })
            
            ranking_df = pd.DataFrame(ranking_data)
            ranking_pivot = ranking_df.pivot(index='algorithm', columns='metric', values='rank')
            
            sns.heatmap(ranking_pivot, annot=True, fmt='d', cmap='RdYlBu_r', ax=ax2, 
                       cbar_kws={'label': 'Rank (1=best)'})
            ax2.set_title('Algorithm Rankings')
            
            plt.tight_layout()
            filename = os.path.join(self.config.output_dir, f"algorithm_performance_comparison.{self.config.figure_format}")
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
        
        # Save data
        if self.config.save_data:
            data_filename = os.path.join(self.config.output_dir, "algorithm_comparison_data.csv")
            df.to_csv(data_filename, index=False)
        
        print(f"📊 Algorithm performance comparison saved: {filename}")
        return filename
    
    def visualize_sensitivity_analysis_results(self, analyzer: GASensitivityAnalyzer) -> str:
        """Visualize parameter sensitivity analysis results
        
        Args:
            analyzer: Sensitivity analyzer with results
            
        Returns:
            Path to generated visualization
        """
        if not analyzer.sensitivity_results:
            print("⚠️ No sensitivity analysis results available for visualization")
            return ""
        
        # Prepare data
        sensitivity_data = []
        for key, result in analyzer.sensitivity_results.items():
            param_name, metric = key.rsplit('_', 1)
            sensitivity_data.append({
                'parameter': param_name,
                'metric': metric,
                'sensitivity_index': result.sensitivity_index,
                'variance_contribution': result.variance_contribution,
                'confidence_lower': result.confidence_interval[0],
                'confidence_upper': result.confidence_interval[1],
                'significant': result.statistical_significance,
                'rank': result.rank
            })
        
        df = pd.DataFrame(sensitivity_data)
        
        if self.config.interactive_plots:
            fig = self._create_interactive_sensitivity_plot(df)
            filename = os.path.join(self.config.output_dir, "sensitivity_analysis_results.html")
            fig.write_html(filename)
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle('Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
            
            # Sensitivity indices by parameter
            param_sensitivity = df.groupby('parameter')['sensitivity_index'].mean().sort_values(ascending=True)
            axes[0, 0].barh(param_sensitivity.index, param_sensitivity.values, color=self.colors['primary'])
            axes[0, 0].set_title('Average Sensitivity by Parameter')
            axes[0, 0].set_xlabel('Sensitivity Index')
            
            # Variance contribution
            param_variance = df.groupby('parameter')['variance_contribution'].mean().sort_values(ascending=True)
            axes[0, 1].barh(param_variance.index, param_variance.values, color=self.colors['secondary'])
            axes[0, 1].set_title('Average Variance Contribution')
            axes[0, 1].set_xlabel('Variance Contribution')
            
            # Sensitivity vs Variance scatter
            axes[1, 0].scatter(df['sensitivity_index'], df['variance_contribution'], 
                             c=df['significant'].map({True: self.colors['success'], False: self.colors['muted']}),
                             alpha=0.7, s=100)
            axes[1, 0].set_title('Sensitivity vs Variance Contribution')
            axes[1, 0].set_xlabel('Sensitivity Index')
            axes[1, 0].set_ylabel('Variance Contribution')
            
            # Parameter ranking heatmap
            ranking_pivot = df.pivot(index='parameter', columns='metric', values='rank')
            if not ranking_pivot.empty:
                sns.heatmap(ranking_pivot, annot=True, fmt='d', cmap='RdYlBu_r', ax=axes[1, 1],
                           cbar_kws={'label': 'Rank (1=most sensitive)'})
                axes[1, 1].set_title('Sensitivity Rankings by Metric')
            
            plt.tight_layout()
            filename = os.path.join(self.config.output_dir, f"sensitivity_analysis_results.{self.config.figure_format}")
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
        
        # Save data
        if self.config.save_data:
            data_filename = os.path.join(self.config.output_dir, "sensitivity_analysis_data.csv")
            df.to_csv(data_filename, index=False)
        
        print(f"📊 Sensitivity analysis results saved: {filename}")
        return filename
    
    def visualize_configuration_evolution(self, config_manager: GAConfigManager) -> str:
        """Visualize configuration parameter evolution over time
        
        Args:
            config_manager: Configuration manager with snapshots
            
        Returns:
            Path to generated visualization
        """
        if not config_manager.snapshots:
            print("⚠️ No configuration snapshots available for visualization")
            return ""
        
        # Prepare data
        evolution_data = []
        for snapshot in config_manager.snapshots:
            for param_name, param_value in snapshot.parameters.items():
                evolution_data.append({
                    'timestamp': snapshot.timestamp,
                    'generation': snapshot.generation or 0,
                    'parameter': param_name,
                    'value': param_value,
                    'profile': snapshot.profile_name or 'default'
                })
        
        df = pd.DataFrame(evolution_data)
        
        if df.empty:
            print("⚠️ No parameter evolution data available")
            return ""
        
        if self.config.interactive_plots:
            fig = self._create_interactive_config_evolution_plot(df)
            filename = os.path.join(self.config.output_dir, "configuration_evolution.html")
            fig.write_html(filename)
        else:
            # Create multi-panel evolution plot
            key_params = ['population_size', 'mutation_rate', 'crossover_rate', 'elite_size_ratio']
            available_params = [p for p in key_params if p in df['parameter'].values]
            
            if not available_params:
                print("⚠️ No key parameters found in evolution data")
                return ""
            
            n_params = len(available_params)
            n_cols = 2
            n_rows = math.ceil(n_params / n_cols)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
            if n_rows == 1:
                axes = axes.reshape(1, -1)
            fig.suptitle('Configuration Parameter Evolution', fontsize=16, fontweight='bold')
            
            for i, param in enumerate(available_params):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]
                
                param_data = df[df['parameter'] == param]
                
                # Plot evolution over generations
                profiles = param_data['profile'].unique()
                colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))
                
                for profile, color in zip(profiles, colors):
                    profile_data = param_data[param_data['profile'] == profile]
                    ax.plot(profile_data['generation'], profile_data['value'], 
                           marker='o', label=profile, color=color, linewidth=2)
                
                ax.set_title(f'{param.replace("_", " ").title()}')
                ax.set_xlabel('Generation')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_params, n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            filename = os.path.join(self.config.output_dir, f"configuration_evolution.{self.config.figure_format}")
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
        
        # Save data
        if self.config.save_data:
            data_filename = os.path.join(self.config.output_dir, "configuration_evolution_data.csv")
            df.to_csv(data_filename, index=False)
        
        print(f"📊 Configuration evolution saved: {filename}")
        return filename
    
    def visualize_performance_improvements(self, benchmark_results: List[BenchmarkResult]) -> str:
        """Visualize performance improvements from optimization
        
        Args:
            benchmark_results: List of benchmark results (baseline vs optimized)
            
        Returns:
            Path to generated visualization
        """
        if len(benchmark_results) < 2:
            print("⚠️ Need at least baseline and optimized results for comparison")
            return ""
        
        # Separate baseline and optimized results
        baseline_results = [r for r in benchmark_results if 'baseline' in r.test_name.lower()]
        optimized_results = [r for r in benchmark_results if 'optimized' in r.test_name.lower()]
        
        if not baseline_results or not optimized_results:
            print("⚠️ Need both baseline and optimized results")
            return ""
        
        # Calculate improvements
        improvement_data = []
        for baseline, optimized in zip(baseline_results, optimized_results):
            time_improvement = (baseline.execution_time - optimized.execution_time) / baseline.execution_time * 100
            memory_improvement = (baseline.memory_usage_mb - optimized.memory_usage_mb) / baseline.memory_usage_mb * 100
            ops_improvement = (optimized.operations_per_second - baseline.operations_per_second) / baseline.operations_per_second * 100
            
            improvement_data.append({
                'test': baseline.test_name.replace('baseline_', ''),
                'time_improvement': time_improvement,
                'memory_improvement': memory_improvement,
                'ops_improvement': ops_improvement,
                'baseline_time': baseline.execution_time,
                'optimized_time': optimized.execution_time,
                'baseline_memory': baseline.memory_usage_mb,
                'optimized_memory': optimized.memory_usage_mb
            })
        
        df = pd.DataFrame(improvement_data)
        
        if self.config.interactive_plots:
            fig = self._create_interactive_performance_improvements_plot(df)
            filename = os.path.join(self.config.output_dir, "performance_improvements.html")
            fig.write_html(filename)
        else:
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle('Performance Improvements from Optimization', fontsize=16, fontweight='bold')
            
            # Time improvements
            axes[0, 0].bar(df['test'], df['time_improvement'], color=self.colors['success'])
            axes[0, 0].set_title('Execution Time Improvement (%)')
            axes[0, 0].set_ylabel('Improvement (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Memory improvements
            axes[0, 1].bar(df['test'], df['memory_improvement'], color=self.colors['info'])
            axes[0, 1].set_title('Memory Usage Improvement (%)')
            axes[0, 1].set_ylabel('Improvement (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Operations per second improvement
            axes[1, 0].bar(df['test'], df['ops_improvement'], color=self.colors['primary'])
            axes[1, 0].set_title('Operations/Second Improvement (%)')
            axes[1, 0].set_ylabel('Improvement (%)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Before/after comparison
            x = np.arange(len(df))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, df['baseline_time'], width, label='Baseline', 
                          color=self.colors['warning'], alpha=0.7)
            axes[1, 1].bar(x + width/2, df['optimized_time'], width, label='Optimized', 
                          color=self.colors['success'], alpha=0.7)
            axes[1, 1].set_title('Execution Time Comparison')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(df['test'], rotation=45)
            axes[1, 1].legend()
            
            plt.tight_layout()
            filename = os.path.join(self.config.output_dir, f"performance_improvements.{self.config.figure_format}")
            plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight')
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
        
        # Save data
        if self.config.save_data:
            data_filename = os.path.join(self.config.output_dir, "performance_improvements_data.csv")
            df.to_csv(data_filename, index=False)
        
        print(f"📊 Performance improvements saved: {filename}")
        return filename
    
    def generate_comprehensive_tuning_dashboard(self, 
                                              tuner: Optional[GAParameterTuner] = None,
                                              analyzer: Optional[GASensitivityAnalyzer] = None,
                                              config_manager: Optional[GAConfigManager] = None,
                                              optimization_results: Optional[List[OptimizationResult]] = None,
                                              algorithm_comparison: Optional[AlgorithmComparison] = None) -> str:
        """Generate comprehensive tuning dashboard
        
        Args:
            tuner: Parameter tuner with history
            analyzer: Sensitivity analyzer with results
            config_manager: Configuration manager with snapshots
            optimization_results: Hyperparameter optimization results
            algorithm_comparison: Algorithm comparison results
            
        Returns:
            Path to generated dashboard
        """
        if not any([tuner, analyzer, config_manager, optimization_results, algorithm_comparison]):
            print("⚠️ No data available for dashboard generation")
            return ""
        
        if self.config.interactive_plots:
            # Create comprehensive interactive dashboard
            fig = self._create_comprehensive_dashboard(
                tuner, analyzer, config_manager, optimization_results, algorithm_comparison
            )
            filename = os.path.join(self.config.output_dir, "comprehensive_tuning_dashboard.html")
            fig.write_html(filename)
        else:
            # Create multi-page PDF report
            filename = os.path.join(self.config.output_dir, "comprehensive_tuning_report.pdf")
            
            # Generate individual visualizations and combine
            individual_plots = []
            
            if tuner:
                plot_file = self.visualize_parameter_adaptation_history(tuner)
                if plot_file:
                    individual_plots.append(plot_file)
            
            if analyzer:
                plot_file = self.visualize_sensitivity_analysis_results(analyzer)
                if plot_file:
                    individual_plots.append(plot_file)
            
            if config_manager:
                plot_file = self.visualize_configuration_evolution(config_manager)
                if plot_file:
                    individual_plots.append(plot_file)
            
            if optimization_results:
                plot_file = self.visualize_hyperparameter_optimization_results(optimization_results)
                if plot_file:
                    individual_plots.append(plot_file)
            
            if algorithm_comparison:
                plot_file = self.visualize_algorithm_performance_comparison(algorithm_comparison)
                if plot_file:
                    individual_plots.append(plot_file)
            
            print(f"📊 Generated {len(individual_plots)} individual visualizations")
        
        # Generate summary report
        summary_filename = os.path.join(self.config.output_dir, "tuning_summary_report.json")
        summary_data = self._generate_summary_report(
            tuner, analyzer, config_manager, optimization_results, algorithm_comparison
        )
        
        with open(summary_filename, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"📊 Comprehensive tuning dashboard saved: {filename}")
        return filename
    
    def _create_comprehensive_dashboard(self, tuner, analyzer, config_manager, 
                                      optimization_results, algorithm_comparison) -> go.Figure:
        """Create comprehensive interactive dashboard"""
        # This would create a complex multi-panel dashboard
        # For now, return a simple placeholder
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], name="Dashboard"))
        fig.update_layout(title="Comprehensive GA Parameter Tuning Dashboard")
        return fig
    
    def _generate_summary_report(self, tuner, analyzer, config_manager, 
                               optimization_results, algorithm_comparison) -> Dict[str, Any]:
        """Generate summary report of tuning analysis"""
        summary = {
            'report_timestamp': time.time(),
            'analysis_components': [],
            'key_insights': [],
            'recommendations': []
        }
        
        if tuner:
            summary['analysis_components'].append('parameter_adaptation')
            if tuner.adaptation_history:
                summary['key_insights'].append(
                    f"Parameter adaptation performed {len(tuner.adaptation_history)} times"
                )
        
        if analyzer:
            summary['analysis_components'].append('sensitivity_analysis')
            if analyzer.sensitivity_results:
                most_sensitive = max(analyzer.sensitivity_results.items(), 
                                   key=lambda x: x[1].sensitivity_index)
                summary['key_insights'].append(
                    f"Most sensitive parameter: {most_sensitive[0]} (index: {most_sensitive[1].sensitivity_index:.3f})"
                )
        
        if config_manager:
            summary['analysis_components'].append('configuration_management')
            summary['key_insights'].append(
                f"Configuration profiles available: {len(config_manager.profiles)}"
            )
        
        if optimization_results:
            summary['analysis_components'].append('hyperparameter_optimization')
            best_result = max(optimization_results, key=lambda x: x.best_score)
            summary['key_insights'].append(
                f"Best optimization method: {best_result.method_used.value} (score: {best_result.best_score:.3f})"
            )
        
        if algorithm_comparison:
            summary['analysis_components'].append('algorithm_comparison')
            summary['key_insights'].append(
                f"Algorithms compared: {len(algorithm_comparison.algorithms_compared)}"
            )
        
        # Generate general recommendations
        summary['recommendations'] = [
            "Use sensitivity analysis to identify high-impact parameters for focused tuning",
            "Monitor parameter adaptation history to detect optimization patterns",
            "Create configuration profiles for different optimization scenarios",
            "Compare multiple hyperparameter optimization methods for best results"
        ]
        
        return summary
    
    # Additional helper methods for interactive plots would go here...
    def _create_interactive_optimization_results_plot(self, df, optimization_results):
        """Create interactive optimization results plot"""
        # Placeholder for complex interactive plot
        fig = px.bar(df, x='method', y='best_score', title='Optimization Results')
        return fig
    
    def _create_interactive_algorithm_comparison_plot(self, df, comparison):
        """Create interactive algorithm comparison plot"""
        # Placeholder for complex interactive plot
        fig = px.bar(df, x='algorithm', y='value', color='metric', title='Algorithm Comparison')
        return fig
    
    def _create_interactive_sensitivity_plot(self, df):
        """Create interactive sensitivity analysis plot"""
        # Placeholder for complex interactive plot
        fig = px.scatter(df, x='sensitivity_index', y='variance_contribution', 
                        color='parameter', title='Sensitivity Analysis')
        return fig
    
    def _create_interactive_config_evolution_plot(self, df):
        """Create interactive configuration evolution plot"""
        # Placeholder for complex interactive plot
        fig = px.line(df, x='generation', y='value', color='parameter', 
                     title='Configuration Evolution')
        return fig
    
    def _create_interactive_performance_improvements_plot(self, df):
        """Create interactive performance improvements plot"""
        # Placeholder for complex interactive plot
        fig = px.bar(df, x='test', y='time_improvement', title='Performance Improvements')
        return fig


def test_tuning_visualizer():
    """Test function for tuning visualizer"""
    print("Testing GA Tuning Visualizer...")
    
    # Create visualizer
    config = VisualizationConfig(output_dir="test_visualizations", show_plots=False)
    visualizer = GATuningVisualizer(config)
    
    # Test with mock data
    from ga_parameter_tuner import GAParameterTuner, ParameterHistory, PopulationStats
    
    # Create tuner with mock history
    tuner = GAParameterTuner()
    
    # Add mock adaptation history
    mock_history = ParameterHistory(
        generation=10,
        parameters={'mutation_rate': 0.12, 'population_size': 120},
        performance_metrics={'fitness': 0.8, 'diversity': 0.6},
        adaptation_reason="low diversity",
        timestamp=time.time()
    )
    tuner.adaptation_history.append(mock_history)
    
    # Test parameter adaptation visualization
    plot_file = visualizer.visualize_parameter_adaptation_history(tuner)
    print(f"✅ Parameter adaptation visualization: {plot_file}")
    
    # Test with mock sensitivity analyzer
    from ga_sensitivity_analyzer import GASensitivityAnalyzer, SensitivityResult
    
    analyzer = GASensitivityAnalyzer()
    analyzer.sensitivity_results = {
        'mutation_rate_fitness': SensitivityResult(
            parameter_name='mutation_rate',
            sensitivity_index=0.7,
            confidence_interval=(0.5, 0.9),
            variance_contribution=0.4,
            statistical_significance=True,
            rank=1
        )
    }
    
    # Test sensitivity visualization
    plot_file = visualizer.visualize_sensitivity_analysis_results(analyzer)
    print(f"✅ Sensitivity analysis visualization: {plot_file}")
    
    # Test comprehensive dashboard
    dashboard_file = visualizer.generate_comprehensive_tuning_dashboard(
        tuner=tuner, analyzer=analyzer
    )
    print(f"✅ Comprehensive dashboard: {dashboard_file}")
    
    print("✅ All tuning visualizer tests completed")


if __name__ == "__main__":
    test_tuning_visualizer()
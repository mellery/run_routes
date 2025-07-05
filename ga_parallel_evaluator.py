#!/usr/bin/env python3
"""
GA Parallel Population Evaluator
High-performance parallel evaluation system for genetic algorithm populations
"""

import multiprocessing as mp
import threading
import time
import math
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import queue
import os

from ga_chromosome import RouteChromosome
from ga_fitness import GAFitnessEvaluator


@dataclass
class ParallelConfig:
    """Configuration for parallel evaluation"""
    max_workers: Optional[int] = None
    chunk_size: int = 10
    use_processes: bool = True
    timeout_seconds: float = 30.0
    memory_limit_mb: float = 1000.0
    enable_batching: bool = True
    batch_size: int = 50


@dataclass
class EvaluationTask:
    """Individual evaluation task"""
    chromosome: RouteChromosome
    task_id: int
    objective: str
    target_distance_km: float
    priority: int = 0


@dataclass
class EvaluationResult:
    """Result of chromosome evaluation"""
    task_id: int
    chromosome: RouteChromosome
    fitness: float
    evaluation_time: float
    success: bool
    error_message: Optional[str] = None


class WorkerProcess:
    """Individual worker process for chromosome evaluation"""
    
    @staticmethod
    def evaluate_chromosome_worker(args: Tuple[RouteChromosome, str, float, int]) -> EvaluationResult:
        """Worker function for evaluating a single chromosome
        
        Args:
            args: Tuple of (chromosome, objective, target_distance_km, task_id)
            
        Returns:
            EvaluationResult with fitness and timing information
        """
        chromosome, objective, target_distance_km, task_id = args
        start_time = time.time()
        
        try:
            # Create fitness evaluator (each worker needs its own instance)
            evaluator = GAFitnessEvaluator(objective, target_distance_km)
            
            # Evaluate chromosome
            fitness = evaluator.evaluate_chromosome(chromosome)
            
            evaluation_time = time.time() - start_time
            
            return EvaluationResult(
                task_id=task_id,
                chromosome=chromosome,
                fitness=fitness,
                evaluation_time=evaluation_time,
                success=True
            )
            
        except Exception as e:
            evaluation_time = time.time() - start_time
            return EvaluationResult(
                task_id=task_id,
                chromosome=chromosome,
                fitness=0.0,
                evaluation_time=evaluation_time,
                success=False,
                error_message=str(e)
            )
    
    @staticmethod
    def evaluate_batch_worker(args: Tuple[List[RouteChromosome], str, float, List[int]]) -> List[EvaluationResult]:
        """Worker function for evaluating a batch of chromosomes
        
        Args:
            args: Tuple of (chromosomes, objective, target_distance_km, task_ids)
            
        Returns:
            List of EvaluationResult objects
        """
        chromosomes, objective, target_distance_km, task_ids = args
        results = []
        
        # Create single evaluator for the batch (more efficient)
        evaluator = GAFitnessEvaluator(objective, target_distance_km)
        
        for chromosome, task_id in zip(chromosomes, task_ids):
            start_time = time.time()
            
            try:
                fitness = evaluator.evaluate_chromosome(chromosome)
                evaluation_time = time.time() - start_time
                
                results.append(EvaluationResult(
                    task_id=task_id,
                    chromosome=chromosome,
                    fitness=fitness,
                    evaluation_time=evaluation_time,
                    success=True
                ))
                
            except Exception as e:
                evaluation_time = time.time() - start_time
                results.append(EvaluationResult(
                    task_id=task_id,
                    chromosome=chromosome,
                    fitness=0.0,
                    evaluation_time=evaluation_time,
                    success=False,
                    error_message=str(e)
                ))
        
        return results


class GAParallelEvaluator:
    """High-performance parallel evaluator for GA populations"""
    
    def __init__(self, config: Optional[ParallelConfig] = None):
        """Initialize parallel evaluator
        
        Args:
            config: Parallel evaluation configuration
        """
        self.config = config or ParallelConfig()
        
        # Auto-detect optimal worker count if not specified
        if self.config.max_workers is None:
            cpu_count = os.cpu_count() or 4
            self.config.max_workers = max(1, cpu_count - 1)  # Leave one CPU free
        
        # Performance tracking
        self.total_evaluations = 0
        self.total_evaluation_time = 0.0
        self.parallel_evaluations = 0
        self.sequential_evaluations = 0
        self.failed_evaluations = 0
        
        # Execution pools
        self.process_pool = None
        self.thread_pool = None
        
        # Task queue for batch processing
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        print(f"🚀 Parallel evaluator initialized: {self.config.max_workers} workers, "
              f"{'processes' if self.config.use_processes else 'threads'}")
    
    def evaluate_population_parallel(self, population: List[RouteChromosome],
                                   objective: str, target_distance_km: float,
                                   progress_callback: Optional[Callable[[float], None]] = None) -> List[float]:
        """Evaluate population fitness using parallel processing
        
        Args:
            population: Population to evaluate
            objective: Optimization objective
            target_distance_km: Target route distance
            progress_callback: Optional progress reporting function
            
        Returns:
            List of fitness scores
        """
        if not population:
            return []
        
        start_time = time.time()
        
        # Determine if parallel evaluation is beneficial
        if len(population) < self.config.chunk_size:
            # Use sequential evaluation for small populations
            return self._evaluate_population_sequential(population, objective, target_distance_km)
        
        # Use parallel evaluation
        if self.config.enable_batching and len(population) >= self.config.batch_size:
            fitness_scores = self._evaluate_population_batched(
                population, objective, target_distance_km, progress_callback
            )
        else:
            fitness_scores = self._evaluate_population_individual(
                population, objective, target_distance_km, progress_callback
            )
        
        # Update statistics
        evaluation_time = time.time() - start_time
        self.total_evaluations += len(population)
        self.total_evaluation_time += evaluation_time
        self.parallel_evaluations += len(population)
        
        return fitness_scores
    
    def _evaluate_population_sequential(self, population: List[RouteChromosome],
                                      objective: str, target_distance_km: float) -> List[float]:
        """Sequential evaluation fallback for small populations"""
        evaluator = GAFitnessEvaluator(objective, target_distance_km)
        fitness_scores = []
        
        for chromosome in population:
            try:
                fitness = evaluator.evaluate_chromosome(chromosome)
                fitness_scores.append(fitness)
            except Exception:
                fitness_scores.append(0.0)
                self.failed_evaluations += 1
        
        self.sequential_evaluations += len(population)
        return fitness_scores
    
    def _evaluate_population_individual(self, population: List[RouteChromosome],
                                      objective: str, target_distance_km: float,
                                      progress_callback: Optional[Callable[[float], None]]) -> List[float]:
        """Parallel evaluation with individual chromosome tasks"""
        
        # Prepare tasks
        tasks = []
        for i, chromosome in enumerate(population):
            task = (chromosome, objective, target_distance_km, i)
            tasks.append(task)
        
        # Execute in parallel
        fitness_scores = [0.0] * len(population)
        completed = 0
        
        try:
            if self.config.use_processes:
                with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_index = {
                        executor.submit(WorkerProcess.evaluate_chromosome_worker, task): i
                        for i, task in enumerate(tasks)
                    }
                    
                    for future in as_completed(future_to_index, timeout=self.config.timeout_seconds):
                        try:
                            result = future.result()
                            fitness_scores[result.task_id] = result.fitness
                            
                            if not result.success:
                                self.failed_evaluations += 1
                            
                            completed += 1
                            if progress_callback:
                                progress_callback(completed / len(population))
                                
                        except Exception as e:
                            print(f"⚠️ Evaluation task failed: {e}")
                            self.failed_evaluations += 1
            else:
                # Use thread pool for I/O bound tasks
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_index = {
                        executor.submit(WorkerProcess.evaluate_chromosome_worker, task): i
                        for i, task in enumerate(tasks)
                    }
                    
                    for future in as_completed(future_to_index, timeout=self.config.timeout_seconds):
                        try:
                            result = future.result()
                            fitness_scores[result.task_id] = result.fitness
                            
                            if not result.success:
                                self.failed_evaluations += 1
                            
                            completed += 1
                            if progress_callback:
                                progress_callback(completed / len(population))
                                
                        except Exception as e:
                            print(f"⚠️ Evaluation task failed: {e}")
                            self.failed_evaluations += 1
        
        except Exception as e:
            print(f"⚠️ Parallel evaluation failed, falling back to sequential: {e}")
            return self._evaluate_population_sequential(population, objective, target_distance_km)
        
        return fitness_scores
    
    def _evaluate_population_batched(self, population: List[RouteChromosome],
                                   objective: str, target_distance_km: float,
                                   progress_callback: Optional[Callable[[float], None]]) -> List[float]:
        """Parallel evaluation with batched tasks (more efficient)"""
        
        # Create batches
        batch_size = self.config.batch_size
        batches = []
        task_id_batches = []
        
        for i in range(0, len(population), batch_size):
            batch = population[i:i + batch_size]
            task_ids = list(range(i, i + len(batch)))
            
            batches.append((batch, objective, target_distance_km, task_ids))
            task_id_batches.append(task_ids)
        
        # Execute batches in parallel
        fitness_scores = [0.0] * len(population)
        completed_batches = 0
        
        try:
            if self.config.use_processes:
                with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(WorkerProcess.evaluate_batch_worker, batch): i
                        for i, batch in enumerate(batches)
                    }
                    
                    for future in as_completed(future_to_batch, timeout=self.config.timeout_seconds):
                        try:
                            batch_results = future.result()
                            
                            for result in batch_results:
                                fitness_scores[result.task_id] = result.fitness
                                
                                if not result.success:
                                    self.failed_evaluations += 1
                            
                            completed_batches += 1
                            if progress_callback:
                                progress_callback(completed_batches / len(batches))
                                
                        except Exception as e:
                            print(f"⚠️ Batch evaluation failed: {e}")
                            self.failed_evaluations += batch_size
            else:
                # Use thread pool
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_batch = {
                        executor.submit(WorkerProcess.evaluate_batch_worker, batch): i
                        for i, batch in enumerate(batches)
                    }
                    
                    for future in as_completed(future_to_batch, timeout=self.config.timeout_seconds):
                        try:
                            batch_results = future.result()
                            
                            for result in batch_results:
                                fitness_scores[result.task_id] = result.fitness
                                
                                if not result.success:
                                    self.failed_evaluations += 1
                            
                            completed_batches += 1
                            if progress_callback:
                                progress_callback(completed_batches / len(batches))
                                
                        except Exception as e:
                            print(f"⚠️ Batch evaluation failed: {e}")
                            self.failed_evaluations += batch_size
        
        except Exception as e:
            print(f"⚠️ Parallel batch evaluation failed, falling back to sequential: {e}")
            return self._evaluate_population_sequential(population, objective, target_distance_km)
        
        return fitness_scores
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get parallel evaluation performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        avg_evaluation_time = (self.total_evaluation_time / max(self.total_evaluations, 1))
        
        parallel_ratio = self.parallel_evaluations / max(self.total_evaluations, 1)
        failure_rate = self.failed_evaluations / max(self.total_evaluations, 1)
        
        return {
            'total_evaluations': self.total_evaluations,
            'total_evaluation_time': self.total_evaluation_time,
            'avg_evaluation_time': avg_evaluation_time,
            'parallel_evaluations': self.parallel_evaluations,
            'sequential_evaluations': self.sequential_evaluations,
            'failed_evaluations': self.failed_evaluations,
            'parallel_ratio': parallel_ratio,
            'failure_rate': failure_rate,
            'max_workers': self.config.max_workers,
            'use_processes': self.config.use_processes,
            'batch_size': self.config.batch_size if self.config.enable_batching else None
        }
    
    def benchmark_evaluation_methods(self, population: List[RouteChromosome],
                                   objective: str, target_distance_km: float) -> Dict[str, float]:
        """Benchmark different evaluation methods for performance comparison
        
        Args:
            population: Test population
            objective: Optimization objective
            target_distance_km: Target distance
            
        Returns:
            Dictionary with timing results for each method
        """
        if len(population) < 10:
            print("⚠️ Population too small for meaningful benchmark")
            return {}
        
        # Test sequential evaluation
        print("🔄 Benchmarking sequential evaluation...")
        start_time = time.time()
        sequential_scores = self._evaluate_population_sequential(population, objective, target_distance_km)
        sequential_time = time.time() - start_time
        
        # Test individual parallel evaluation
        print("🔄 Benchmarking individual parallel evaluation...")
        start_time = time.time()
        individual_scores = self._evaluate_population_individual(population, objective, target_distance_km, None)
        individual_time = time.time() - start_time
        
        # Test batched parallel evaluation (if enabled)
        batched_time = None
        if self.config.enable_batching and len(population) >= self.config.batch_size:
            print("🔄 Benchmarking batched parallel evaluation...")
            start_time = time.time()
            batched_scores = self._evaluate_population_batched(population, objective, target_distance_km, None)
            batched_time = time.time() - start_time
        
        # Calculate speedups
        speedup_individual = sequential_time / individual_time if individual_time > 0 else 0
        speedup_batched = sequential_time / batched_time if batched_time and batched_time > 0 else 0
        
        results = {
            'sequential_time': sequential_time,
            'individual_parallel_time': individual_time,
            'individual_speedup': speedup_individual,
            'population_size': len(population),
            'workers': self.config.max_workers
        }
        
        if batched_time:
            results.update({
                'batched_parallel_time': batched_time,
                'batched_speedup': speedup_batched
            })
        
        print(f"📊 Benchmark results:")
        print(f"   Sequential: {sequential_time:.2f}s")
        print(f"   Individual parallel: {individual_time:.2f}s (speedup: {speedup_individual:.1f}x)")
        if batched_time:
            print(f"   Batched parallel: {batched_time:.2f}s (speedup: {speedup_batched:.1f}x)")
        
        return results
    
    def cleanup(self):
        """Clean up resources"""
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)


def test_parallel_evaluator():
    """Test function for parallel evaluator"""
    print("Testing GA Parallel Evaluator...")
    
    # Create test population
    from ga_chromosome import RouteSegment, RouteChromosome
    
    test_population = []
    for i in range(20):
        segment = RouteSegment(1, 2, [1, 2])
        segment.length = 1000.0 + i * 100
        segment.elevation_gain = 50.0 + i * 5
        
        chromosome = RouteChromosome([segment])
        test_population.append(chromosome)
    
    # Test parallel evaluator
    config = ParallelConfig(max_workers=2, chunk_size=5, batch_size=10)
    evaluator = GAParallelEvaluator(config)
    
    # Test parallel evaluation
    start_time = time.time()
    fitness_scores = evaluator.evaluate_population_parallel(
        test_population, "elevation", 3.0
    )
    evaluation_time = time.time() - start_time
    
    print(f"✅ Parallel evaluation completed: {len(fitness_scores)} scores in {evaluation_time:.2f}s")
    print(f"   Average fitness: {sum(fitness_scores)/len(fitness_scores):.3f}")
    
    # Test benchmark
    benchmark_results = evaluator.benchmark_evaluation_methods(
        test_population[:10], "elevation", 3.0
    )
    print(f"✅ Benchmark completed: {benchmark_results.get('individual_speedup', 0):.1f}x speedup")
    
    # Test performance stats
    stats = evaluator.get_performance_stats()
    print(f"✅ Performance stats: {stats['total_evaluations']} evaluations")
    
    print("✅ All parallel evaluator tests completed")


if __name__ == "__main__":
    test_parallel_evaluator()
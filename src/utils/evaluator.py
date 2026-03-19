"""
Building Energy Management System - Evaluation Utilities

This module provides comprehensive evaluation capabilities for energy prediction models,
including accuracy metrics, edge performance benchmarking, and robustness testing.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import time
import psutil
import os

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""
    mae: float
    rmse: float
    r2: float
    mape: float
    mse: float
    model_size: int
    inference_time_ms: float
    memory_usage_mb: float


@dataclass
class EdgePerformanceMetrics:
    """Container for edge deployment performance metrics."""
    avg_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float
    model_size_kb: float
    memory_usage_mb: float
    cpu_usage_percent: float
    power_consumption_w: Optional[float] = None


class ModelEvaluator:
    """
    Comprehensive model evaluation for building energy prediction.
    
    Provides accuracy metrics, edge performance benchmarking,
    and robustness testing capabilities.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.results: Dict[str, Any] = {}
    
    def evaluate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate accuracy metrics for energy prediction.
        
        Args:
            y_true: True energy consumption values
            y_pred: Predicted energy consumption values
            
        Returns:
            Dictionary of accuracy metrics
        """
        # Basic regression metrics
        mae = np.mean(np.abs(y_true - y_pred))
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        # Additional metrics
        max_error = np.max(np.abs(y_true - y_pred))
        mean_error = np.mean(y_true - y_pred)
        
        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'max_error': float(max_error),
            'mean_error': float(mean_error)
        }
        
        logger.info(f"Accuracy metrics - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
        return metrics
    
    def benchmark_inference_performance(self, model, X_test: np.ndarray, 
                                      num_runs: int = 100) -> EdgePerformanceMetrics:
        """
        Benchmark model inference performance.
        
        Args:
            model: Model to benchmark
            X_test: Test input data
            num_runs: Number of inference runs for averaging
            
        Returns:
            Edge performance metrics
        """
        logger.info(f"Benchmarking inference performance with {num_runs} runs")
        
        # Warmup runs
        for _ in range(10):
            if hasattr(model, 'predict'):
                model.predict(X_test[:1], verbose=0)
            else:
                model(X_test[:1])
        
        # Measure inference times
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            # Measure memory before inference
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time inference
            start_time = time.time()
            
            if hasattr(model, 'predict'):
                _ = model.predict(X_test[i:i+1], verbose=0)
            else:
                _ = model(X_test[i:i+1])
            
            end_time = time.time()
            
            # Measure memory after inference
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            times.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
        
        # Calculate metrics
        avg_latency_ms = np.mean(times) * 1000
        p95_latency_ms = np.percentile(times, 95) * 1000
        throughput_fps = 1.0 / np.mean(times)
        avg_memory_mb = np.mean(memory_usage)
        
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        metrics = EdgePerformanceMetrics(
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
            throughput_fps=throughput_fps,
            model_size_kb=0,  # Will be set separately
            memory_usage_mb=avg_memory_mb,
            cpu_usage_percent=cpu_usage
        )
        
        logger.info(f"Performance metrics - Latency: {avg_latency_ms:.2f}ms, "
                   f"Throughput: {throughput_fps:.1f} FPS")
        
        return metrics
    
    def test_robustness(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       noise_levels: List[float] = [0.01, 0.05, 0.1]) -> Dict[str, Dict[str, float]]:
        """
        Test model robustness to input noise.
        
        Args:
            model: Model to test
            X_test: Test input data
            y_test: Test target data
            noise_levels: List of noise levels to test
            
        Returns:
            Dictionary of robustness results
        """
        logger.info("Testing model robustness to input noise")
        
        robustness_results = {}
        
        for noise_level in noise_levels:
            # Add noise to inputs
            noise = np.random.normal(0, noise_level, X_test.shape)
            X_noisy = X_test + noise
            
            # Get predictions
            if hasattr(model, 'predict'):
                y_pred_noisy = model.predict(X_noisy, verbose=0).flatten()
            else:
                y_pred_noisy = model(X_noisy).numpy().flatten()
            
            # Calculate metrics
            metrics = self.evaluate_accuracy(y_test, y_pred_noisy)
            robustness_results[f'noise_{noise_level}'] = metrics
        
        logger.info(f"Robustness testing completed for {len(noise_levels)} noise levels")
        return robustness_results
    
    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray, 
                      y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple models and return results as DataFrame.
        
        Args:
            models: Dictionary of model names and model objects
            X_test: Test input data
            y_test: Test target data
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing {len(models)} models")
        
        results = []
        
        for model_name, model in models.items():
            # Get predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test, verbose=0).flatten()
            else:
                y_pred = model(X_test).numpy().flatten()
            
            # Calculate accuracy metrics
            accuracy_metrics = self.evaluate_accuracy(y_test, y_pred)
            
            # Benchmark performance
            performance_metrics = self.benchmark_inference_performance(model, X_test)
            
            # Get model size
            model_size = 0
            if hasattr(model, 'count_params'):
                model_size = model.count_params()
            
            result = {
                'model_name': model_name,
                'mae': accuracy_metrics['mae'],
                'rmse': accuracy_metrics['rmse'],
                'r2': accuracy_metrics['r2'],
                'mape': accuracy_metrics['mape'],
                'avg_latency_ms': performance_metrics.avg_latency_ms,
                'throughput_fps': performance_metrics.throughput_fps,
                'memory_usage_mb': performance_metrics.memory_usage_mb,
                'model_size': model_size
            }
            
            results.append(result)
        
        df = pd.DataFrame(results)
        
        # Sort by MAE (lower is better)
        df = df.sort_values('mae')
        
        logger.info("Model comparison completed")
        return df
    
    def create_performance_report(self, comparison_df: pd.DataFrame, 
                                output_dir: str = 'assets') -> None:
        """
        Create comprehensive performance report with visualizations.
        
        Args:
            comparison_df: DataFrame with model comparison results
            output_dir: Output directory for reports
        """
        logger.info("Creating performance report")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Accuracy metrics
        metrics = ['mae', 'rmse', 'r2']
        for i, metric in enumerate(metrics):
            axes[0, i].bar(comparison_df['model_name'], comparison_df[metric])
            axes[0, i].set_title(f'{metric.upper()} Comparison')
            axes[0, i].set_ylabel(metric.upper())
            axes[0, i].tick_params(axis='x', rotation=45)
        
        # Performance metrics
        axes[1, 0].bar(comparison_df['model_name'], comparison_df['avg_latency_ms'])
        axes[1, 0].set_title('Average Latency (ms)')
        axes[1, 0].set_ylabel('Latency (ms)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        axes[1, 1].bar(comparison_df['model_name'], comparison_df['throughput_fps'])
        axes[1, 1].set_title('Throughput (FPS)')
        axes[1, 1].set_ylabel('FPS')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 2].bar(comparison_df['model_name'], comparison_df['memory_usage_mb'])
        axes[1, 2].set_title('Memory Usage (MB)')
        axes[1, 2].set_ylabel('Memory (MB)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Accuracy vs Performance scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(comparison_df['avg_latency_ms'], comparison_df['mae'], 
                            s=comparison_df['model_size']/1000, alpha=0.7)
        
        for i, model_name in enumerate(comparison_df['model_name']):
            plt.annotate(model_name, 
                        (comparison_df['avg_latency_ms'].iloc[i], 
                         comparison_df['mae'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Average Latency (ms)')
        plt.ylabel('Mean Absolute Error (kWh)')
        plt.title('Accuracy vs Performance Trade-off\n(Bubble size = Model Size)')
        plt.grid(True, alpha=0.3)
        
        # Add colorbar for model size
        cbar = plt.colorbar(scatter)
        cbar.set_label('Model Size (Parameters)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/accuracy_vs_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed report
        report_path = f'{output_dir}/performance_report.txt'
        with open(report_path, 'w') as f:
            f.write("Building Energy Management System - Performance Report\n")
            f.write("=" * 60 + "\n\n")
            f.write("DISCLAIMER: This system is for research and education purposes only.\n")
            f.write("NOT FOR SAFETY-CRITICAL USE.\n\n")
            
            f.write("MODEL COMPARISON RESULTS\n")
            f.write("-" * 30 + "\n")
            f.write(comparison_df.to_string(index=False, float_format='%.4f'))
            f.write("\n\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            
            # Find best models for different criteria
            best_accuracy = comparison_df.loc[comparison_df['mae'].idxmin()]
            best_performance = comparison_df.loc[comparison_df['avg_latency_ms'].idxmin()]
            best_balanced = comparison_df.loc[(comparison_df['mae'] / comparison_df['mae'].max() + 
                                             comparison_df['avg_latency_ms'] / comparison_df['avg_latency_ms'].max()).idxmin()]
            
            f.write(f"Best Accuracy: {best_accuracy['model_name']} (MAE: {best_accuracy['mae']:.3f})\n")
            f.write(f"Best Performance: {best_performance['model_name']} (Latency: {best_performance['avg_latency_ms']:.2f}ms)\n")
            f.write(f"Best Balanced: {best_balanced['model_name']} (MAE: {best_balanced['mae']:.3f}, Latency: {best_balanced['avg_latency_ms']:.2f}ms)\n")
        
        logger.info(f"Performance report saved to {output_dir}")
    
    def evaluate_edge_deployment(self, model_path: str, X_test: np.ndarray,
                               runtime: str = 'tflite') -> EdgePerformanceMetrics:
        """
        Evaluate model performance on edge deployment.
        
        Args:
            model_path: Path to edge model file
            X_test: Test input data
            runtime: Runtime type ('tflite', 'onnx')
            
        Returns:
            Edge performance metrics
        """
        logger.info(f"Evaluating edge deployment: {model_path}")
        
        if runtime == 'tflite':
            return self._evaluate_tflite_model(model_path, X_test)
        elif runtime == 'onnx':
            return self._evaluate_onnx_model(model_path, X_test)
        else:
            raise ValueError(f"Unsupported runtime: {runtime}")
    
    def _evaluate_tflite_model(self, model_path: str, X_test: np.ndarray) -> EdgePerformanceMetrics:
        """Evaluate TFLite model performance."""
        import tensorflow as tf
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], X_test[:1].astype(np.float32))
            interpreter.invoke()
        
        # Benchmark
        times = []
        for i in range(len(X_test)):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], X_test[i:i+1].astype(np.float32))
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            times.append(time.time() - start_time)
        
        avg_latency_ms = np.mean(times) * 1000
        p95_latency_ms = np.percentile(times, 95) * 1000
        throughput_fps = 1.0 / np.mean(times)
        model_size_kb = os.path.getsize(model_path) / 1024
        
        return EdgePerformanceMetrics(
            avg_latency_ms=avg_latency_ms,
            p95_latency_ms=p95_latency_ms,
            throughput_fps=throughput_fps,
            model_size_kb=model_size_kb,
            memory_usage_mb=0,  # TFLite memory usage is hard to measure
            cpu_usage_percent=psutil.cpu_percent(interval=1)
        )
    
    def _evaluate_onnx_model(self, model_path: str, X_test: np.ndarray) -> EdgePerformanceMetrics:
        """Evaluate ONNX model performance."""
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                session.run(None, {input_name: X_test[:1].astype(np.float32)})
            
            # Benchmark
            times = []
            for i in range(len(X_test)):
                start_time = time.time()
                session.run(None, {input_name: X_test[i:i+1].astype(np.float32)})
                times.append(time.time() - start_time)
            
            avg_latency_ms = np.mean(times) * 1000
            p95_latency_ms = np.percentile(times, 95) * 1000
            throughput_fps = 1.0 / np.mean(times)
            model_size_kb = os.path.getsize(model_path) / 1024
            
            return EdgePerformanceMetrics(
                avg_latency_ms=avg_latency_ms,
                p95_latency_ms=p95_latency_ms,
                throughput_fps=throughput_fps,
                model_size_kb=model_size_kb,
                memory_usage_mb=0,  # ONNX memory usage is hard to measure
                cpu_usage_percent=psutil.cpu_percent(interval=1)
            )
            
        except ImportError:
            logger.error("onnxruntime not installed")
            raise

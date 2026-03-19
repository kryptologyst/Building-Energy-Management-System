"""
Building Energy Management System - Evaluation Script

Command-line script for comprehensive model evaluation and benchmarking.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.energy_predictor import EnergyPredictor, ModelConfig
from src.data.data_generator import BuildingDataGenerator, DataConfig
from src.utils.evaluator import ModelEvaluator
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_models(model_paths: List[str], test_data_path: str = None) -> pd.DataFrame:
    """
    Evaluate multiple models and return comparison results.
    
    Args:
        model_paths: List of paths to trained models
        test_data_path: Path to test data (if None, generates synthetic data)
        
    Returns:
        DataFrame with evaluation results
    """
    logger.info(f"Evaluating {len(model_paths)} models")
    
    # Load or generate test data
    if test_data_path and os.path.exists(test_data_path):
        logger.info(f"Loading test data from {test_data_path}")
        df = pd.read_csv(test_data_path)
        feature_cols = ['hvac_runtime_hours', 'lighting_usage_hours', 
                       'occupancy_count', 'outside_temp_celsius']
        X_test = df[feature_cols].values
        y_test = df['total_energy_kwh'].values
    else:
        logger.info("Generating synthetic test data")
        generator = BuildingDataGenerator(DataConfig(n_samples=500))
        X_test, y_test = generator.generate_sensor_data()
    
    # Load models
    models = {}
    for model_path in model_paths:
        if os.path.exists(model_path):
            model_name = os.path.basename(model_path).replace('.h5', '')
            model = EnergyPredictor(ModelConfig())
            model.load_model(model_path)
            models[model_name] = model
            logger.info(f"Loaded {model_name} model")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    if not models:
        logger.error("No models loaded for evaluation")
        return pd.DataFrame()
    
    # Evaluate models
    evaluator = ModelEvaluator()
    comparison_df = evaluator.compare_models(models, X_test, y_test)
    
    # Create performance report
    os.makedirs('assets', exist_ok=True)
    evaluator.create_performance_report(comparison_df, 'assets')
    
    return comparison_df


def benchmark_edge_models(edge_model_paths: List[str]) -> Dict[str, Any]:
    """
    Benchmark edge-optimized models.
    
    Args:
        edge_model_paths: List of paths to edge models
        
    Returns:
        Dictionary with benchmarking results
    """
    logger.info(f"Benchmarking {len(edge_model_paths)} edge models")
    
    # Generate test data
    generator = BuildingDataGenerator(DataConfig(n_samples=100))
    X_test, _ = generator.generate_sensor_data()
    
    evaluator = ModelEvaluator()
    results = {}
    
    for model_path in edge_model_paths:
        if os.path.exists(model_path):
            model_name = os.path.basename(model_path)
            
            # Determine runtime type
            if model_path.endswith('.tflite'):
                runtime = 'tflite'
            elif model_path.endswith('.onnx'):
                runtime = 'onnx'
            else:
                logger.warning(f"Unknown model format: {model_path}")
                continue
            
            # Benchmark model
            metrics = evaluator.evaluate_edge_deployment(model_path, X_test, runtime)
            results[model_name] = metrics
            
            logger.info(f"{model_name} - Latency: {metrics.avg_latency_ms:.2f}ms, "
                       f"Throughput: {metrics.throughput_fps:.1f} FPS")
    
    return results


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate Building Energy Management Models')
    parser.add_argument('--models', nargs='+', required=True,
                      help='Paths to trained model files')
    parser.add_argument('--test-data', type=str,
                      help='Path to test data CSV file')
    parser.add_argument('--edge-models', nargs='+',
                      help='Paths to edge-optimized model files')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--output', type=str, default='assets/evaluation_results.csv',
                      help='Output file for evaluation results')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("Starting model evaluation process")
    logger.info("DISCLAIMER: This system is for research and education purposes only")
    logger.info("NOT FOR SAFETY-CRITICAL USE")
    
    # Evaluate models
    comparison_df = evaluate_models(args.models, args.test_data)
    
    if not comparison_df.empty:
        logger.info("Model evaluation completed")
        
        # Save results
        comparison_df.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Find best models
        best_accuracy = comparison_df.loc[comparison_df['mae'].idxmin()]
        best_performance = comparison_df.loc[comparison_df['avg_latency_ms'].idxmin()]
        
        print(f"\nBest Accuracy: {best_accuracy['model_name']} (MAE: {best_accuracy['mae']:.3f})")
        print(f"Best Performance: {best_performance['model_name']} (Latency: {best_performance['avg_latency_ms']:.2f}ms)")
    
    # Benchmark edge models if provided
    if args.edge_models:
        edge_results = benchmark_edge_models(args.edge_models)
        
        if edge_results:
            print("\n" + "="*60)
            print("EDGE MODEL BENCHMARKS")
            print("="*60)
            
            for model_name, metrics in edge_results.items():
                print(f"{model_name}:")
                print(f"  Latency: {metrics.avg_latency_ms:.2f} ms")
                print(f"  Throughput: {metrics.throughput_fps:.1f} FPS")
                print(f"  Model Size: {metrics.model_size_kb:.1f} KB")
                print(f"  Memory Usage: {metrics.memory_usage_mb:.1f} MB")
                print()


if __name__ == '__main__':
    main()

"""
Building Energy Management System - Main Training Script

This script trains both baseline and edge-optimized models for building energy prediction,
with comprehensive evaluation and edge deployment capabilities.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.energy_predictor import EnergyPredictor, ModelConfig, create_baseline_model, create_edge_optimized_model
from src.data.data_generator import BuildingDataGenerator, DataConfig, create_default_generator
from src.export.edge_exporter import EdgeModelExporter, ModelCompressor
from src.utils.evaluator import ModelEvaluator
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate and prepare training data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Generating synthetic building sensor data")
    
    # Create data generator
    data_config = DataConfig(**config['data'])
    generator = BuildingDataGenerator(data_config)
    
    # Generate data
    X, y = generator.generate_sensor_data()
    
    # Split data
    test_size = config['evaluation']['test_size']
    random_state = config['evaluation']['random_state']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
    
    # Save data for later use
    os.makedirs('data/processed', exist_ok=True)
    generator.save_data(X, y, 'data/processed/building_sensor_data.csv')
    
    return X_train, X_test, y_train, y_test


def train_baseline_model(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        config: Dict[str, Any]) -> EnergyPredictor:
    """
    Train baseline energy prediction model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        config: Configuration dictionary
        
    Returns:
        Trained baseline model
    """
    logger.info("Training baseline energy prediction model")
    
    # Create baseline model
    model_config = ModelConfig(**config['model'])
    baseline_model = EnergyPredictor(model_config)
    
    # Train model
    baseline_model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = baseline_model.evaluate(X_test, y_test)
    logger.info(f"Baseline model metrics: {metrics}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    baseline_model.save_model('models/baseline_energy_predictor.h5')
    
    return baseline_model


def train_edge_optimized_model(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray,
                              config: Dict[str, Any]) -> EnergyPredictor:
    """
    Train edge-optimized energy prediction model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        config: Configuration dictionary
        
    Returns:
        Trained edge-optimized model
    """
    logger.info("Training edge-optimized energy prediction model")
    
    # Create edge-optimized model
    edge_config = ModelConfig(**config['edge_model'])
    edge_model = EnergyPredictor(edge_config)
    
    # Train model
    edge_model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = edge_model.evaluate(X_test, y_test)
    logger.info(f"Edge model metrics: {metrics}")
    
    # Save model
    edge_model.save_model('models/edge_energy_predictor.h5')
    
    return edge_model


def apply_model_compression(baseline_model: EnergyPredictor, 
                          X_train: np.ndarray, y_train: np.ndarray,
                          config: Dict[str, Any]) -> EnergyPredictor:
    """
    Apply compression techniques to create optimized models.
    
    Args:
        baseline_model: Trained baseline model
        X_train: Training features
        y_train: Training targets
        config: Configuration dictionary
        
    Returns:
        Compressed model
    """
    logger.info("Applying model compression techniques")
    
    compressor = ModelCompressor(baseline_model.model)
    
    # Apply pruning
    pruning_sparsity = config['deployment']['optimization']['pruning_sparsity']
    pruned_model = compressor.apply_pruning(sparsity=pruning_sparsity)
    
    # Create compressed predictor
    compressed_config = ModelConfig(**config['edge_model'])
    compressed_predictor = EnergyPredictor(compressed_config)
    compressed_predictor.model = pruned_model
    
    # Fine-tune pruned model
    compressed_predictor.train(X_train, y_train, epochs=20)
    
    # Save compressed model
    compressed_predictor.save_model('models/compressed_energy_predictor.h5')
    
    return compressed_predictor


def export_edge_models(models: Dict[str, EnergyPredictor], 
                      X_test: np.ndarray, config: Dict[str, Any]) -> None:
    """
    Export models to edge deployment formats.
    
    Args:
        models: Dictionary of trained models
        X_test: Test data for calibration
        config: Configuration dictionary
    """
    logger.info("Exporting models for edge deployment")
    
    os.makedirs('models/edge', exist_ok=True)
    
    for model_name, model in models.items():
        logger.info(f"Exporting {model_name} model")
        
        exporter = EdgeModelExporter(model.model)
        
        # Export to TFLite
        if 'tflite' in config['deployment']['targets']:
            tflite_path = f'models/edge/{model_name}_int8.tflite'
            exporter.export_to_tflite(
                tflite_path, 
                quantization='int8',
                representative_dataset=X_test[:100]  # Use subset for calibration
            )
            
            # Benchmark TFLite model
            tflite_metrics = exporter.benchmark_model(X_test[:50], tflite_path, 'tflite')
            logger.info(f"TFLite {model_name} metrics: {tflite_metrics}")
        
        # Export to ONNX
        if 'onnx' in config['deployment']['targets']:
            onnx_path = f'models/edge/{model_name}.onnx'
            exporter.export_to_onnx(onnx_path)
            
            # Benchmark ONNX model
            onnx_metrics = exporter.benchmark_model(X_test[:50], onnx_path, 'onnx')
            logger.info(f"ONNX {model_name} metrics: {onnx_metrics}")


def create_evaluation_report(models: Dict[str, EnergyPredictor], 
                            X_test: np.ndarray, y_test: np.ndarray,
                            config: Dict[str, Any]) -> None:
    """
    Create comprehensive evaluation report.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test targets
        config: Configuration dictionary
    """
    logger.info("Creating evaluation report")
    
    os.makedirs('assets', exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate all models
    results = {}
    for model_name, model in models.items():
        metrics = model.evaluate(X_test, y_test)
        predictions = model.predict(X_test)
        
        results[model_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'model_size': model.model.count_params()
        }
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    model_names = list(results.keys())
    mae_values = [results[name]['metrics']['mae'] for name in model_names]
    rmse_values = [results[name]['metrics']['rmse'] for name in model_names]
    r2_values = [results[name]['metrics']['r2'] for name in model_names]
    
    axes[0, 0].bar(model_names, mae_values)
    axes[0, 0].set_title('Mean Absolute Error (kWh)')
    axes[0, 0].set_ylabel('MAE')
    
    axes[0, 1].bar(model_names, rmse_values)
    axes[0, 1].set_title('Root Mean Square Error (kWh)')
    axes[0, 1].set_ylabel('RMSE')
    
    axes[1, 0].bar(model_names, r2_values)
    axes[1, 0].set_title('R² Score')
    axes[1, 0].set_ylabel('R²')
    
    # Model size comparison
    model_sizes = [results[name]['model_size'] for name in model_names]
    axes[1, 1].bar(model_names, model_sizes)
    axes[1, 1].set_title('Model Size (Parameters)')
    axes[1, 1].set_ylabel('Parameters')
    
    plt.tight_layout()
    plt.savefig('assets/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prediction vs actual plots
    fig, axes = plt.subplots(1, len(models), figsize=(5*len(models), 5))
    if len(models) == 1:
        axes = [axes]
    
    for i, (model_name, model) in enumerate(models.items()):
        predictions = results[model_name]['predictions']
        
        axes[i].scatter(y_test, predictions, alpha=0.6)
        axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual Energy (kWh)')
        axes[i].set_ylabel('Predicted Energy (kWh)')
        axes[i].set_title(f'{model_name.title()} Model')
        
        # Add R² score
        r2 = results[model_name]['metrics']['r2']
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('assets/prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    report_path = 'assets/evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("Building Energy Management System - Model Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DISCLAIMER: This system is for research and education purposes only.\n")
        f.write("NOT FOR SAFETY-CRITICAL USE.\n\n")
        
        for model_name, result in results.items():
            f.write(f"{model_name.upper()} MODEL\n")
            f.write("-" * 20 + "\n")
            f.write(f"Parameters: {result['model_size']:,}\n")
            f.write(f"MAE: {result['metrics']['mae']:.3f} kWh\n")
            f.write(f"RMSE: {result['metrics']['rmse']:.3f} kWh\n")
            f.write(f"R²: {result['metrics']['r2']:.3f}\n")
            f.write(f"MAPE: {result['metrics']['mape']:.3f}%\n\n")
    
    logger.info(f"Evaluation report saved to {report_path}")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train Building Energy Management Models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip training and only export existing models')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = load_config(args.config)
    
    logger.info("Starting Building Energy Management System Training")
    logger.info("DISCLAIMER: This system is for research and education purposes only")
    logger.info("NOT FOR SAFETY-CRITICAL USE")
    
    if not args.skip_training:
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(config)
        
        # Train models
        baseline_model = train_baseline_model(X_train, y_train, X_test, y_test, config)
        edge_model = train_edge_optimized_model(X_train, y_train, X_test, y_test, config)
        
        # Apply compression
        compressed_model = apply_model_compression(baseline_model, X_train, y_train, config)
        
        # Collect all models
        models = {
            'baseline': baseline_model,
            'edge': edge_model,
            'compressed': compressed_model
        }
        
        # Create evaluation report
        create_evaluation_report(models, X_test, y_test, config)
        
    else:
        # Load existing models
        logger.info("Loading existing models for export")
        models = {}
        
        model_files = {
            'baseline': 'models/baseline_energy_predictor.h5',
            'edge': 'models/edge_energy_predictor.h5',
            'compressed': 'models/compressed_energy_predictor.h5'
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                model = EnergyPredictor(ModelConfig())
                model.load_model(path)
                models[name] = model
                logger.info(f"Loaded {name} model from {path}")
        
        # Load test data for benchmarking
        if os.path.exists('data/processed/building_sensor_data.csv'):
            df = pd.read_csv('data/processed/building_sensor_data.csv')
            feature_cols = ['hvac_runtime_hours', 'lighting_usage_hours', 
                           'occupancy_count', 'outside_temp_celsius']
            X_test = df[feature_cols].values
            y_test = df['total_energy_kwh'].values
        else:
            logger.error("No test data found. Run training first.")
            return
    
    # Export models for edge deployment
    if models:
        export_edge_models(models, X_test, config)
        logger.info("Edge model export completed")
    
    logger.info("Training pipeline completed successfully")


if __name__ == '__main__':
    main()

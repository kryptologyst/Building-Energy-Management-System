"""
Building Energy Management System - Export Script

Command-line script for exporting trained models to edge deployment formats.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import yaml

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.energy_predictor import EnergyPredictor, ModelConfig
from src.export.edge_exporter import EdgeModelExporter, ModelCompressor
from src.data.data_generator import BuildingDataGenerator, DataConfig
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def export_model(model_path: str, targets: List[str], 
                quantization: str = 'int8', output_dir: str = 'models/edge') -> None:
    """
    Export model to specified edge deployment formats.
    
    Args:
        model_path: Path to trained model
        targets: List of target formats (tflite, onnx, openvino)
        quantization: Quantization type for TFLite
        output_dir: Output directory for exported models
    """
    logger.info(f"Exporting model from {model_path}")
    
    # Load model
    model = EnergyPredictor(ModelConfig())
    model.load_model(model_path)
    
    # Create exporter
    exporter = EdgeModelExporter(model.model)
    
    # Generate calibration data
    generator = BuildingDataGenerator(DataConfig(n_samples=100))
    X_calibration, _ = generator.generate_sensor_data()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export to each target format
    for target in targets:
        logger.info(f"Exporting to {target}")
        
        if target == 'tflite':
            output_path = os.path.join(output_dir, f"model_{quantization}.tflite")
            exporter.export_to_tflite(
                output_path,
                quantization=quantization,
                representative_dataset=X_calibration[:50]
            )
            
            # Benchmark TFLite model
            metrics = exporter.benchmark_model(X_calibration[:20], output_path, 'tflite')
            logger.info(f"TFLite metrics: {metrics}")
            
        elif target == 'onnx':
            output_path = os.path.join(output_dir, "model.onnx")
            exporter.export_to_onnx(output_path)
            
            # Benchmark ONNX model
            metrics = exporter.benchmark_model(X_calibration[:20], output_path, 'onnx')
            logger.info(f"ONNX metrics: {metrics}")
            
        elif target == 'openvino':
            logger.warning("OpenVINO export not implemented in this version")
            
        else:
            logger.error(f"Unsupported target format: {target}")


def main():
    """Main export script."""
    parser = argparse.ArgumentParser(description='Export Building Energy Management Models')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to trained model file')
    parser.add_argument('--targets', nargs='+', default=['tflite'],
                      choices=['tflite', 'onnx', 'openvino'],
                      help='Target deployment formats')
    parser.add_argument('--quantization', type=str, default='int8',
                      choices=['float32', 'float16', 'int16', 'int8'],
                      help='Quantization type for TFLite')
    parser.add_argument('--output-dir', type=str, default='models/edge',
                      help='Output directory for exported models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                      help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("Starting model export process")
    logger.info("DISCLAIMER: This system is for research and education purposes only")
    logger.info("NOT FOR SAFETY-CRITICAL USE")
    
    # Check if model exists
    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return
    
    # Export model
    try:
        export_model(
            model_path=args.model,
            targets=args.targets,
            quantization=args.quantization,
            output_dir=args.output_dir
        )
        logger.info("Model export completed successfully")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == '__main__':
    main()

"""
Building Energy Management System - Main Package

This package provides a comprehensive Edge AI & IoT system for building energy
consumption prediction, optimization, and real-time monitoring.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

__version__ = "1.0.0"
__author__ = "Edge AI Research Team"
__email__ = "research@example.com"
__description__ = "Edge AI Building Energy Management System for Research and Education"

# Import main components
from .models.energy_predictor import (
    EnergyPredictor,
    ModelConfig,
    create_baseline_model,
    create_edge_optimized_model
)

from .data.data_generator import (
    BuildingDataGenerator,
    DataConfig,
    create_default_generator
)

from .export.edge_exporter import (
    EdgeModelExporter,
    ModelCompressor
)

from .utils.evaluator import (
    ModelEvaluator,
    EvaluationMetrics,
    EdgePerformanceMetrics
)

from .utils.logger import (
    setup_logging,
    BEMSLogger
)

from .pipelines.data_pipeline import (
    SensorSimulator,
    MQTTDataStreamer,
    EdgeDataProcessor,
    BuildingDataPipeline,
    BuildingState,
    SensorReading
)

__all__ = [
    # Models
    "EnergyPredictor",
    "ModelConfig", 
    "create_baseline_model",
    "create_edge_optimized_model",
    
    # Data
    "BuildingDataGenerator",
    "DataConfig",
    "create_default_generator",
    
    # Export
    "EdgeModelExporter",
    "ModelCompressor",
    
    # Evaluation
    "ModelEvaluator",
    "EvaluationMetrics",
    "EdgePerformanceMetrics",
    
    # Logging
    "setup_logging",
    "BEMSLogger",
    
    # Pipelines
    "SensorSimulator",
    "MQTTDataStreamer", 
    "EdgeDataProcessor",
    "BuildingDataPipeline",
    "BuildingState",
    "SensorReading",
    
    # Package info
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]

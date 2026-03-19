"""
Building Energy Management System - Test Suite

Unit tests for the BEMS system components.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.energy_predictor import EnergyPredictor, ModelConfig, create_baseline_model
from src.data.data_generator import BuildingDataGenerator, DataConfig, create_default_generator
from src.export.edge_exporter import EdgeModelExporter, ModelCompressor
from src.utils.evaluator import ModelEvaluator
from src.pipelines.data_pipeline import SensorSimulator, BuildingState


class TestEnergyPredictor:
    """Test cases for EnergyPredictor model."""
    
    def test_model_creation(self):
        """Test model creation and configuration."""
        config = ModelConfig(input_features=4, hidden_units=(32, 16))
        model = EnergyPredictor(config)
        
        assert model.config.input_features == 4
        assert model.config.hidden_units == (32, 16)
        assert model.model is None  # Not built yet
    
    def test_model_building(self):
        """Test model architecture building."""
        config = ModelConfig(input_features=4, hidden_units=(32, 16))
        model = EnergyPredictor(config)
        built_model = model.build_model()
        
        assert built_model is not None
        assert model.model is not None
        assert model.model.input_shape == (None, 4)
        assert model.model.output_shape == (None, 1)
    
    def test_model_training(self):
        """Test model training with synthetic data."""
        # Generate synthetic data
        generator = BuildingDataGenerator(DataConfig(n_samples=100))
        X, y = generator.generate_sensor_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        config = ModelConfig(epochs=5)  # Short training for testing
        model = EnergyPredictor(config)
        model.build_model()
        
        # Train model
        history = model.train(X_train, y_train, X_test, y_test)
        
        assert history is not None
        assert 'loss' in history
        assert len(history['loss']) > 0
    
    def test_model_prediction(self):
        """Test model prediction."""
        # Generate synthetic data
        generator = BuildingDataGenerator(DataConfig(n_samples=50))
        X, y = generator.generate_sensor_data()
        
        # Create and train model
        config = ModelConfig(epochs=5)
        model = EnergyPredictor(config)
        model.build_model()
        model.train(X, y, epochs=5)
        
        # Make predictions
        predictions = model.predict(X[:10])
        
        assert predictions is not None
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float)) for p in predictions)
    
    def test_model_evaluation(self):
        """Test model evaluation."""
        # Generate synthetic data
        generator = BuildingDataGenerator(DataConfig(n_samples=100))
        X, y = generator.generate_sensor_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        config = ModelConfig(epochs=5)
        model = EnergyPredictor(config)
        model.build_model()
        model.train(X_train, y_train, epochs=5)
        
        # Evaluate model
        metrics = model.evaluate(X_test, y_test)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_model_save_load(self):
        """Test model saving and loading."""
        # Generate synthetic data
        generator = BuildingDataGenerator(DataConfig(n_samples=50))
        X, y = generator.generate_sensor_data()
        
        # Create and train model
        config = ModelConfig(epochs=5)
        model = EnergyPredictor(config)
        model.build_model()
        model.train(X, y, epochs=5)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            model_path = tmp_file.name
        
        try:
            model.save_model(model_path)
            assert os.path.exists(model_path)
            
            # Load model
            loaded_model = EnergyPredictor(config)
            loaded_model.load_model(model_path)
            
            assert loaded_model.model is not None
            assert loaded_model.model.input_shape == model.model.input_shape
            
            # Test predictions are similar
            original_preds = model.predict(X[:5])
            loaded_preds = loaded_model.predict(X[:5])
            
            np.testing.assert_array_almost_equal(original_preds, loaded_preds, decimal=5)
            
        finally:
            if os.path.exists(model_path):
                os.unlink(model_path)


class TestDataGenerator:
    """Test cases for BuildingDataGenerator."""
    
    def test_data_generation(self):
        """Test synthetic data generation."""
        config = DataConfig(n_samples=100)
        generator = BuildingDataGenerator(config)
        
        X, y = generator.generate_sensor_data()
        
        assert X.shape == (100, 4)
        assert y.shape == (100,)
        assert all(isinstance(val, (int, float)) for val in X.flatten())
        assert all(isinstance(val, (int, float)) for val in y)
    
    def test_dataframe_creation(self):
        """Test DataFrame creation."""
        config = DataConfig(n_samples=50)
        generator = BuildingDataGenerator(config)
        
        X, y = generator.generate_sensor_data()
        df = generator.create_dataframe(X, y)
        
        assert len(df) == 50
        assert 'hvac_runtime_hours' in df.columns
        assert 'lighting_usage_hours' in df.columns
        assert 'occupancy_count' in df.columns
        assert 'outside_temp_celsius' in df.columns
        assert 'total_energy_kwh' in df.columns
        assert 'timestamp' in df.columns
    
    def test_data_save_load(self):
        """Test data saving and loading."""
        config = DataConfig(n_samples=30)
        generator = BuildingDataGenerator(config)
        
        X, y = generator.generate_sensor_data()
        
        # Save data
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
            data_path = tmp_file.name
        
        try:
            generator.save_data(X, y, data_path)
            assert os.path.exists(data_path)
            
            # Load data
            X_loaded, y_loaded = generator.load_data(data_path)
            
            np.testing.assert_array_almost_equal(X, X_loaded)
            np.testing.assert_array_almost_equal(y, y_loaded)
            
        finally:
            if os.path.exists(data_path):
                os.unlink(data_path)


class TestEdgeExporter:
    """Test cases for EdgeModelExporter."""
    
    def test_exporter_initialization(self):
        """Test exporter initialization."""
        # Create a simple model
        config = ModelConfig()
        model = EnergyPredictor(config)
        model.build_model()
        
        exporter = EdgeModelExporter(model.model)
        
        assert exporter.model is not None
        assert exporter.input_shape == (4,)
    
    def test_tflite_export(self):
        """Test TFLite model export."""
        # Create and train a simple model
        generator = BuildingDataGenerator(DataConfig(n_samples=50))
        X, y = generator.generate_sensor_data()
        
        config = ModelConfig(epochs=5)
        model = EnergyPredictor(config)
        model.build_model()
        model.train(X, y, epochs=5)
        
        exporter = EdgeModelExporter(model.model)
        
        # Export to TFLite
        with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp_file:
            tflite_path = tmp_file.name
        
        try:
            exporter.export_to_tflite(tflite_path, quantization='float32')
            assert os.path.exists(tflite_path)
            assert os.path.getsize(tflite_path) > 0
            
        finally:
            if os.path.exists(tflite_path):
                os.unlink(tflite_path)


class TestModelEvaluator:
    """Test cases for ModelEvaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ModelEvaluator()
        assert evaluator.results == {}
    
    def test_accuracy_evaluation(self):
        """Test accuracy metrics calculation."""
        evaluator = ModelEvaluator()
        
        # Create synthetic data
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0, 51.0])
        
        metrics = evaluator.evaluate_accuracy(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert 'mape' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())
    
    def test_model_comparison(self):
        """Test model comparison functionality."""
        evaluator = ModelEvaluator()
        
        # Create synthetic data
        generator = BuildingDataGenerator(DataConfig(n_samples=50))
        X, y = generator.generate_sensor_data()
        
        # Create simple models
        config = ModelConfig(epochs=5)
        model1 = EnergyPredictor(config)
        model1.build_model()
        model1.train(X, y, epochs=5)
        
        config2 = ModelConfig(hidden_units=(16, 8), epochs=5)
        model2 = EnergyPredictor(config2)
        model2.build_model()
        model2.train(X, y, epochs=5)
        
        models = {'model1': model1, 'model2': model2}
        
        comparison_df = evaluator.compare_models(models, X, y)
        
        assert len(comparison_df) == 2
        assert 'model_name' in comparison_df.columns
        assert 'mae' in comparison_df.columns
        assert 'rmse' in comparison_df.columns


class TestSensorSimulator:
    """Test cases for SensorSimulator."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = SensorSimulator("test_building")
        
        assert simulator.building_id == "test_building"
        assert simulator.current_time is not None
        assert simulator.state is not None
    
    def test_sensor_reading_generation(self):
        """Test sensor reading generation."""
        simulator = SensorSimulator("test_building")
        
        # Generate next reading
        state = simulator.generate_next_reading()
        
        assert isinstance(state, BuildingState)
        assert state.building_id == "test_building"
        assert state.hvac_runtime >= 0
        assert state.lighting_usage >= 0
        assert state.occupancy >= 0
        assert state.energy_consumption >= 0


class TestIntegration:
    """Integration tests."""
    
    @pytest.mark.integration
    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # Generate data
        generator = BuildingDataGenerator(DataConfig(n_samples=100))
        X, y = generator.generate_sensor_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train baseline model
        baseline_model = create_baseline_model()
        baseline_model.train(X_train, y_train, X_test, y_test)
        
        # Train edge model
        edge_model = create_edge_optimized_model()
        edge_model.train(X_train, y_train, X_test, y_test)
        
        # Evaluate models
        evaluator = ModelEvaluator()
        models = {'baseline': baseline_model, 'edge': edge_model}
        comparison_df = evaluator.compare_models(models, X_test, y_test)
        
        assert len(comparison_df) == 2
        assert comparison_df['mae'].min() > 0  # Should have some error
    
    @pytest.mark.slow
    def test_model_compression_pipeline(self):
        """Test model compression pipeline."""
        # Generate data
        generator = BuildingDataGenerator(DataConfig(n_samples=100))
        X, y = generator.generate_sensor_data()
        
        # Train baseline model
        baseline_model = create_baseline_model()
        baseline_model.train(X, y, epochs=10)
        
        # Apply compression
        compressor = ModelCompressor(baseline_model.model)
        compressed_model = compressor.apply_pruning(sparsity=0.3)
        
        assert compressed_model is not None
        assert compressed_model.count_params() < baseline_model.model.count_params()


# Import train_test_split for tests
from sklearn.model_selection import train_test_split

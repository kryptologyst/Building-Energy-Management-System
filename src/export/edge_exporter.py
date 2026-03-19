"""
Building Energy Management System - Edge Deployment and Quantization

This module handles model quantization and edge deployment for building energy
prediction models, supporting TFLite, ONNX, and OpenVINO runtimes.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import numpy as np
import tensorflow as tf
import os
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EdgeModelExporter:
    """
    Exports TensorFlow models to various edge deployment formats.
    
    Supports:
    - TensorFlow Lite (TFLite) for mobile/embedded
    - ONNX for cross-platform deployment
    - OpenVINO for Intel hardware acceleration
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize the edge exporter.
        
        Args:
            model: Trained Keras model to export
        """
        self.model = model
        self.input_shape = model.input_shape[1:]  # Remove batch dimension
        
    def export_to_tflite(self, output_path: str, 
                        quantization: str = 'int8',
                        representative_dataset: Optional[np.ndarray] = None) -> str:
        """
        Export model to TensorFlow Lite format.
        
        Args:
            output_path: Output file path
            quantization: Quantization type ('float32', 'int8', 'int16')
            representative_dataset: Calibration data for quantization
            
        Returns:
            Path to exported TFLite model
        """
        logger.info(f"Exporting model to TFLite with {quantization} quantization")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        
        if quantization == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_dataset is not None:
                converter.representative_dataset = self._create_representative_dataset(
                    representative_dataset
                )
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        elif quantization == 'int16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.int16]
        elif quantization == 'float16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size = len(tflite_model) / 1024  # KB
        logger.info(f"TFLite model exported to {output_path} ({model_size:.1f} KB)")
        
        return output_path
    
    def _create_representative_dataset(self, data: np.ndarray):
        """Create representative dataset for quantization calibration."""
        def representative_data_gen():
            for i in range(min(100, len(data))):  # Use up to 100 samples
                yield [data[i:i+1].astype(np.float32)]
        return representative_data_gen
    
    def export_to_onnx(self, output_path: str) -> str:
        """
        Export model to ONNX format.
        
        Args:
            output_path: Output file path
            
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting model to ONNX format")
        
        try:
            import tf2onnx
            import onnx
            
            # Convert to ONNX
            spec = (tf.TensorSpec((None,) + self.input_shape, tf.float32, name="input"),)
            onnx_model, _ = tf2onnx.convert.from_keras(
                self.model, 
                input_signature=spec,
                opset=13
            )
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            model_size = len(onnx_model.SerializeToString()) / 1024  # KB
            logger.info(f"ONNX model exported to {output_path} ({model_size:.1f} KB)")
            
            return output_path
            
        except ImportError:
            logger.error("tf2onnx not installed. Install with: pip install tf2onnx")
            raise
    
    def benchmark_model(self, test_data: np.ndarray, 
                       model_path: Optional[str] = None,
                       runtime: str = 'tflite') -> Dict[str, float]:
        """
        Benchmark model performance on edge hardware.
        
        Args:
            test_data: Test input data
            model_path: Path to exported model (if None, uses original model)
            runtime: Runtime type ('tflite', 'onnx', 'tf')
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"Benchmarking model with {runtime} runtime")
        
        if runtime == 'tflite' and model_path:
            return self._benchmark_tflite(model_path, test_data)
        elif runtime == 'onnx' and model_path:
            return self._benchmark_onnx(model_path, test_data)
        else:
            return self._benchmark_tf(test_data)
    
    def _benchmark_tflite(self, model_path: str, test_data: np.ndarray) -> Dict[str, float]:
        """Benchmark TFLite model performance."""
        import time
        
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], test_data[:1])
            interpreter.invoke()
        
        # Benchmark
        times = []
        for i in range(len(test_data)):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], test_data[i:i+1])
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])
            times.append(time.time() - start_time)
        
        return {
            'avg_latency_ms': np.mean(times) * 1000,
            'p95_latency_ms': np.percentile(times, 95) * 1000,
            'throughput_fps': 1.0 / np.mean(times),
            'model_size_kb': os.path.getsize(model_path) / 1024
        }
    
    def _benchmark_onnx(self, model_path: str, test_data: np.ndarray) -> Dict[str, float]:
        """Benchmark ONNX model performance."""
        try:
            import onnxruntime as ort
            import time
            
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                session.run(None, {input_name: test_data[:1].astype(np.float32)})
            
            # Benchmark
            times = []
            for i in range(len(test_data)):
                start_time = time.time()
                session.run(None, {input_name: test_data[i:i+1].astype(np.float32)})
                times.append(time.time() - start_time)
            
            return {
                'avg_latency_ms': np.mean(times) * 1000,
                'p95_latency_ms': np.percentile(times, 95) * 1000,
                'throughput_fps': 1.0 / np.mean(times),
                'model_size_kb': os.path.getsize(model_path) / 1024
            }
            
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime")
            return {}
    
    def _benchmark_tf(self, test_data: np.ndarray) -> Dict[str, float]:
        """Benchmark TensorFlow model performance."""
        import time
        
        # Warmup
        for _ in range(10):
            self.model.predict(test_data[:1], verbose=0)
        
        # Benchmark
        times = []
        for i in range(len(test_data)):
            start_time = time.time()
            self.model.predict(test_data[i:i+1], verbose=0)
            times.append(time.time() - start_time)
        
        return {
            'avg_latency_ms': np.mean(times) * 1000,
            'p95_latency_ms': np.percentile(times, 95) * 1000,
            'throughput_fps': 1.0 / np.mean(times),
            'model_size_kb': 0  # TF model size not easily measurable
        }


class ModelCompressor:
    """
    Applies compression techniques to reduce model size for edge deployment.
    
    Supports:
    - Post-training quantization
    - Pruning
    - Knowledge distillation
    """
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize the model compressor.
        
        Args:
            model: Keras model to compress
        """
        self.model = model
    
    def apply_pruning(self, sparsity: float = 0.5) -> tf.keras.Model:
        """
        Apply magnitude-based pruning to the model.
        
        Args:
            sparsity: Target sparsity level (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        logger.info(f"Applying pruning with {sparsity:.1%} sparsity")
        
        try:
            import tensorflow_model_optimization as tfmot
            
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=sparsity,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            model_for_pruning = prune_low_magnitude(self.model, **pruning_params)
            
            # Compile the pruned model
            model_for_pruning.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return model_for_pruning
            
        except ImportError:
            logger.error("tensorflow-model-optimization not installed")
            logger.info("Install with: pip install tensorflow-model-optimization")
            return self.model
    
    def create_student_model(self, teacher_model: tf.keras.Model) -> tf.keras.Model:
        """
        Create a smaller student model for knowledge distillation.
        
        Args:
            teacher_model: Larger teacher model
            
        Returns:
            Smaller student model
        """
        logger.info("Creating student model for knowledge distillation")
        
        # Create smaller architecture
        student_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.model.input_shape[1:]),
            tf.keras.layers.Dense(16, activation='relu', name='student_dense_1'),
            tf.keras.layers.Dense(8, activation='relu', name='student_dense_2'),
            tf.keras.layers.Dense(1, name='student_output')
        ])
        
        student_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Student model created with {student_model.count_params():,} parameters")
        return student_model
    
    def distill_knowledge(self, teacher_model: tf.keras.Model, 
                         X_train: np.ndarray, y_train: np.ndarray,
                         temperature: float = 3.0,
                         alpha: float = 0.7) -> tf.keras.Model:
        """
        Perform knowledge distillation from teacher to student model.
        
        Args:
            teacher_model: Pre-trained teacher model
            X_train: Training data
            y_train: Training labels
            temperature: Distillation temperature
            alpha: Weight for distillation loss
            
        Returns:
            Trained student model
        """
        logger.info("Performing knowledge distillation")
        
        student_model = self.create_student_model(teacher_model)
        
        # Get teacher predictions
        teacher_preds = teacher_model.predict(X_train, verbose=0)
        
        # Custom training loop for distillation
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        for epoch in range(50):
            with tf.GradientTape() as tape:
                student_preds = student_model(X_train, training=True)
                
                # Hard loss (MSE with true labels)
                hard_loss = tf.keras.losses.mse(y_train, student_preds)
                
                # Soft loss (MSE with teacher predictions)
                soft_loss = tf.keras.losses.mse(teacher_preds, student_preds)
                
                # Combined loss
                total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
            
            gradients = tape.gradient(total_loss, student_model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))
            
            if epoch % 10 == 0:
                logger.info(f"Distillation epoch {epoch}, loss: {total_loss:.4f}")
        
        logger.info("Knowledge distillation completed")
        return student_model

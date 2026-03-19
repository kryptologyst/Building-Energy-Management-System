"""
Building Energy Management System - Data Generation and Processing

This module handles synthetic building sensor data generation and preprocessing
for energy consumption prediction models.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class SensorData:
    """Container for building sensor readings."""
    hvac_runtime: float  # hours per day
    lighting_usage: float  # hours per day
    occupancy: int  # people count
    outside_temp: float  # Celsius
    timestamp: datetime
    building_id: str = "building_001"


@dataclass
class DataConfig:
    """Configuration for synthetic data generation."""
    n_samples: int = 2000
    hvac_mean: float = 5.0
    hvac_std: float = 2.0
    lighting_mean: float = 6.0
    lighting_std: float = 1.5
    occupancy_min: int = 1
    occupancy_max: int = 20
    temp_mean: float = 25.0
    temp_std: float = 5.0
    noise_std: float = 2.0
    seasonal_factor: float = 0.3


class BuildingDataGenerator:
    """
    Generates synthetic building sensor data for energy management training.
    
    Simulates realistic patterns including:
    - Seasonal temperature variations
    - Occupancy patterns (weekday/weekend)
    - HVAC and lighting correlations
    - Energy consumption relationships
    """
    
    def __init__(self, config: DataConfig):
        """
        Initialize the data generator.
        
        Args:
            config: Data generation configuration
        """
        self.config = config
        np.random.seed(42)  # Deterministic generation
        
    def generate_sensor_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic building sensor data.
        
        Returns:
            Tuple of (features, targets) where features are [hvac, lighting, occupancy, temp]
            and targets are total energy consumption in kWh
        """
        logger.info(f"Generating {self.config.n_samples} samples of building sensor data")
        
        # Generate base features with realistic correlations
        hvac_runtime = np.random.normal(
            self.config.hvac_mean, 
            self.config.hvac_std, 
            self.config.n_samples
        )
        hvac_runtime = np.clip(hvac_runtime, 0, 24)  # Constrain to valid range
        
        lighting_usage = np.random.normal(
            self.config.lighting_mean, 
            self.config.lighting_std, 
            self.config.n_samples
        )
        lighting_usage = np.clip(lighting_usage, 0, 24)
        
        occupancy = np.random.randint(
            self.config.occupancy_min, 
            self.config.occupancy_max + 1, 
            self.config.n_samples
        )
        
        outside_temp = np.random.normal(
            self.config.temp_mean, 
            self.config.temp_std, 
            self.config.n_samples
        )
        
        # Add seasonal patterns
        seasonal_temp = self._add_seasonal_pattern(outside_temp)
        
        # Add realistic correlations
        hvac_runtime = self._add_hvac_correlation(hvac_runtime, seasonal_temp)
        lighting_usage = self._add_lighting_correlation(lighting_usage, occupancy)
        
        # Generate energy consumption with realistic physics
        total_energy = self._calculate_energy_consumption(
            hvac_runtime, lighting_usage, occupancy, seasonal_temp
        )
        
        # Create feature matrix
        features = np.stack([
            hvac_runtime, 
            lighting_usage, 
            occupancy, 
            seasonal_temp
        ], axis=1)
        
        logger.info(f"Generated data - Energy range: {total_energy.min():.1f} to {total_energy.max():.1f} kWh")
        return features, total_energy
    
    def _add_seasonal_pattern(self, base_temp: np.ndarray) -> np.ndarray:
        """Add seasonal temperature variations."""
        # Simulate seasonal sine wave
        seasonal_cycle = np.sin(np.linspace(0, 4 * np.pi, len(base_temp))) * self.config.seasonal_factor
        return base_temp + seasonal_cycle * 10  # ±10°C seasonal variation
    
    def _add_hvac_correlation(self, hvac_base: np.ndarray, temp: np.ndarray) -> np.ndarray:
        """Add temperature-dependent HVAC usage."""
        # HVAC runs more when temperature deviates from comfort zone (22°C)
        temp_factor = np.abs(temp - 22) / 10  # Normalize temperature deviation
        hvac_adjustment = temp_factor * 2  # Additional 0-4 hours based on temp
        
        return np.clip(hvac_base + hvac_adjustment, 0, 24)
    
    def _add_lighting_correlation(self, lighting_base: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        """Add occupancy-dependent lighting usage."""
        # More occupancy = more lighting
        occupancy_factor = (occupancy - 1) / 19  # Normalize to 0-1
        lighting_adjustment = occupancy_factor * 3  # Additional 0-3 hours based on occupancy
        
        return np.clip(lighting_base + lighting_adjustment, 0, 24)
    
    def _calculate_energy_consumption(self, hvac: np.ndarray, lighting: np.ndarray, 
                                   occupancy: np.ndarray, temp: np.ndarray) -> np.ndarray:
        """
        Calculate realistic energy consumption based on building physics.
        
        Energy consumption model:
        - HVAC: 3.5 kWh per hour of runtime + temperature-dependent load
        - Lighting: 1.2 kWh per hour of usage
        - Occupancy: 0.6 kWh per person (computers, appliances)
        - Temperature compensation: additional cooling/heating load
        """
        hvac_energy = hvac * 3.5
        lighting_energy = lighting * 1.2
        occupancy_energy = occupancy * 0.6
        
        # Temperature-dependent energy (cooling/heating load)
        temp_energy = np.abs(temp - 22) * 0.8  # 0.8 kWh per degree deviation
        
        # Base building energy (always-on systems)
        base_energy = np.full_like(hvac, 5.0)  # 5 kWh base consumption
        
        # Add realistic noise
        noise = np.random.normal(0, self.config.noise_std, len(hvac))
        
        total_energy = (hvac_energy + lighting_energy + occupancy_energy + 
                       temp_energy + base_energy + noise)
        
        return np.clip(total_energy, 0, None)  # Ensure non-negative
    
    def create_dataframe(self, features: np.ndarray, targets: np.ndarray) -> pd.DataFrame:
        """
        Create a pandas DataFrame with proper column names and metadata.
        
        Args:
            features: Feature matrix
            targets: Target values
            
        Returns:
            DataFrame with sensor data
        """
        df = pd.DataFrame(features, columns=[
            'hvac_runtime_hours',
            'lighting_usage_hours', 
            'occupancy_count',
            'outside_temp_celsius'
        ])
        
        df['total_energy_kwh'] = targets
        df['timestamp'] = pd.date_range(
            start='2024-01-01', 
            periods=len(df), 
            freq='D'
        )
        
        # Add derived features
        df['energy_per_person'] = df['total_energy_kwh'] / df['occupancy_count']
        df['hvac_efficiency'] = df['total_energy_kwh'] / (df['hvac_runtime_hours'] + 0.1)
        
        logger.info(f"Created DataFrame with {len(df)} samples and {len(df.columns)} features")
        return df
    
    def save_data(self, features: np.ndarray, targets: np.ndarray, 
                  filepath: str) -> None:
        """
        Save generated data to file.
        
        Args:
            features: Feature matrix
            targets: Target values
            filepath: Output file path
        """
        df = self.create_dataframe(features, targets)
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            df.to_json(filepath, orient='records', date_format='iso')
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        logger.info(f"Data saved to {filepath}")
    
    def load_data(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data from file.
        
        Args:
            filepath: Input file path
            
        Returns:
            Tuple of (features, targets)
        """
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format. Use .csv or .json")
        
        feature_cols = [
            'hvac_runtime_hours',
            'lighting_usage_hours',
            'occupancy_count', 
            'outside_temp_celsius'
        ]
        
        features = df[feature_cols].values
        targets = df['total_energy_kwh'].values
        
        logger.info(f"Loaded {len(df)} samples from {filepath}")
        return features, targets


def create_default_generator() -> BuildingDataGenerator:
    """
    Create a data generator with default configuration.
    
    Returns:
        Configured BuildingDataGenerator instance
    """
    config = DataConfig(
        n_samples=2000,
        hvac_mean=5.0,
        hvac_std=2.0,
        lighting_mean=6.0,
        lighting_std=1.5,
        occupancy_min=1,
        occupancy_max=20,
        temp_mean=25.0,
        temp_std=5.0,
        noise_std=2.0
    )
    return BuildingDataGenerator(config)

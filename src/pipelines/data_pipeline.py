"""
Building Energy Management System - IoT Data Pipeline

This module implements realistic building sensor data streaming with MQTT simulation,
real-time data processing, and edge-constrained inference pipelines.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import threading
from collections import deque
import random

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    logging.warning("paho-mqtt not available. MQTT features disabled.")

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Container for individual sensor readings."""
    timestamp: datetime
    building_id: str
    sensor_type: str
    value: float
    unit: str
    quality: float = 1.0  # Data quality score (0-1)


@dataclass
class BuildingState:
    """Container for complete building state."""
    timestamp: datetime
    building_id: str
    hvac_runtime: float  # hours
    lighting_usage: float  # hours
    occupancy: int  # people count
    outside_temp: float  # Celsius
    energy_consumption: float  # kWh
    hvac_status: str = "on"  # on/off
    lighting_status: str = "on"  # on/off


class SensorSimulator:
    """
    Simulates realistic building sensor data with temporal patterns.
    
    Generates correlated sensor readings that reflect real building behavior:
    - Seasonal temperature variations
    - Occupancy patterns (weekday/weekend)
    - HVAC and lighting correlations
    - Equipment status changes
    """
    
    def __init__(self, building_id: str = "building_001", 
                 base_config: Optional[Dict[str, Any]] = None):
        """
        Initialize sensor simulator.
        
        Args:
            building_id: Unique building identifier
            base_config: Base configuration for sensor behavior
        """
        self.building_id = building_id
        self.config = base_config or self._get_default_config()
        
        # Initialize state
        self.current_time = datetime.now()
        self.state = self._initialize_state()
        
        # Simulation parameters
        self.seasonal_offset = random.uniform(0, 2 * np.pi)
        self.occupancy_pattern = self._create_occupancy_pattern()
        
        logger.info(f"Initialized sensor simulator for {building_id}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default sensor configuration."""
        return {
            'hvac': {
                'base_runtime': 5.0,
                'temp_sensitivity': 0.3,
                'occupancy_factor': 0.1,
                'noise_std': 0.5
            },
            'lighting': {
                'base_usage': 6.0,
                'occupancy_factor': 0.2,
                'seasonal_factor': 0.1,
                'noise_std': 0.3
            },
            'occupancy': {
                'base_count': 8,
                'weekday_multiplier': 1.5,
                'hourly_pattern': [0.1, 0.1, 0.1, 0.1, 0.2, 0.4, 0.7, 0.9, 1.0, 0.9, 0.8, 0.8, 0.7, 0.8, 0.9, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1],
                'noise_std': 1.0
            },
            'temperature': {
                'base_temp': 25.0,
                'seasonal_amplitude': 8.0,
                'daily_amplitude': 3.0,
                'noise_std': 1.0
            }
        }
    
    def _initialize_state(self) -> BuildingState:
        """Initialize building state."""
        return BuildingState(
            timestamp=self.current_time,
            building_id=self.building_id,
            hvac_runtime=0.0,
            lighting_usage=0.0,
            occupancy=0,
            outside_temp=25.0,
            energy_consumption=0.0
        )
    
    def _create_occupancy_pattern(self) -> List[float]:
        """Create realistic occupancy pattern."""
        # 24-hour pattern with peak during business hours
        pattern = []
        for hour in range(24):
            if 6 <= hour <= 18:  # Business hours
                base = 0.3 + 0.7 * np.sin((hour - 6) * np.pi / 12)
            else:  # Off hours
                base = 0.1
            pattern.append(max(0.05, base))
        return pattern
    
    def generate_next_reading(self, time_step: timedelta = timedelta(minutes=15)) -> BuildingState:
        """
        Generate next sensor reading based on current state and time.
        
        Args:
            time_step: Time increment for simulation
            
        Returns:
            Updated building state
        """
        self.current_time += time_step
        
        # Update temperature with seasonal and daily patterns
        temp = self._update_temperature()
        
        # Update occupancy based on time patterns
        occupancy = self._update_occupancy()
        
        # Update HVAC based on temperature and occupancy
        hvac_runtime = self._update_hvac(temp, occupancy)
        
        # Update lighting based on occupancy and time
        lighting_usage = self._update_lighting(occupancy)
        
        # Calculate energy consumption
        energy = self._calculate_energy_consumption(hvac_runtime, lighting_usage, occupancy, temp)
        
        # Update state
        self.state = BuildingState(
            timestamp=self.current_time,
            building_id=self.building_id,
            hvac_runtime=hvac_runtime,
            lighting_usage=lighting_usage,
            occupancy=occupancy,
            outside_temp=temp,
            energy_consumption=energy,
            hvac_status="on" if hvac_runtime > 0.1 else "off",
            lighting_status="on" if lighting_usage > 0.1 else "off"
        )
        
        return self.state
    
    def _update_temperature(self) -> float:
        """Update outside temperature with seasonal and daily patterns."""
        config = self.config['temperature']
        
        # Seasonal variation (yearly cycle)
        day_of_year = self.current_time.timetuple().tm_yday
        seasonal_temp = config['base_temp'] + config['seasonal_amplitude'] * np.sin(
            2 * np.pi * day_of_year / 365 + self.seasonal_offset
        )
        
        # Daily variation
        hour = self.current_time.hour + self.current_time.minute / 60.0
        daily_temp = seasonal_temp + config['daily_amplitude'] * np.sin(
            2 * np.pi * (hour - 6) / 24
        )
        
        # Add noise
        noise = np.random.normal(0, config['noise_std'])
        
        return max(-10, min(45, daily_temp + noise))  # Clamp to realistic range
    
    def _update_occupancy(self) -> int:
        """Update occupancy based on time patterns."""
        config = self.config['occupancy']
        
        # Get hourly pattern
        hour = self.current_time.hour
        hourly_factor = self.occupancy_pattern[hour]
        
        # Weekday/weekend factor
        weekday_factor = config['weekday_multiplier'] if self.current_time.weekday() < 5 else 0.3
        
        # Calculate occupancy
        occupancy = int(config['base_count'] * hourly_factor * weekday_factor)
        
        # Add noise
        noise = np.random.normal(0, config['noise_std'])
        occupancy = max(0, occupancy + int(noise))
        
        return occupancy
    
    def _update_hvac(self, temp: float, occupancy: int) -> float:
        """Update HVAC runtime based on temperature and occupancy."""
        config = self.config['hvac']
        
        # Base runtime
        base_runtime = config['base_runtime']
        
        # Temperature-dependent runtime
        temp_deviation = abs(temp - 22)  # Comfort zone around 22°C
        temp_factor = config['temp_sensitivity'] * temp_deviation
        
        # Occupancy factor
        occupancy_factor = config['occupancy_factor'] * occupancy / 10
        
        # Calculate runtime
        runtime = base_runtime + temp_factor + occupancy_factor
        
        # Add noise
        noise = np.random.normal(0, config['noise_std'])
        runtime = max(0, runtime + noise)
        
        return min(24, runtime)  # Clamp to 24 hours
    
    def _update_lighting(self, occupancy: int) -> float:
        """Update lighting usage based on occupancy and time."""
        config = self.config['lighting']
        
        # Base usage
        base_usage = config['base_usage']
        
        # Occupancy factor
        occupancy_factor = config['occupancy_factor'] * occupancy / 10
        
        # Time-based factor (more lighting during dark hours)
        hour = self.current_time.hour
        if 6 <= hour <= 18:  # Daylight hours
            time_factor = 0.3
        else:  # Dark hours
            time_factor = 1.0
        
        # Calculate usage
        usage = base_usage * time_factor + occupancy_factor
        
        # Add noise
        noise = np.random.normal(0, config['noise_std'])
        usage = max(0, usage + noise)
        
        return min(24, usage)  # Clamp to 24 hours
    
    def _calculate_energy_consumption(self, hvac: float, lighting: float, 
                                    occupancy: int, temp: float) -> float:
        """Calculate total energy consumption."""
        # HVAC energy (3.5 kWh per hour + temperature load)
        hvac_energy = hvac * 3.5 + abs(temp - 22) * 0.8
        
        # Lighting energy (1.2 kWh per hour)
        lighting_energy = lighting * 1.2
        
        # Occupancy energy (0.6 kWh per person)
        occupancy_energy = occupancy * 0.6
        
        # Base building energy
        base_energy = 5.0
        
        # Add noise
        noise = np.random.normal(0, 1.0)
        
        total_energy = hvac_energy + lighting_energy + occupancy_energy + base_energy + noise
        
        return max(0, total_energy)


class MQTTDataStreamer:
    """
    MQTT-based data streaming for building sensor data.
    
    Publishes sensor readings to MQTT topics for real-time monitoring
    and edge device consumption.
    """
    
    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883,
                 topic_prefix: str = "bems/sensors", qos: int = 1):
        """
        Initialize MQTT data streamer.
        
        Args:
            broker_host: MQTT broker hostname
            broker_port: MQTT broker port
            topic_prefix: Topic prefix for sensor data
            qos: Quality of Service level
        """
        if not MQTT_AVAILABLE:
            raise ImportError("paho-mqtt not available. Install with: pip install paho-mqtt")
        
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_prefix = topic_prefix
        self.qos = qos
        
        # MQTT client
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        
        self.connected = False
        self.running = False
        
        logger.info(f"Initialized MQTT streamer for {broker_host}:{broker_port}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Handle MQTT connection."""
        if rc == 0:
            self.connected = True
            logger.info("Connected to MQTT broker")
        else:
            logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection."""
        self.connected = False
        logger.info("Disconnected from MQTT broker")
    
    def connect(self) -> bool:
        """
        Connect to MQTT broker.
        
        Returns:
            True if connection successful
        """
        try:
            self.client.connect(self.broker_host, self.broker_port, 60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 10
            while not self.connected and timeout > 0:
                time.sleep(0.1)
                timeout -= 0.1
            
            return self.connected
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from MQTT broker."""
        self.running = False
        self.client.loop_stop()
        self.client.disconnect()
        logger.info("Disconnected from MQTT broker")
    
    def publish_sensor_reading(self, reading: SensorReading) -> bool:
        """
        Publish sensor reading to MQTT topic.
        
        Args:
            reading: Sensor reading to publish
            
        Returns:
            True if publish successful
        """
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        topic = f"{self.topic_prefix}/{reading.building_id}/{reading.sensor_type}"
        
        payload = {
            'timestamp': reading.timestamp.isoformat(),
            'building_id': reading.building_id,
            'sensor_type': reading.sensor_type,
            'value': reading.value,
            'unit': reading.unit,
            'quality': reading.quality
        }
        
        try:
            result = self.client.publish(topic, json.dumps(payload), qos=self.qos)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Failed to publish sensor reading: {e}")
            return False
    
    def publish_building_state(self, state: BuildingState) -> bool:
        """
        Publish complete building state to MQTT.
        
        Args:
            state: Building state to publish
            
        Returns:
            True if publish successful
        """
        if not self.connected:
            logger.warning("Not connected to MQTT broker")
            return False
        
        topic = f"{self.topic_prefix}/{state.building_id}/state"
        
        payload = asdict(state)
        payload['timestamp'] = state.timestamp.isoformat()
        
        try:
            result = self.client.publish(topic, json.dumps(payload), qos=self.qos)
            return result.rc == mqtt.MQTT_ERR_SUCCESS
        except Exception as e:
            logger.error(f"Failed to publish building state: {e}")
            return False


class EdgeDataProcessor:
    """
    Edge-constrained data processor for real-time building energy analysis.
    
    Processes streaming sensor data with memory and compute constraints
    typical of edge devices.
    """
    
    def __init__(self, buffer_size: int = 100, 
                 processing_interval: float = 1.0):
        """
        Initialize edge data processor.
        
        Args:
            buffer_size: Maximum number of readings to keep in memory
            processing_interval: Processing interval in seconds
        """
        self.buffer_size = buffer_size
        self.processing_interval = processing_interval
        
        # Data buffers
        self.sensor_buffer = deque(maxlen=buffer_size)
        self.state_buffer = deque(maxlen=buffer_size)
        
        # Processing state
        self.last_processing_time = time.time()
        self.processing_stats = {
            'total_readings': 0,
            'processed_readings': 0,
            'errors': 0,
            'avg_processing_time': 0.0
        }
        
        logger.info(f"Initialized edge processor with buffer size {buffer_size}")
    
    def add_sensor_reading(self, reading: SensorReading) -> None:
        """
        Add sensor reading to processing buffer.
        
        Args:
            reading: Sensor reading to add
        """
        self.sensor_buffer.append(reading)
        self.processing_stats['total_readings'] += 1
        
        # Process if interval elapsed
        current_time = time.time()
        if current_time - self.last_processing_time >= self.processing_interval:
            self._process_buffer()
            self.last_processing_time = current_time
    
    def add_building_state(self, state: BuildingState) -> None:
        """
        Add building state to processing buffer.
        
        Args:
            state: Building state to add
        """
        self.state_buffer.append(state)
    
    def _process_buffer(self) -> None:
        """Process buffered sensor readings."""
        if not self.sensor_buffer:
            return
        
        start_time = time.time()
        
        try:
            # Convert to DataFrame for processing
            readings_data = []
            for reading in self.sensor_buffer:
                readings_data.append({
                    'timestamp': reading.timestamp,
                    'sensor_type': reading.sensor_type,
                    'value': reading.value,
                    'quality': reading.quality
                })
            
            df = pd.DataFrame(readings_data)
            
            # Simple aggregation by sensor type
            aggregated = df.groupby('sensor_type').agg({
                'value': ['mean', 'std', 'count'],
                'quality': 'mean'
            }).round(3)
            
            # Update processing stats
            processing_time = time.time() - start_time
            self.processing_stats['processed_readings'] += len(self.sensor_buffer)
            self.processing_stats['avg_processing_time'] = (
                self.processing_stats['avg_processing_time'] * 0.9 + processing_time * 0.1
            )
            
            logger.debug(f"Processed {len(self.sensor_buffer)} readings in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing sensor buffer: {e}")
            self.processing_stats['errors'] += 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()
    
    def get_latest_state(self) -> Optional[BuildingState]:
        """Get latest building state."""
        return self.state_buffer[-1] if self.state_buffer else None
    
    def get_sensor_summary(self, sensor_type: str, window_minutes: int = 60) -> Optional[Dict[str, float]]:
        """
        Get sensor summary for specified time window.
        
        Args:
            sensor_type: Type of sensor to summarize
            window_minutes: Time window in minutes
            
        Returns:
            Dictionary with sensor summary statistics
        """
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # Filter readings by time and sensor type
        recent_readings = [
            r for r in self.sensor_buffer
            if r.timestamp >= cutoff_time and r.sensor_type == sensor_type
        ]
        
        if not recent_readings:
            return None
        
        values = [r.value for r in recent_readings]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'avg_quality': np.mean([r.quality for r in recent_readings])
        }


class BuildingDataPipeline:
    """
    Complete building data pipeline combining simulation, streaming, and processing.
    
    Orchestrates sensor simulation, MQTT streaming, and edge processing
    for end-to-end building energy management.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize building data pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.running = False
        
        # Initialize components
        self.simulator = SensorSimulator(
            building_id=config.get('building_id', 'building_001')
        )
        
        mqtt_config = config.get('mqtt', {})
        self.mqtt_streamer = MQTTDataStreamer(
            broker_host=mqtt_config.get('broker_host', 'localhost'),
            broker_port=mqtt_config.get('broker_port', 1883),
            topic_prefix=mqtt_config.get('topic_prefix', 'bems/sensors'),
            qos=mqtt_config.get('qos', 1)
        )
        
        self.edge_processor = EdgeDataProcessor(
            buffer_size=config.get('buffer_size', 100),
            processing_interval=config.get('processing_interval', 1.0)
        )
        
        logger.info("Initialized building data pipeline")
    
    def start(self) -> bool:
        """
        Start the data pipeline.
        
        Returns:
            True if started successfully
        """
        logger.info("Starting building data pipeline")
        
        # Connect to MQTT broker
        if not self.mqtt_streamer.connect():
            logger.error("Failed to connect to MQTT broker")
            return False
        
        self.running = True
        
        # Start simulation loop
        self._simulation_thread = threading.Thread(target=self._simulation_loop)
        self._simulation_thread.daemon = True
        self._simulation_thread.start()
        
        logger.info("Building data pipeline started")
        return True
    
    def stop(self) -> None:
        """Stop the data pipeline."""
        logger.info("Stopping building data pipeline")
        self.running = False
        
        if hasattr(self, '_simulation_thread'):
            self._simulation_thread.join(timeout=5)
        
        self.mqtt_streamer.disconnect()
        logger.info("Building data pipeline stopped")
    
    def _simulation_loop(self) -> None:
        """Main simulation loop."""
        time_step = timedelta(minutes=15)  # 15-minute intervals
        
        while self.running:
            try:
                # Generate next state
                state = self.simulator.generate_next_reading(time_step)
                
                # Add to edge processor
                self.edge_processor.add_building_state(state)
                
                # Publish to MQTT
                self.mqtt_streamer.publish_building_state(state)
                
                # Publish individual sensor readings
                sensors = [
                    SensorReading(state.timestamp, state.building_id, 'hvac', 
                                state.hvac_runtime, 'hours', 1.0),
                    SensorReading(state.timestamp, state.building_id, 'lighting', 
                                state.lighting_usage, 'hours', 1.0),
                    SensorReading(state.timestamp, state.building_id, 'occupancy', 
                                float(state.occupancy), 'count', 1.0),
                    SensorReading(state.timestamp, state.building_id, 'temperature', 
                                state.outside_temp, 'celsius', 1.0),
                    SensorReading(state.timestamp, state.building_id, 'energy', 
                                state.energy_consumption, 'kwh', 1.0)
                ]
                
                for sensor in sensors:
                    self.edge_processor.add_sensor_reading(sensor)
                    self.mqtt_streamer.publish_sensor_reading(sensor)
                
                # Sleep for simulation interval
                time.sleep(1.0)  # Real-time simulation
                
            except Exception as e:
                logger.error(f"Error in simulation loop: {e}")
                time.sleep(1.0)
    
    def get_current_state(self) -> Optional[BuildingState]:
        """Get current building state."""
        return self.edge_processor.get_latest_state()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.edge_processor.get_processing_stats()
    
    def get_sensor_summary(self, sensor_type: str, window_minutes: int = 60) -> Optional[Dict[str, float]]:
        """Get sensor summary."""
        return self.edge_processor.get_sensor_summary(sensor_type, window_minutes)

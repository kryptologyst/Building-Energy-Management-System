"""
Building Energy Management System - Streamlit Demo

Interactive demo simulating edge constraints with real-time energy monitoring,
model comparison, and edge deployment visualization.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.models.energy_predictor import EnergyPredictor, ModelConfig
from src.data.data_generator import BuildingDataGenerator, DataConfig
from src.pipelines.data_pipeline import BuildingDataPipeline, BuildingState
from src.utils.evaluator import ModelEvaluator
from src.utils.logger import setup_logging

# Page configuration
st.set_page_config(
    page_title="Building Energy Management System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .disclaimer {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin: 0.5rem 0;
    }
    .edge-constraint {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.5rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_running' not in st.session_state:
    st.session_state.pipeline_running = False
if 'current_state' not in st.session_state:
    st.session_state.current_state = None
if 'model_predictions' not in st.session_state:
    st.session_state.model_predictions = {}
if 'performance_data' not in st.session_state:
    st.session_state.performance_data = []


def load_models() -> Dict[str, EnergyPredictor]:
    """Load trained models."""
    models = {}
    
    model_files = {
        'baseline': 'models/baseline_energy_predictor.h5',
        'edge': 'models/edge_energy_predictor.h5',
        'compressed': 'models/compressed_energy_predictor.h5'
    }
    
    for name, path in model_files.items():
        if os.path.exists(path):
            try:
                model = EnergyPredictor(ModelConfig())
                model.load_model(path)
                models[name] = model
                st.success(f"Loaded {name} model")
            except Exception as e:
                st.error(f"Failed to load {name} model: {e}")
    
    return models


def simulate_edge_constraints() -> Dict[str, Any]:
    """Simulate edge device constraints."""
    return {
        'cpu_cores': 4,
        'memory_mb': 1024,
        'storage_gb': 8,
        'power_consumption_w': 5.0,
        'inference_latency_ms': 50,
        'model_size_kb': 256,
        'battery_life_hours': 8
    }


def create_energy_prediction_chart(predictions: Dict[str, float], 
                                 actual: float) -> go.Figure:
    """Create energy prediction comparison chart."""
    fig = go.Figure()
    
    # Add actual value
    fig.add_trace(go.Bar(
        name='Actual',
        x=['Energy Consumption'],
        y=[actual],
        marker_color='#2ecc71',
        text=[f'{actual:.1f} kWh'],
        textposition='auto'
    ))
    
    # Add predictions
    colors = ['#3498db', '#e74c3c', '#f39c12']
    for i, (model_name, pred) in enumerate(predictions.items()):
        fig.add_trace(go.Bar(
            name=f'{model_name.title()} Prediction',
            x=['Energy Consumption'],
            y=[pred],
            marker_color=colors[i % len(colors)],
            text=[f'{pred:.1f} kWh'],
            textposition='auto'
        ))
    
    fig.update_layout(
        title='Energy Consumption Predictions vs Actual',
        yaxis_title='Energy (kWh)',
        barmode='group',
        height=400
    )
    
    return fig


def create_performance_comparison(performance_data: List[Dict]) -> go.Figure:
    """Create performance comparison chart."""
    if not performance_data:
        return go.Figure()
    
    df = pd.DataFrame(performance_data)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Latency (ms)', 'Throughput (FPS)', 'Memory Usage (MB)', 'Model Size (KB)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Latency
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['latency_ms'], name='Latency', line=dict(color='#e74c3c')),
        row=1, col=1
    )
    
    # Throughput
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['throughput_fps'], name='Throughput', line=dict(color='#2ecc71')),
        row=1, col=2
    )
    
    # Memory usage
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['memory_mb'], name='Memory', line=dict(color='#f39c12')),
        row=2, col=1
    )
    
    # Model size
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['model_size_kb'], name='Model Size', line=dict(color='#9b59b6')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Edge Performance Metrics")
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    fig.update_yaxes(title_text="FPS", row=1, col=2)
    fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
    fig.update_yaxes(title_text="Size (KB)", row=2, col=2)
    
    return fig


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">🏢 Building Energy Management System</div>', 
                unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ DISCLAIMER:</strong> This system is for research and education purposes only. 
        NOT FOR SAFETY-CRITICAL USE. This demo simulates building energy management scenarios 
        and should not be used for actual building control systems.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Configuration")
    
    # Model selection
    st.sidebar.subheader("Models")
    selected_models = st.sidebar.multiselect(
        "Select models to compare:",
        ["baseline", "edge", "compressed"],
        default=["baseline", "edge"]
    )
    
    # Edge device simulation
    st.sidebar.subheader("Edge Device Simulation")
    device_type = st.sidebar.selectbox(
        "Device Type:",
        ["Raspberry Pi", "Jetson Nano", "Android Device", "Custom"]
    )
    
    if device_type == "Custom":
        cpu_cores = st.sidebar.slider("CPU Cores", 1, 8, 4)
        memory_mb = st.sidebar.slider("Memory (MB)", 512, 8192, 1024)
        power_w = st.sidebar.slider("Power (W)", 1, 20, 5)
    else:
        constraints = simulate_edge_constraints()
        cpu_cores = constraints['cpu_cores']
        memory_mb = constraints['memory_mb']
        power_w = constraints['power_consumption_w']
    
    # Simulation controls
    st.sidebar.subheader("Simulation Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 1, 10, 2)
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("No trained models found. Please run training first.")
        st.info("Run: `python src/scripts/train.py` to train models")
        return
    
    # Filter selected models
    selected_models = {name: model for name, model in models.items() 
                      if name in selected_models}
    
    if not selected_models:
        st.warning("Please select at least one model.")
        return
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["Real-time Monitoring", "Model Comparison", "Edge Performance", "Data Pipeline"])
    
    with tab1:
        st.header("Real-time Energy Monitoring")
        
        # Generate synthetic current state
        if st.button("Generate New Reading") or auto_refresh:
            # Create synthetic building state
            generator = BuildingDataGenerator(DataConfig())
            X_synthetic, _ = generator.generate_sensor_data()
            current_reading = X_synthetic[-1]  # Use last generated reading
            
            # Create building state
            current_state = BuildingState(
                timestamp=datetime.now(),
                building_id="demo_building",
                hvac_runtime=current_reading[0],
                lighting_usage=current_reading[1],
                occupancy=int(current_reading[2]),
                outside_temp=current_reading[3],
                energy_consumption=0  # Will be calculated
            )
            
            # Calculate actual energy consumption
            actual_energy = (current_reading[0] * 3.5 + current_reading[1] * 1.2 + 
                           current_reading[2] * 0.6 + abs(current_reading[3] - 22) * 0.8 + 5.0)
            current_state.energy_consumption = actual_energy
            
            # Get predictions from selected models
            predictions = {}
            for model_name, model in selected_models.items():
                pred = model.predict(current_reading.reshape(1, -1))[0]
                predictions[model_name] = pred
            
            st.session_state.current_state = current_state
            st.session_state.model_predictions = predictions
        
        # Display current state
        if st.session_state.current_state:
            state = st.session_state.current_state
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("HVAC Runtime", f"{state.hvac_runtime:.1f} hrs", "0.0 hrs")
            with col2:
                st.metric("Lighting Usage", f"{state.lighting_usage:.1f} hrs", "0.0 hrs")
            with col3:
                st.metric("Occupancy", f"{state.occupancy} people", "0 people")
            with col4:
                st.metric("Outside Temp", f"{state.outside_temp:.1f}°C", "0.0°C")
            
            # Energy consumption
            st.metric("Actual Energy Consumption", f"{state.energy_consumption:.1f} kWh", "0.0 kWh")
            
            # Predictions chart
            if st.session_state.model_predictions:
                fig = create_energy_prediction_chart(
                    st.session_state.model_predictions, 
                    state.energy_consumption
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Prediction accuracy
                st.subheader("Prediction Accuracy")
                accuracy_data = []
                for model_name, pred in st.session_state.model_predictions.items():
                    error = abs(pred - state.energy_consumption)
                    accuracy = max(0, 100 - (error / state.energy_consumption) * 100)
                    accuracy_data.append({
                        'Model': model_name.title(),
                        'Prediction': f"{pred:.1f} kWh",
                        'Error': f"{error:.1f} kWh",
                        'Accuracy': f"{accuracy:.1f}%"
                    })
                
                st.table(pd.DataFrame(accuracy_data))
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()
    
    with tab2:
        st.header("Model Comparison")
        
        if st.button("Run Model Comparison"):
            # Generate test data
            generator = BuildingDataGenerator(DataConfig(n_samples=100))
            X_test, y_test = generator.generate_sensor_data()
            
            # Evaluate models
            evaluator = ModelEvaluator()
            comparison_df = evaluator.compare_models(selected_models, X_test, y_test)
            
            # Display results
            st.subheader("Model Performance Comparison")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Create comparison charts
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(comparison_df, x='model_name', y='mae', 
                            title='Mean Absolute Error Comparison')
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(comparison_df, x='model_name', y='avg_latency_ms',
                            title='Average Latency Comparison')
                st.plotly_chart(fig2, use_container_width=True)
            
            # Accuracy vs Performance scatter
            fig3 = px.scatter(comparison_df, x='avg_latency_ms', y='mae',
                            size='model_size', hover_data=['model_name'],
                            title='Accuracy vs Performance Trade-off')
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.header("Edge Performance Analysis")
        
        # Edge constraints display
        st.subheader("Edge Device Constraints")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="edge-constraint">
                <strong>CPU:</strong> {cpu_cores} cores<br>
                <strong>Memory:</strong> {memory_mb} MB<br>
                <strong>Power:</strong> {power_w} W
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="edge-constraint">
                <strong>Target Latency:</strong> &lt; 100ms<br>
                <strong>Target Throughput:</strong> &gt; 10 FPS<br>
                <strong>Memory Limit:</strong> &lt; 512 MB
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="edge-constraint">
                <strong>Model Size:</strong> &lt; 1 MB<br>
                <strong>Battery Life:</strong> &gt; 8 hours<br>
                <strong>Offline Capable:</strong> Yes
            </div>
            """, unsafe_allow_html=True)
        
        # Simulate performance data
        if st.button("Simulate Edge Performance"):
            performance_data = []
            
            for i in range(20):  # Simulate 20 data points
                timestamp = datetime.now() - timedelta(minutes=20-i)
                
                # Simulate realistic performance metrics
                latency = np.random.normal(50, 10)  # ms
                throughput = np.random.normal(20, 5)  # FPS
                memory = np.random.normal(256, 50)  # MB
                model_size = np.random.normal(512, 100)  # KB
                
                performance_data.append({
                    'timestamp': timestamp,
                    'latency_ms': max(10, latency),
                    'throughput_fps': max(1, throughput),
                    'memory_mb': max(100, memory),
                    'model_size_kb': max(100, model_size)
                })
            
            st.session_state.performance_data = performance_data
        
        # Display performance charts
        if st.session_state.performance_data:
            fig = create_performance_comparison(st.session_state.performance_data)
            st.plotly_chart(fig, use_container_width=True)
            
            # Performance summary
            df_perf = pd.DataFrame(st.session_state.performance_data)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_latency = df_perf['latency_ms'].mean()
                st.metric("Avg Latency", f"{avg_latency:.1f} ms")
            
            with col2:
                avg_throughput = df_perf['throughput_fps'].mean()
                st.metric("Avg Throughput", f"{avg_throughput:.1f} FPS")
            
            with col3:
                avg_memory = df_perf['memory_mb'].mean()
                st.metric("Avg Memory", f"{avg_memory:.1f} MB")
            
            with col4:
                avg_size = df_perf['model_size_kb'].mean()
                st.metric("Avg Model Size", f"{avg_size:.1f} KB")
    
    with tab4:
        st.header("Data Pipeline Simulation")
        
        st.subheader("MQTT Data Streaming")
        
        # MQTT configuration
        col1, col2 = st.columns(2)
        
        with col1:
            broker_host = st.text_input("MQTT Broker Host", "localhost")
            broker_port = st.number_input("MQTT Broker Port", 1883, 65535, 1883)
        
        with col2:
            topic_prefix = st.text_input("Topic Prefix", "bems/sensors")
            qos_level = st.selectbox("QoS Level", [0, 1, 2], index=1)
        
        # Pipeline controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Pipeline"):
                st.session_state.pipeline_running = True
                st.success("Data pipeline started")
        
        with col2:
            if st.button("Stop Pipeline"):
                st.session_state.pipeline_running = False
                st.warning("Data pipeline stopped")
        
        with col3:
            if st.button("Generate Sample Data"):
                # Generate sample building state
                generator = BuildingDataGenerator(DataConfig())
                X_sample, y_sample = generator.generate_sensor_data()
                
                sample_state = BuildingState(
                    timestamp=datetime.now(),
                    building_id="sample_building",
                    hvac_runtime=X_sample[-1][0],
                    lighting_usage=X_sample[-1][1],
                    occupancy=int(X_sample[-1][2]),
                    outside_temp=X_sample[-1][3],
                    energy_consumption=y_sample[-1]
                )
                
                st.json({
                    "timestamp": sample_state.timestamp.isoformat(),
                    "building_id": sample_state.building_id,
                    "hvac_runtime": sample_state.hvac_runtime,
                    "lighting_usage": sample_state.lighting_usage,
                    "occupancy": sample_state.occupancy,
                    "outside_temp": sample_state.outside_temp,
                    "energy_consumption": sample_state.energy_consumption
                })
        
        # Pipeline status
        if st.session_state.pipeline_running:
            st.success("🟢 Pipeline Running")
            
            # Simulate real-time data
            if st.button("Simulate Real-time Data"):
                for i in range(5):
                    generator = BuildingDataGenerator(DataConfig())
                    X_sample, y_sample = generator.generate_sensor_data()
                    
                    sample_state = BuildingState(
                        timestamp=datetime.now() - timedelta(minutes=5-i),
                        building_id="demo_building",
                        hvac_runtime=X_sample[-1][0],
                        lighting_usage=X_sample[-1][1],
                        occupancy=int(X_sample[-1][2]),
                        outside_temp=X_sample[-1][3],
                        energy_consumption=y_sample[-1]
                    )
                    
                    st.json({
                        "timestamp": sample_state.timestamp.isoformat(),
                        "energy_consumption": sample_state.energy_consumption,
                        "hvac_runtime": sample_state.hvac_runtime
                    })
                    
                    time.sleep(0.5)
        else:
            st.error("🔴 Pipeline Stopped")
        
        # Data flow diagram
        st.subheader("Data Flow Architecture")
        
        # Create a simple data flow diagram
        st.markdown("""
        ```
        Building Sensors → MQTT Broker → Edge Processor → ML Models → Predictions
              ↓              ↓              ↓              ↓           ↓
        [HVAC, Lighting,   [Real-time    [Buffer,      [Baseline,   [Energy
         Occupancy, Temp]   Streaming]    Processing]   Edge,        Consumption
                                                       Compressed]   Predictions]
        ```
        """)


if __name__ == "__main__":
    main()

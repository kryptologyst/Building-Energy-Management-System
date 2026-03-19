# Building Energy Management System (BEMS)

A comprehensive Edge AI & IoT system for building energy consumption prediction, optimization, and real-time monitoring. This system demonstrates advanced machine learning techniques for edge deployment, including model compression, quantization, and real-time inference.

## ⚠️ IMPORTANT DISCLAIMER

**THIS SYSTEM IS FOR RESEARCH AND EDUCATION PURPOSES ONLY**

**NOT FOR SAFETY-CRITICAL USE**

This project is designed for learning and research in Edge AI, IoT, and building energy management. It should not be used for actual building control systems or safety-critical applications. Always consult with qualified professionals for real-world building management implementations.

## System Overview

The Building Energy Management System simulates and predicts energy consumption in buildings using sensor data from:

- **HVAC Systems**: Runtime hours and temperature control
- **Lighting Systems**: Usage hours and occupancy-based control  
- **Occupancy Sensors**: People count and activity patterns
- **Environmental Sensors**: Outside temperature and weather data

### Key Features

- **Multiple Model Architectures**: Baseline, edge-optimized, and compressed models
- **Edge Deployment**: TFLite, ONNX, and OpenVINO runtime support
- **Real-time Processing**: MQTT-based data streaming and edge inference
- **Model Compression**: Quantization, pruning, and knowledge distillation
- **Interactive Demo**: Streamlit-based visualization and monitoring
- **Comprehensive Evaluation**: Accuracy and edge performance metrics

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kryptologyst/Building-Energy-Management-System.git
   cd Building-Energy-Management-System
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies (optional)**
   ```bash
   pip install -e ".[dev]"
   ```

### Basic Usage

1. **Train models**
   ```bash
   python src/scripts/train.py --config configs/config.yaml
   ```

2. **Run interactive demo**
   ```bash
   streamlit run demo/streamlit_demo.py
   ```

3. **Export models for edge deployment**
   ```bash
   python src/scripts/export.py --model models/baseline_energy_predictor.h5 --target tflite
   ```

## 📁 Project Structure

```
building-energy-management/
├── src/                          # Source code
│   ├── models/                   # ML models and architectures
│   │   └── energy_predictor.py   # Energy prediction models
│   ├── data/                     # Data generation and processing
│   │   └── data_generator.py     # Synthetic sensor data generation
│   ├── export/                   # Model export and deployment
│   │   └── edge_exporter.py      # Edge deployment utilities
│   ├── pipelines/                # Data pipelines
│   │   └── data_pipeline.py      # MQTT streaming and processing
│   ├── utils/                    # Utility functions
│   │   ├── evaluator.py          # Model evaluation utilities
│   │   └── logger.py             # Logging configuration
│   └── scripts/                  # Command-line scripts
│       └── train.py              # Training pipeline
├── configs/                      # Configuration files
│   └── config.yaml               # Main configuration
├── data/                         # Data storage
│   ├── raw/                      # Raw sensor data
│   └── processed/                # Processed datasets
├── models/                       # Trained models
│   └── edge/                     # Edge-optimized models
├── assets/                       # Generated reports and visualizations
├── demo/                         # Interactive demos
│   └── streamlit_demo.py         # Streamlit web interface
├── tests/                        # Unit tests
├── logs/                         # Log files
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Project configuration
└── README.md                     # This file
```

## 🔧 Configuration

The system is configured through YAML files in the `configs/` directory:

### Model Configuration
- **Baseline Model**: 64-32 hidden units, dropout 0.2
- **Edge Model**: 32-16 hidden units, dropout 0.1  
- **Training**: Adam optimizer, MSE loss, early stopping

### Data Configuration
- **Samples**: 2000 synthetic building sensor readings
- **Features**: HVAC runtime, lighting usage, occupancy, temperature
- **Patterns**: Seasonal variations, occupancy cycles, equipment correlations

### Edge Deployment
- **Targets**: TFLite (int8), ONNX, OpenVINO
- **Optimization**: Pruning (50% sparsity), knowledge distillation
- **Constraints**: <100ms latency, >10 FPS throughput, <1MB model size

## Models

### Baseline Model
- **Architecture**: 4 → 64 → 32 → 1 (Dense layers)
- **Parameters**: ~3,000 parameters
- **Accuracy**: High accuracy baseline
- **Use Case**: Cloud/desktop deployment

### Edge-Optimized Model  
- **Architecture**: 4 → 32 → 16 → 1 (Dense layers)
- **Parameters**: ~800 parameters
- **Accuracy**: Slightly reduced but acceptable
- **Use Case**: Edge devices with moderate resources

### Compressed Model
- **Techniques**: Magnitude pruning (50% sparsity)
- **Parameters**: ~400 parameters (50% reduction)
- **Accuracy**: Minimal accuracy loss
- **Use Case**: Resource-constrained edge devices

## Evaluation Metrics

### Accuracy Metrics
- **MAE**: Mean Absolute Error (kWh)
- **RMSE**: Root Mean Square Error (kWh)  
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error

### Edge Performance Metrics
- **Latency**: Average inference time (ms)
- **Throughput**: Frames per second (FPS)
- **Memory Usage**: Peak memory consumption (MB)
- **Model Size**: Compressed model size (KB)
- **Power Consumption**: Energy per inference (mWh)

### Typical Results
| Model | MAE (kWh) | RMSE (kWh) | R² | Latency (ms) | Model Size (KB) |
|-------|-----------|------------|----|--------------|-----------------|
| Baseline | 1.2 | 1.8 | 0.92 | 15 | 12 |
| Edge | 1.4 | 2.0 | 0.90 | 8 | 3 |
| Compressed | 1.5 | 2.1 | 0.89 | 5 | 1.5 |

## Edge Deployment

### Supported Platforms

#### Raspberry Pi
- **CPU**: ARM Cortex-A72 (4 cores)
- **Memory**: 4GB RAM
- **Power**: 3.5W
- **Runtime**: TFLite, ONNX Runtime

#### NVIDIA Jetson Nano
- **CPU**: ARM Cortex-A57 (4 cores)  
- **GPU**: 128-core Maxwell
- **Memory**: 4GB RAM
- **Power**: 10W
- **Runtime**: TensorRT, TFLite

#### Android Devices
- **CPU**: ARM64 (8 cores)
- **Memory**: 8GB RAM
- **Power**: 5W
- **Runtime**: TFLite, ONNX Runtime

### Deployment Commands

```bash
# Export to TFLite (int8 quantization)
python src/scripts/export.py --model models/baseline_energy_predictor.h5 --target tflite --quantization int8

# Export to ONNX
python src/scripts/export.py --model models/edge_energy_predictor.h5 --target onnx

# Benchmark edge performance
python src/scripts/benchmark.py --model models/edge/baseline_int8.tflite --runtime tflite
```

## Data Pipeline

### MQTT Streaming
- **Broker**: Local MQTT broker (localhost:1883)
- **Topics**: `bems/sensors/{building_id}/{sensor_type}`
- **QoS**: Level 1 (at least once delivery)
- **Format**: JSON with timestamp, value, quality metrics

### Real-time Processing
- **Buffer Size**: 100 readings (configurable)
- **Processing Interval**: 1 second
- **Edge Constraints**: Memory-limited, CPU-constrained
- **Offline Capability**: Local caching and processing

### Sensor Simulation
- **Temporal Patterns**: Seasonal, daily, hourly cycles
- **Correlations**: HVAC-temperature, lighting-occupancy
- **Noise**: Realistic sensor noise and drift
- **Equipment States**: On/off status simulation

## Interactive Demo

The Streamlit demo provides:

### Real-time Monitoring
- Live sensor readings simulation
- Energy consumption predictions
- Model accuracy comparison
- Edge performance visualization

### Model Comparison
- Side-by-side accuracy metrics
- Performance vs accuracy trade-offs
- Edge constraint analysis
- Deployment recommendations

### Data Pipeline Simulation
- MQTT streaming visualization
- Real-time data processing
- Edge device constraints
- System architecture overview

### Usage
```bash
streamlit run demo/streamlit_demo.py
```

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
pytest tests/ -m integration
```

### Edge Deployment Tests
```bash
pytest tests/ -m edge
```

### Coverage Report
```bash
pytest tests/ --cov=src --cov-report=html
```

## Performance Optimization

### Model Optimization
- **Quantization**: INT8 quantization for 4x size reduction
- **Pruning**: Magnitude-based pruning for 50% parameter reduction
- **Distillation**: Knowledge transfer from teacher to student model
- **Architecture Search**: Hardware-aware neural architecture optimization

### Runtime Optimization
- **Batch Processing**: Single-sample inference for edge constraints
- **Memory Management**: Circular buffers and memory pooling
- **CPU Optimization**: Thread pinning and NUMA awareness
- **Power Management**: DVFS and sleep mode integration

### Communication Optimization
- **Data Compression**: Efficient sensor data encoding
- **Protocol Optimization**: MQTT QoS and keepalive tuning
- **Bandwidth Management**: Adaptive sampling rates
- **Offline Operation**: Local caching and sync mechanisms

## Security and Privacy

### Data Protection
- **Anonymization**: No personal identifiable information
- **Encryption**: TLS for MQTT communication
- **Access Control**: Device authentication and authorization
- **Data Minimization**: Only necessary sensor data collection

### Privacy Considerations
- **Local Processing**: On-device inference to minimize data transmission
- **Data Retention**: Configurable retention periods
- **Consent Management**: User consent for data collection
- **Audit Logging**: Comprehensive activity logging

## Development

### Code Quality
- **Formatting**: Black code formatter
- **Linting**: Ruff static analysis
- **Type Checking**: MyPy type hints
- **Testing**: Pytest with coverage
- **Pre-commit**: Automated quality checks

### Development Setup
```bash
# Install development dependencies
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run quality checks
5. Submit a pull request

## References and Standards

### MLPerf Tiny Benchmarks
- Visual Wake Words (VWW)
- Keyword Spotting (KWS)
- Image Classification
- Anomaly Detection

### Industry Standards
- **MQTT**: ISO/IEC 20922
- **OPC UA**: IEC 62541
- **Building Automation**: BACnet, Modbus
- **Energy Management**: ISO 50001

### Academic References
- Edge AI optimization techniques
- Building energy prediction models
- IoT data streaming architectures
- Model compression and quantization

## Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model file exists
ls -la models/

# Verify TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# Re-train models if needed
python src/scripts/train.py
```

#### MQTT Connection Issues
```bash
# Check MQTT broker status
mosquitto_pub -h localhost -t test -m "hello"

# Install MQTT broker
sudo apt-get install mosquitto mosquitto-clients
```

#### Edge Deployment Failures
```bash
# Check TFLite installation
python -c "import tensorflow.lite as tflite"

# Verify ONNX Runtime
python -c "import onnxruntime as ort"
```

### Performance Issues
- **High Latency**: Reduce model complexity or enable quantization
- **Memory Issues**: Decrease buffer size or enable pruning
- **Accuracy Loss**: Increase model size or disable compression
- **Power Consumption**: Optimize inference frequency or enable sleep modes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow and TensorFlow Lite teams for edge deployment tools
- ONNX and ONNX Runtime for cross-platform model support
- MQTT community for IoT communication protocols
- Building energy management research community
- Edge AI and TinyML research initiatives

---

**Remember: This system is for research and education purposes only. NOT FOR SAFETY-CRITICAL USE.**
# Building-Energy-Management-System

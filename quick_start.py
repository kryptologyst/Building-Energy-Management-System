#!/usr/bin/env python3
"""
Building Energy Management System - Quick Start Script

This script provides a quick way to get started with the BEMS system,
including training models, running evaluations, and launching the demo.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command to run as list
        description: Description of what the command does
        
    Returns:
        True if command succeeded
    """
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(f"❌ Command not found: {cmd[0]}")
        return False


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'numpy', 'pandas', 'tensorflow', 'scikit-learn', 
        'matplotlib', 'plotly', 'streamlit'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package} is not installed")
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All required dependencies are installed")
    return True


def train_models() -> bool:
    """Train all models."""
    logger.info("Training models...")
    
    cmd = [
        sys.executable, 
        "src/scripts/train.py",
        "--config", "configs/config.yaml"
    ]
    
    return run_command(cmd, "Model training")


def evaluate_models() -> bool:
    """Evaluate trained models."""
    logger.info("Evaluating models...")
    
    # Check if models exist
    model_files = [
        "models/baseline_energy_predictor.h5",
        "models/edge_energy_predictor.h5",
        "models/compressed_energy_predictor.h5"
    ]
    
    existing_models = [f for f in model_files if os.path.exists(f)]
    
    if not existing_models:
        logger.error("No trained models found. Run training first.")
        return False
    
    cmd = [
        sys.executable,
        "src/scripts/evaluate.py",
        "--models"
    ] + existing_models
    
    return run_command(cmd, "Model evaluation")


def export_models() -> bool:
    """Export models for edge deployment."""
    logger.info("Exporting models for edge deployment...")
    
    # Check if models exist
    model_files = [
        "models/baseline_energy_predictor.h5",
        "models/edge_energy_predictor.h5"
    ]
    
    existing_models = [f for f in model_files if os.path.exists(f)]
    
    if not existing_models:
        logger.error("No trained models found. Run training first.")
        return False
    
    success = True
    for model_path in existing_models:
        cmd = [
            sys.executable,
            "src/scripts/export.py",
            "--model", model_path,
            "--targets", "tflite", "onnx",
            "--quantization", "int8"
        ]
        
        if not run_command(cmd, f"Export {os.path.basename(model_path)}"):
            success = False
    
    return success


def run_demo() -> bool:
    """Run the Streamlit demo."""
    logger.info("Starting Streamlit demo...")
    
    cmd = [
        "streamlit", "run", "demo/streamlit_demo.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ]
    
    return run_command(cmd, "Streamlit demo")


def run_tests() -> bool:
    """Run unit tests."""
    logger.info("Running unit tests...")
    
    cmd = [
        sys.executable, "-m", "pytest", "tests/", "-v"
    ]
    
    return run_command(cmd, "Unit tests")


def main():
    """Main quick start script."""
    parser = argparse.ArgumentParser(description='Building Energy Management System - Quick Start')
    parser.add_argument('--action', type=str, default='all',
                      choices=['all', 'deps', 'train', 'evaluate', 'export', 'demo', 'test'],
                      help='Action to perform')
    parser.add_argument('--skip-deps', action='store_true',
                      help='Skip dependency checking')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger.info("🏢 Building Energy Management System - Quick Start")
    logger.info("DISCLAIMER: This system is for research and education purposes only")
    logger.info("NOT FOR SAFETY-CRITICAL USE")
    
    success = True
    
    # Check dependencies
    if not args.skip_deps and args.action in ['all', 'deps']:
        if not check_dependencies():
            logger.error("Dependency check failed. Please install missing packages.")
            return False
    
    # Train models
    if args.action in ['all', 'train']:
        if not train_models():
            success = False
    
    # Evaluate models
    if args.action in ['all', 'evaluate']:
        if not evaluate_models():
            success = False
    
    # Export models
    if args.action in ['all', 'export']:
        if not export_models():
            success = False
    
    # Run tests
    if args.action in ['all', 'test']:
        if not run_tests():
            success = False
    
    # Run demo
    if args.action in ['all', 'demo']:
        if not run_demo():
            success = False
    
    # Summary
    if success:
        logger.info("🎉 All operations completed successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. View evaluation results in assets/evaluation_report.txt")
        logger.info("2. Check exported models in models/edge/")
        logger.info("3. Run the demo: streamlit run demo/streamlit_demo.py")
        logger.info("4. Read the README.md for detailed documentation")
    else:
        logger.error("❌ Some operations failed. Check the logs above.")
        return False
    
    return True


if __name__ == '__main__':
    main()

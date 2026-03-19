"""
Building Energy Management System - Demo Script

Command-line script for running the interactive Streamlit demo.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import argparse
import logging
import os
import sys
import subprocess
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import plotly
        import pandas
        import numpy
        import matplotlib
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return False


def run_demo(port: int = 8501, host: str = "localhost") -> None:
    """
    Run the Streamlit demo.
    
    Args:
        port: Port number for the demo
        host: Host address for the demo
    """
    logger.info("Starting Building Energy Management System Demo")
    logger.info("DISCLAIMER: This system is for research and education purposes only")
    logger.info("NOT FOR SAFETY-CRITICAL USE")
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing required dependencies. Please install them first:")
        logger.error("pip install streamlit plotly pandas numpy matplotlib")
        return
    
    # Get demo script path
    demo_script = Path(__file__).parent.parent / "demo" / "streamlit_demo.py"
    
    if not demo_script.exists():
        logger.error(f"Demo script not found: {demo_script}")
        return
    
    # Run Streamlit
    cmd = [
        "streamlit", "run", str(demo_script),
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true"
    ]
    
    logger.info(f"Running demo on http://{host}:{port}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run demo: {e}")
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")


def main():
    """Main demo script."""
    parser = argparse.ArgumentParser(description='Run Building Energy Management System Demo')
    parser.add_argument('--port', type=int, default=8501,
                      help='Port number for the demo')
    parser.add_argument('--host', type=str, default='localhost',
                      help='Host address for the demo')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Run demo
    run_demo(port=args.port, host=args.host)


if __name__ == '__main__':
    main()

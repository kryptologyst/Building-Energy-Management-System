"""
Building Energy Management System - Logging Utilities

This module provides structured logging configuration for the BEMS system.

NOT FOR SAFETY-CRITICAL USE - Research and education purposes only.
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional
import yaml


def setup_logging(config_path: Optional[str] = None, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None) -> None:
    """
    Setup structured logging for the BEMS system.
    
    Args:
        config_path: Path to logging configuration file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default from config)
    """
    # Load logging config if provided
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            log_config = config.get('logging', {})
            log_level = log_config.get('level', log_level)
            log_file = log_file or log_config.get('file', 'logs/bems.log')
    
    # Create logs directory
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.info("Logging system initialized")
    logging.info("DISCLAIMER: This system is for research and education purposes only")
    logging.info("NOT FOR SAFETY-CRITICAL USE")


class BEMSLogger:
    """
    Custom logger for BEMS system with structured logging capabilities.
    """
    
    def __init__(self, name: str):
        """
        Initialize BEMS logger.
        
        Args:
            name: Logger name
        """
        self.logger = logging.getLogger(name)
    
    def log_training_start(self, model_name: str, config: dict) -> None:
        """Log training start with configuration."""
        self.logger.info(f"Starting training for {model_name}")
        self.logger.info(f"Configuration: {config}")
    
    def log_training_progress(self, epoch: int, loss: float, metrics: dict) -> None:
        """Log training progress."""
        self.logger.info(f"Epoch {epoch}: Loss={loss:.4f}, Metrics={metrics}")
    
    def log_evaluation_results(self, model_name: str, metrics: dict) -> None:
        """Log evaluation results."""
        self.logger.info(f"Evaluation results for {model_name}: {metrics}")
    
    def log_edge_deployment(self, model_name: str, target: str, metrics: dict) -> None:
        """Log edge deployment results."""
        self.logger.info(f"Edge deployment {model_name} to {target}: {metrics}")
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """Log error with context."""
        self.logger.error(f"Error in {context}: {str(error)}", exc_info=True)
    
    def log_warning(self, message: str, context: str = "") -> None:
        """Log warning with context."""
        self.logger.warning(f"Warning in {context}: {message}")
    
    def log_info(self, message: str, context: str = "") -> None:
        """Log info message with context."""
        self.logger.info(f"Info in {context}: {message}")
    
    def log_disclaimer(self) -> None:
        """Log safety disclaimer."""
        self.logger.warning("DISCLAIMER: This system is for research and education purposes only")
        self.logger.warning("NOT FOR SAFETY-CRITICAL USE")

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    project_name: str,
    log_dir: Path,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file_prefix: str = "ttsizer"
) -> logging.Logger:
    """
    Sets up a logger that outputs to both file and console.
    
    Args:
        project_name: Name of the project for the logger
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format string for log messages
        log_file_prefix: Prefix for log file names
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(project_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    formatter = logging.Formatter(log_format)
    
    # Create handlers
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{log_file_prefix}_{project_name}.log"
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    This is a convenience function to get loggers in other modules.
    
    Args:
        name: Name for the logger
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name) 
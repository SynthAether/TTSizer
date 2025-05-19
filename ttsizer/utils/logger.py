import logging
import sys
from pathlib import Path
import yaml
from typing import Dict, Any

def setup_root_logger(
    log_dir: Path,
    log_level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file_name: str = "ttsizer_app.log"
):
    """
    Sets up the root logger to output to both a single file and console.
    Clears existing handlers on the root logger before adding new ones.

    Args:
        log_dir: Directory to store the log file.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Format string for log messages.
        log_file_name: The name of the log file (e.g., 'app.log').
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    if root_logger.hasHandlers():
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            handler.close()
    
    formatter = logging.Formatter(log_format)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / log_file_name
    
    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

def initialize_logging(config_file_path_str: str = "configs/config.yaml"):
    """
    Initializes and configures logging for the application based on a config file.
    Reads logging settings from the YAML config file, with fallbacks to defaults.
    This function configures the root logger via setup_root_logger.
    Critical errors during setup are logged to specific error log files.
    """
    config_file_path = Path(config_file_path_str)

    # Default logging settings
    log_settings: Dict[str, Any] = {
        "log_level": "INFO",
        "log_dir": "logs",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_file_name": "ttsizer_app.log"  # Default log file name if no prefix from config
    }
    # This will be the final filename used, potentially updated by config or error fallbacks
    final_log_file_name = log_settings["log_file_name"]

    # Flag to indicate if an early error-specific logging setup was done
    error_logging_setup_done = False

    # Attempt to load and parse the configuration file
    if not config_file_path.exists():
        final_log_file_name = "ttsizer_init_error.log"
        # Setup basic logging to capture this critical error
        setup_root_logger(
            log_dir=Path(log_settings["log_dir"]),
            log_level="INFO", # Ensure critical errors are logged
            log_format=log_settings["log_format"],
            log_file_name=final_log_file_name
        )
        error_logging_setup_done = True
        logging.getLogger("LoggingInit").error(
            f"Configuration file not found: {config_file_path.resolve()}. Critical errors logged to: {final_log_file_name}"
        )

    config_data = None
    if not error_logging_setup_done: # Only proceed if config file existed
        try:
            with open(config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except Exception as e:
            final_log_file_name = "ttsizer_config_error.log"
            setup_root_logger(
                log_dir=Path(log_settings["log_dir"]),
                log_level="INFO",
                log_format=log_settings["log_format"],
                log_file_name=final_log_file_name
            )
            error_logging_setup_done = True
            logging.getLogger("LoggingInit").error(
                f"Error loading/parsing config file {config_file_path.resolve()}: {e}. Critical errors logged to: {final_log_file_name}", exc_info=True
            )

    # If config loaded successfully, try to use its logging_config section
    if config_data and "logging_config" in config_data and isinstance(config_data["logging_config"], dict):
        cfg_log_conf = config_data["logging_config"]
        log_settings["log_level"] = cfg_log_conf.get("log_level", log_settings["log_level"])
        log_settings["log_dir"] = cfg_log_conf.get("log_dir", log_settings["log_dir"])
        log_settings["log_format"] = cfg_log_conf.get("log_format", log_settings["log_format"])
        log_file_prefix = cfg_log_conf.get("log_file_prefix", "ttsizer") # Default prefix if not in config
        final_log_file_name = f"{log_file_prefix}_app.log"
    elif config_data and not ("logging_config" in config_data and isinstance(config_data["logging_config"], dict)):
        if not error_logging_setup_done:
            final_log_file_name = "ttsizer_logconfig_warning.log" # Specific name for this warning case
            # Setup with default settings but specific log file name for this warning
            setup_root_logger(
                log_dir=Path(log_settings["log_dir"]),
                log_level=log_settings["log_level"],
                log_format=log_settings["log_format"],
                log_file_name=final_log_file_name
            )
            error_logging_setup_done = True # Mark that logging setup has been done
            logging.getLogger("LoggingInit").warning(
                f"'logging_config' section missing or invalid in {config_file_path.resolve()}. Using default logging settings. Warnings/errors logged to: {final_log_file_name}"
            )
            
    setup_root_logger(
        log_dir=Path(log_settings["log_dir"]),
        log_level=log_settings["log_level"],
        log_format=log_settings["log_format"],
        log_file_name=final_log_file_name
    )
    # Log a final confirmation message to the (now definitively configured) logger.
    logging.getLogger("LoggingInit").info(f"Logging initialized. Log messages will be directed to: {Path(log_settings['log_dir']) / final_log_file_name}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    This logger will inherit handlers from the root logger.
    
    Args:
        name: Name for the logger.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name) 
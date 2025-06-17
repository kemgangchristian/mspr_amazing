#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 1.0        ############
#######################################################

import sys
import logging
from pathlib import Path
from datetime import datetime

_log_initialized = False


def setup_logger(log_dir: Path, log_level: str = "INFO"):
    """Initialise la configuration du logger global avec timestamp"""
    global _log_initialized
    if _log_initialized:
        return

    log_dir.mkdir(parents=True, exist_ok=True)

    # Format du timestamp : YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"mspr_data_cleaning_{timestamp}.log"

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler avec timestamp dans le nom
    file_handler = logging.FileHandler(log_dir / log_file_name)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _log_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger nommé (à utiliser dans chaque module)"""
    return logging.getLogger(name)

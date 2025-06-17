#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 1.0        ############
#######################################################

import os
from pathlib import Path


class Config:
    """Configuration globale de l'application"""

    # Configuration Spark
    APP_NAME = "MSPR AMAZING"
    SPARK_DIR = "/tmp/spark-tmp"

    # Chemins des données
    BASE_DIR = Path(__file__).parent.parent.parent  # Racine du projet
    DATA_DIR = BASE_DIR / "data"
    INPUT_DIR = DATA_DIR / "raw"
    INPUT_CSV = INPUT_DIR / "csv"
    INPUT_PARQUET = INPUT_DIR / "parquet"
    TMP_DIR = DATA_DIR / "tmp"
    OUTPUT_DIR = DATA_DIR / "processed"
    OUTPUT_S3 = "data/processed"
    STORAGE_MODEL = BASE_DIR / "models"
    STORAGE_MODEL_S3 = "models"
    STORAGE_IMG_S3 = "images"
    LOGS_DIR = BASE_DIR / "logs"

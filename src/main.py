#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 1.0        ############
#######################################################

import os
import argparse
import logging
from pathlib import Path
from minio import Minio
from pyspark.sql import SparkSession
from src.conf.settings import Config
from src.processing.converter import CsvToParquetConverter
from src.processing.data_cleaning import DataCleaning
from src.utils.logger import setup_logger, get_logger
from src.training.train import CustomerSegmentation
from src.utils.s3_config import configure_s3_access


def create_spark_session(config: Config, temp_dir: str) -> SparkSession:
    """Crée une session Spark optimisée avec les configurations essentielles"""
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    return (
        SparkSession.builder.appName(config.APP_NAME)
        .config("spark.local.dir", temp_dir)
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.6,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
        )
        .getOrCreate()
    )


def generate_log(config: Config) -> logging.Logger:
    setup_logger(Path(str(config.LOGS_DIR)))
    return get_logger(__name__)


def init_minio_client() -> Minio:
    """Initialise et retourne le client MinIO"""
    return Minio(
        endpoint=f"{os.getenv('MINIO_HOSTNAME', 'minio')}:{os.getenv('MINIO_PORT', '9000')}",
        access_key=os.getenv("MINIO_ACCESS_KEY", "minio"),
        secret_key=os.getenv("MINIO_SECRET_KEY", "minio123"),
        secure=False,
    )


def parse_arguments(config: Config) -> argparse.Namespace:
    """Gestion des arguments CLI"""
    parser = argparse.ArgumentParser(description="Pipeline complet du MSPR - AMAZING")

    parser.add_argument(
        "--input_csv", default=str(config.INPUT_CSV), help="Répertoire des fichiers CSV"
    )
    parser.add_argument(
        "--input_parquet",
        default=str(config.INPUT_PARQUET),
        help="Répertoire des fichiers Parquet",
    )
    parser.add_argument(
        "--input_s3_model",
        default=config.OUTPUT_S3,
        help="Préfixe des fichiers d'entraînement (dans MinIO)",
    )
    parser.add_argument(
        "--output_data",
        default=str(config.OUTPUT_DIR),
        help="Répertoire de sortie local",
    )
    parser.add_argument(
        "--output_s3", default=config.OUTPUT_S3, help="Répertoire de sortie MinIO"
    )
    parser.add_argument(
        "--output_s3_model",
        default=config.STORAGE_MODEL_S3,
        help="Répertoire S3 pour stocker le modèle",
    )
    parser.add_argument(
        "--output_s3_img",
        default=config.STORAGE_IMG_S3,
        help="Répertoire S3 pour stocker le modèle",
    )

    return parser.parse_args()


def convert_data_pipeline(converter, logger):
    """Exécute le pipeline de conversion CSV → PARQUET"""
    logger.info("Conversion du CSV -> PARQUET")
    converter.run_pipeline_convert()
    logger.info("Conversion terminée")


def cleaning_data_pipeline(cleaner, logger):
    """Exécute le pipeline de traitement de données"""
    logger.info("Début du traitement de données")
    cleaner.run_pipeline_cleaning()
    logger.info("Traitement de données terminée")


def train_model_pipeline(segmentation, logger):
    """Exécute le pipeline d'entraînement du modèle"""
    logger.info("Début de l'entraînement du modèle")
    segmentation.run_pipeline_model()
    logger.info("Entraînement du modèle terminé")


def main():
    config = Config()
    logger = generate_log(config)
    args = parse_arguments(config)
    input_csv = args.input_csv
    input_parquet = args.input_parquet
    output_s3 = args.output_s3
    s3_bucket = os.getenv("MINIO_BUCKET_NAME", "mspr")
    s3_prefix_in = args.input_s3_model
    s3_prefix_out = args.output_s3_model
    s3_prefix_img = args.output_s3_img
    # tmp = config.SPARK_DIR
    temp_dir = os.getenv("SPARK_LOCAL_DIR", config.SPARK_DIR)

    spark = None

    try:
        logger.info("🔧 Initialisation")
        spark = create_spark_session(config, temp_dir)
        s3_client = init_minio_client()
        # Configuration S3A centralisée
        configure_s3_access(spark, logger)

        # Initialisation des contructeurs
        converter = CsvToParquetConverter(spark, input_csv, input_parquet)
        cleaner = DataCleaning(
            spark,
            input_parquet,
            output_s3,
            s3_client,
            s3_bucket,
            s3_prefix_in,
            s3_prefix_out,
        )
        segmentation = CustomerSegmentation(
            spark,
            temp_dir,
            s3_client,
            s3_bucket,
            s3_prefix_in,
            s3_prefix_out,
            s3_prefix_img,
        )

        logger.info("Démarrage du pipeline complet")
        # Exécution du pipeline de conversion
        convert_data_pipeline(converter, logger)

        # Exécution du pipeline de traitement des données
        cleaning_data_pipeline(cleaner, logger)

        # Exécution du pipeline d'entraînement
        train_model_pipeline(segmentation, logger)

    except Exception as e:
        logger.error(f"💥 Erreur critique: {e}", exc_info=True)
        raise

    finally:
        if spark:
            spark.stop()
            logger.info("🛑 Session Spark arrêtée")


if __name__ == "__main__":
    main()

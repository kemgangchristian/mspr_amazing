#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 1.0        ############
#######################################################

import os
from pyspark.sql import SparkSession
from typing import Optional
import logging


def configure_s3_access(spark: SparkSession, logger: Optional[logging.Logger] = None):
    """Configuration optimisée pour MinIO via S3A avec gestion des logs"""

    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()

    # Configuration principale
    base_config = {
        # Authentification
        "fs.s3a.access.key": os.getenv("MINIO_ACCESS_KEY", "minio"),
        "fs.s3a.secret.key": os.getenv("MINIO_SECRET_KEY", "minio123"),
        "fs.s3a.endpoint": os.getenv("MINIO_ENDPOINT", "http://minio:9000"),
        # Configuration requise
        "fs.s3a.path.style.access": "true",
        "fs.s3a.connection.ssl.enabled": "false",
        "fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
        # Optimisation des performances
        "fs.s3a.fast.upload": "true",
        "fs.s3a.fast.upload.buffer": "disk",
        "fs.s3a.fast.upload.active.blocks": "4",
        # Configuration des threads et timeouts supplémentaires
        "fs.s3a.threads.max": "10",
        "fs.s3a.threads.keepalivetime": "60",  # en secondes
        # Gestion des erreurs
        "fs.s3a.attempts.maximum": "5",
        "fs.s3a.retry.limit": "3",
        "fs.s3a.retry.interval": "5000",
        # Timeouts (en millisecondes)
        "fs.s3a.connection.timeout": "60000",  # 60 secondes
        "fs.s3a.connection.establish.timeout": "5000",  # 5 secondes
        "fs.s3a.connection.tcp.timeout": "60000",  # 60 secondes
        # Désactivation des métadonnées
        "fs.s3a.metadatastore.impl": "org.apache.hadoop.fs.s3a.s3guard.NullMetadataStore",
        "fs.s3a.multipart.purge.age": "86400000",  # 24h en ms
    }

    # Application des configurations
    for key, value in base_config.items():
        hadoop_conf.set(key, value)

    # Logging des configurations
    if logger:
        logger.debug("🔧 Configuration S3A appliquée:")
        for key in sorted(base_config.keys()):
            logger.debug(f"{key}: {hadoop_conf.get(key)}")

    # Validation supplémentaire
    required_configs = ["fs.s3a.access.key", "fs.s3a.secret.key", "fs.s3a.endpoint"]

    for config in required_configs:
        if not hadoop_conf.get(config):
            error_msg = f"Configuration manquante: {config}"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)

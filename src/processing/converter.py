#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 1.0        ############
#######################################################

import shutil
from pathlib import Path
from pyspark.sql import SparkSession
from src.utils.logger import get_logger
from pyspark.sql.types import (
    StructType,
    StructField,
    TimestampType,
    StringType,
    LongType,
    DoubleType,
)

logger = get_logger(__name__)


class CsvToParquetConverter:
    def __init__(
        self,
        spark: SparkSession,
        input_csv: str,
        input_parquet: str,
    ):
        self.spark = spark
        self.input_csv = input_csv
        self.input_parquet = input_parquet

    def convert_file(self, csv_path: Path, output_path: Path):
        """Convertit un seul fichier CSV en Parquet (coalesce en un seul fichier)."""
        if not csv_path.exists() or csv_path.suffix != ".csv":
            logger.warning(f"Fichier non trouvé ou invalide : {csv_path}")
            return

        parquet_filename = csv_path.stem + ".parquet"
        output_file = output_path / parquet_filename
        temp_output_dir = output_path / f"_tmp_{csv_path.stem}"

        if output_file.exists():
            logger.info(f"{parquet_filename} déjà converti. Ignoré.")
            return

        # Définition du schéma explicite
        schema = StructType(
            [
                StructField("event_time", TimestampType(), True),
                StructField("event_type", StringType(), True),
                StructField("product_id", LongType(), True),
                StructField("category_id", LongType(), True),
                StructField("category_code", StringType(), True),
                StructField("brand", StringType(), True),
                StructField("price", DoubleType(), True),
                StructField("user_id", LongType(), True),
                StructField("user_session", StringType(), True),
            ]
        )

        try:
            logger.info(f"Conversion du fichier : {csv_path.name}")
            df = self.spark.read.csv(
                str(csv_path),
                header=True,
                schema=schema,
                timestampFormat="yyyy-MM-dd HH:mm:ss 'UTC'",
            )
            df.coalesce(1).write.mode("overwrite").parquet(str(temp_output_dir))

            # Trouver le fichier part-xxxx.parquet et le déplacer à la racine en renommant
            for part_file in temp_output_dir.iterdir():
                if (
                    part_file.name.startswith("part-")
                    and part_file.suffix == ".parquet"
                ):
                    shutil.move(str(part_file), str(output_file))
                    break

            shutil.rmtree(temp_output_dir)
            logger.info(f"Conversion réussie : {csv_path.name} → {parquet_filename}")

        except Exception as e:
            logger.error(
                f"Erreur lors de la conversion : {csv_path.name} | {e}",
                exc_info=True,
            )

    def convert_directory(self):
        """Convertit tous les fichiers CSV dans un dossier, déplace les fichiers Parquet existants."""
        input_path = Path(self.input_csv)
        output_path = Path(self.input_parquet)
        output_path.mkdir(parents=True, exist_ok=True)

        if not input_path.exists():
            logger.warning(f"Le dossier d'entrée n'existe pas : {input_path}")
            return

        for file in input_path.iterdir():
            if file.suffix == ".csv":
                self.convert_file(file, output_path)

            elif file.suffix == ".parquet":
                try:
                    destination = output_path / file.name
                    if not destination.exists():
                        shutil.move(str(file), str(destination))
                        logger.info(f"Fichier Parquet déplacé : {file.name}")
                    else:
                        logger.warning(f"Fichier Parquet déjà présent : {file.name}")
                except Exception as e:
                    logger.error(
                        f"Erreur lors du déplacement : {file.name} | {e}",
                        exc_info=True,
                    )

    def run_pipeline_convert(self):
        """Exécute le pipeline de conversion des fichiers CSV -> PARQUET."""
        logger.info("Démarrage du pipeline de conversion des fichiers csv en parquet")

        try:
            # 1. Convertir les fichiers csv en parquet
            self.convert_directory()

            logger.info(f"Pipeline de conversion terminé avec succès.")

        except Exception as e:
            logger.error("Échec du pipeline de conversion", exc_info=True)
            raise

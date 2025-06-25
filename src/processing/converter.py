#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 1.0        ############
#######################################################

import shutil
import requests
import gzip
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
    # Configuration des paths
    BASE_DATA_URL = "https://data.rees46.com/datasets/marketplace/"
    # Liste des données
    DATASET_FILES = [
        "2019-Oct.csv.gz",
        "2019-Nov.csv.gz",
        "2019-Dec.csv.gz",
        "2020-Jan.csv.gz",
        "2020-Feb.csv.gz",
        "2020-Mar.csv.gz",
        "2020-Apr.csv.gz",
    ]

    def __init__(
        self,
        spark: SparkSession,
        raw_tmp: str,
        input_csv: str,
        input_parquet: str,
    ):
        self.spark = spark
        self.raw_tmp = Path(raw_tmp)
        self.input_csv = Path(input_csv)
        self.input_parquet = Path(input_parquet)

        # Création des répertoires si inexistants
        self.raw_tmp.mkdir(parents=True, exist_ok=True)
        self.input_csv.mkdir(parents=True, exist_ok=True)
        self.input_parquet.mkdir(parents=True, exist_ok=True)

    def _get_full_url(self, filename: str) -> str:
        return f"{self.BASE_DATA_URL}{filename}"

    def download_all_datasets(self) -> None:
        """Télécharge tous les jeux de données dans le répertoire data/raw/tmp."""
        logger.info("Début du téléchargement des datasets dans tmp")

        for filename in self.DATASET_FILES:
            try:
                self.download_and_extract_gz(filename)
            except Exception as e:
                logger.error(f"Échec pour {filename}: {str(e)}")
                continue

        logger.info("Téléchargement terminé.")

    def download_and_extract_gz(self, filename: str) -> Path:
        """Télécharge un fichier .gz dans data/raw/tmp."""
        gz_path = self.raw_tmp / filename
        # Déterminez le nom du fichier CSV correspondant
        csv_filename = filename.replace(".gz", "")
        csv_path = self.input_csv / csv_filename

        # Vérifiez si le fichier CSV existe déjà dans input_csv
        if csv_path.exists():
            logger.info(f"Le fichier {csv_filename} existe déjà dans csv. Ignoré.")
            return csv_path  # Retourne le chemin du fichier CSV existant

        url = self._get_full_url(filename)
        logger.info(f"Début du téléchargement: {url}")

        try:
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(gz_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            logger.info(f"Téléchargement réussi: {filename}")
            return gz_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de téléchargement: {url} | {e}")
            if gz_path.exists():
                gz_path.unlink()  # Supprime le fichier .gz en cas d'erreur
            raise

    def decompress_files(self) -> None:
        """Décompresse tous les fichiers .gz dans data/raw/csv et nettoie data/raw/tmp."""
        logger.info("Début de la décompression des fichiers dans raw/csv")

        for gz_file in self.raw_tmp.glob("*.gz"):
            try:
                csv_path = self.input_csv / gz_file.name.replace(".gz", "")
                logger.info(f"Décompression de {gz_file.name}...")
                with gzip.open(gz_file, "rb") as gz_file_obj:
                    with open(csv_path, "wb") as csv_file:
                        shutil.copyfileobj(gz_file_obj, csv_file)

                gz_file.unlink()  # Supprime le fichier .gz après décompression
                logger.info(f"Décompression réussie: {gz_file.name} → {csv_path.name}")

            except Exception as e:
                logger.error(f"Erreur lors de la décompression: {gz_file.name} | {e}")

    def convert_directory(self) -> None:
        """Convertit tous les fichiers CSV dans data/raw/csv en Parquet dans data/raw/parquet."""
        logger.info("Début de la conversion des fichiers CSV en Parquet")

        for csv_file in self.input_csv.glob("*.csv"):
            self.convert_file(csv_file, self.input_parquet)

    def convert_file(self, csv_path: Path, output_path: Path):
        """Convertit un seul fichier CSV en Parquet."""
        if not csv_path.exists() or csv_path.suffix != ".csv":
            logger.warning(f"Fichier non trouvé ou invalide : {csv_path}")
            return

        parquet_filename = csv_path.stem + ".parquet"
        output_file = output_path / parquet_filename

        if output_file.exists():
            logger.info(f"{parquet_filename} déjà converti. Ignoré.")
            return

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
            df.coalesce(1).write.mode("overwrite").parquet(str(output_file))
            logger.info(f"Conversion réussie : {csv_path.name} → {parquet_filename}")

        except Exception as e:
            logger.error(
                f"Erreur lors de la conversion : {csv_path.name} | {e}", exc_info=True
            )

    def run_pipeline_convert(self):
        """Exécute le pipeline de conversion des fichiers CSV -> PARQUET."""
        logger.info("Démarrage du pipeline de conversion des fichiers csv en parquet")

        try:
            # 1. Télécharge tous les jeux de données
            self.download_all_datasets()

            # 2. Décompresse les fichiers téléchargés
            self.decompress_files()

            # 3. Convertit les fichiers csv en parquet
            self.convert_directory()

            logger.info("Pipeline de conversion terminé avec succès.")

        except Exception as e:
            logger.error("Échec du pipeline de conversion", exc_info=True)
            raise

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
        input_csv: str,
        input_parquet: str,
    ):
        self.spark = spark
        self.input_csv = Path(input_csv)
        self.input_parquet = Path(input_parquet)

        # Création des répertoires si inexistants
        self.input_csv.mkdir(parents=True, exist_ok=True)
        self.input_parquet.mkdir(parents=True, exist_ok=True)

    def _get_full_url(self, filename: str) -> str:
        """Construit l'URL complète à partir du nom de fichier."""
        return f"{self.BASE_DATA_URL}{filename}"

    def download_and_extract_gz(self, filename: str) -> Path:
        """
        Télécharge et décompresse un fichier .gz.

        Args:
            filename: Nom du fichier à télécharger (avec extension .gz)

        Returns:
            Path: Chemin du fichier CSV extrait
        """
        csv_filename = filename.replace(".gz", "")
        csv_path = self.input_csv / csv_filename
        gz_path = self.input_csv / filename

        # Vérification si le fichier CSV existe déjà
        if csv_path.exists():
            logger.info(f"Le fichier {csv_filename} existe déjà. Ignoré.")
            return csv_path

        url = self._get_full_url(filename)
        logger.info(f"Début du téléchargement: {url}")

        try:
            # Téléchargement en stream
            with requests.get(url, stream=True) as response:
                response.raise_for_status()
                with open(gz_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

            # Décompression
            logger.info(f"Décompression de {filename}...")
            with gzip.open(gz_path, "rb") as gz_file:
                with open(csv_path, "wb") as csv_file:
                    shutil.copyfileobj(gz_file, csv_file)

            # Nettoyage
            gz_path.unlink()
            logger.info(f"Fichier prêt: {csv_path}")
            return csv_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur de téléchargement: {url} | {e}")
            if gz_path.exists():
                gz_path.unlink()
            raise
        except Exception as e:
            logger.error(f"Erreur inattendue: {url} | {e}")
            if gz_path.exists():
                gz_path.unlink()
            if csv_path.exists():
                csv_path.unlink()
            raise

    def download_all_datasets(self) -> None:
        """Télécharge tous les jeux de données configurés."""
        logger.info("Début du téléchargement des datasets")

        downloaded_files = []
        for filename in self.DATASET_FILES:
            try:
                csv_file = self.download_and_extract_gz(filename)
                downloaded_files.append(csv_file)
            except Exception as e:
                logger.error(f"Échec pour {filename}: {str(e)}")
                continue

        logger.info(
            f"Téléchargement terminé: {len(downloaded_files)}/{len(self.DATASET_FILES)} fichiers"
        )

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
            # 1. Télécharge tous les jeux de données via url
            self.download_all_datasets()

            # 2. Convertir les fichiers csv en parquet
            # self.convert_directory()

            logger.info(f"Pipeline de conversion terminé avec succès.")

        except Exception as e:
            logger.error("Échec du pipeline de conversion", exc_info=True)
            raise

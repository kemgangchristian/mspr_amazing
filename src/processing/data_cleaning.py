#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 1.0        ############
#######################################################

import os
import uuid
import shutil
from pathlib import Path
from minio import Minio
from minio.error import S3Error
from typing import List, Dict, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    lower,
    concat,
    lit,
    when,
    trim,
    size,
    split,
    sha2,
    concat,
)
from pyspark.sql.types import (
    TimestampType,
    StringType,
    LongType,
    DoubleType,
    IntegerType,
)
from prettytable import PrettyTable
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaning:
    def __init__(
        self,
        spark: SparkSession,
        input_parquet: str,
        output_s3: str,
        s3_client: Minio,
        s3_bucket: str,
        s3_prefix_in: str,
        s3_prefix_out: str,
    ):
        self.spark = spark
        self.input_parquet = input_parquet
        self.output_s3 = output_s3
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_prefix_in = s3_prefix_in
        self.s3_prefix_out = s3_prefix_out
        self.valid_event_types = ["view", "cart", "purchase", "remove_from_cart"]
        self.expected_columns = [
            "event_time",
            "event_type",
            "product_id",
            "category_id",
            "category_code",
            "brand",
            "price",
            "user_id",
            "user_session",
        ]

    def validate_schema(self, df: DataFrame) -> bool:
        """Vérifie que le schéma du DataFrame correspond aux attentes."""
        schema_fields = {field.name: field.dataType for field in df.schema.fields}
        expected_types = {
            "event_time": TimestampType,
            "event_type": StringType,
            "product_id": LongType,
            "category_id": LongType,
            "category_code": StringType,
            "brand": StringType,
            "price": DoubleType,
            "user_id": LongType,
            "user_session": StringType,
        }

        for col_name, expected_type in expected_types.items():
            if col_name not in schema_fields:
                logger.error(f"Colonne manquante: {col_name}")
                return False
            actual_type = type(schema_fields[col_name])
            if actual_type != expected_type:
                logger.error(
                    f"Type incorrect pour {col_name}: attendu {expected_type}, obtenu {schema_fields[col_name]}"
                )
                return False
        return True

    def detect_valid_files_in_directory(self, directory_path: str) -> List[str]:
        """Détecte les fichiers Parquet valides dans un répertoire."""
        if not os.path.exists(directory_path):
            logger.error(f"Dossier inexistant : {directory_path}")
            raise FileNotFoundError(f"Dossier introuvable : {directory_path}")

        valid_files = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.endswith(".parquet")
            and os.path.getsize(os.path.join(directory_path, f)) > 0
        ]

        if not valid_files:
            logger.error(f"Aucun fichier Parquet valide trouvé dans : {directory_path}")
            raise ValueError("Aucun fichier .parquet valide trouvé.")

        logger.info(f"{len(valid_files)} fichiers Parquet valides détectés")
        return valid_files

    def load_data(self, input_path: str) -> DataFrame:
        """Charge les données depuis un fichier Parquet avec validation de schéma."""
        logger.info(f"Chargement des données depuis : {input_path}")
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Fichier introuvable : {input_path}")

        try:
            df = self.spark.read.parquet(input_path)

            if not self.validate_schema(df):
                raise ValueError("Schéma de données invalide")

            logger.info("Données chargées avec succès")
            return df
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données : {e}", exc_info=True)
            raise

    def save_cleaned_data(self, df: DataFrame, output_dir: str, file_name: str):
        """Sauvegarde les données nettoyées dans un seul fichier Parquet, si le fichier n'existe pas déjà."""
        os.makedirs(output_dir, exist_ok=True)
        tmp_dir = os.path.join(output_dir, f"_tmp_{uuid.uuid4().hex}")
        final_path = os.path.join(output_dir, file_name)

        # 🚫 Vérifie si le fichier final existe déjà
        if os.path.exists(final_path):
            logger.warning(
                f"Le fichier '{final_path}' existe déjà. Sauvegarde ignorée."
            )
            return None

        try:
            # 📝 Écriture dans un répertoire temporaire
            df.coalesce(1).write.mode("overwrite").parquet(str(tmp_dir))

            # 🔍 Recherche du fichier généré
            part_file = next(
                (
                    f
                    for f in os.listdir(tmp_dir)
                    if f.startswith("part-") and f.endswith(".parquet")
                ),
                None,
            )

            if not part_file:
                raise RuntimeError(
                    "Aucun fichier .parquet trouvé dans le répertoire temporaire."
                )

            # 🚚 Déplacement vers le chemin final
            shutil.move(os.path.join(tmp_dir, part_file), final_path)

            logger.info(f"Données sauvegardées : {final_path}")
            return final_path

        except Exception as e:
            logger.error(f"Échec de la sauvegarde : {e}", exc_info=True)
            raise

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def file_exists_in_minio(
        self, s3_client: Minio, bucket_name: str, s3_object_key: str
    ) -> bool:
        """
        Vérifie si un fichier (s3_object_key) existe dans un bucket MinIO existant.

        Si le bucket n'existe pas, la fonction le loggue et retourne False sans lever d'exception.

        Args:
            s3_client: client MinIO.
            bucket_name: nom du bucket MinIO.
            s3_object_key: chemin/nom du fichier à vérifier dans le bucket.

        Returns:
            True si le fichier existe, False sinon (y compris si le bucket n'existe pas).
        """
        logger.info(f"Vérification de l'existence du bucket '{bucket_name}'...")
        try:
            if not s3_client.bucket_exists(bucket_name):
                logger.warning(
                    f"Le bucket '{bucket_name}' n'existe pas. Poursuite du traitement sans vérification du fichier."
                )
                return False
        except S3Error as e:
            logger.error(
                f"Erreur lors de la vérification du bucket '{bucket_name}' : {e}"
            )
            return False  # continue quand même

        logger.info(
            f"Vérification de la présence du fichier '{s3_object_key}' dans le bucket '{bucket_name}'..."
        )
        try:
            s3_client.stat_object(bucket_name, s3_object_key)
            logger.info(
                f"Le fichier '{s3_object_key}' existe dans le bucket '{bucket_name}'."
            )
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.info(
                    f"Le fichier '{s3_object_key}' n'existe pas dans le bucket '{bucket_name}'."
                )
                return False
            else:
                logger.error(
                    f"Erreur lors de la vérification du fichier '{s3_object_key}' : {e}"
                )
                raise

    def save_cleaned_data_minio(
        self,
        df: DataFrame,
        bucket_name: str,
        output_dir: str,
        file_name: str,
        s3_client: Minio,
    ):
        """
        Sauvegarde un DataFrame Spark en local (Parquet), puis l'upload dans un bucket MinIO.
        Le répertoire utilisé doit être partagé entre tous les conteneurs Spark via un volume Docker.
        """

        os.makedirs(output_dir, exist_ok=True)
        tmp_dir = os.path.join(output_dir, f"_tmp_{uuid.uuid4().hex}")
        final_path = os.path.join(output_dir, file_name)

        # 🔍 Vérifie l'existence du bucket
        try:
            if not s3_client.bucket_exists(bucket_name):
                s3_client.make_bucket(bucket_name)
                logger.info(f"Bucket '{bucket_name}' créé.")
            else:
                logger.info(f"Bucket '{bucket_name}' déjà existant.")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification/création du bucket : {e}")
            raise

        # 🚫 Vérifie si le fichier existe déjà dans le bucket
        try:
            s3_client.stat_object(bucket_name, final_path)
            logger.warning(
                f"Le fichier '{final_path}' existe déjà dans le bucket '{bucket_name}'. Sauvegarde ignorée."
            )
            return None
        except Exception:
            logger.info(
                f"Le fichier '{final_path}' n'existe pas encore. Procédure de sauvegarde en cours."
            )

        try:
            # 📝 Écriture du DataFrame en Parquet dans le répertoire temporaire
            df.coalesce(1).write.mode("overwrite").parquet(tmp_dir)

            # 🔍 Recherche du fichier part-xxxxx.parquet
            part_file = next(
                (
                    os.path.join(root, f)
                    for root, _, files in os.walk(tmp_dir)
                    for f in files
                    if f.startswith("part-") and f.endswith(".parquet")
                ),
                None,
            )

            if not part_file:
                raise RuntimeError(
                    "Aucun fichier .parquet trouvé dans le répertoire temporaire."
                )

            # ⬆️ Upload dans MinIO
            s3_client.fput_object(
                bucket_name, final_path, part_file, content_type="application/parquet"
            )
            logger.info(
                f"Données sauvegardées dans MinIO : s3://{bucket_name}/{final_path}"
            )
            return final_path

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde : {e}", exc_info=True)
            raise

        finally:
            # 🧹 Nettoyage
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def drop_invalid_rows(self, df: DataFrame) -> DataFrame:
        """Supprime les lignes avec des données critiques manquantes."""
        logger.info("Suppression des lignes invalides")

        critical_columns = [
            "event_time",
            "event_type",
            "product_id",
            "user_id",
            "price",
        ]
        before = df.count()
        df = df.dropna(subset=critical_columns, how="any")
        after = df.count()

        logger.info(f"Lignes supprimées (NULLs critiques) : {before - after}")
        return df

    def clean_event_types(self, df: DataFrame) -> DataFrame:
        """
        Nettoie et filtre la colonne event_type en conservant uniquement les valeurs valides,
        en gérant les NULL.
        """
        logger.info("Nettoyage des types d'événements")

        # Conversion en minuscules et suppression des espaces
        df = df.withColumn("event_type", lower(trim(col("event_type"))))

        # Remplacer les valeurs null par une valeur par défaut, ici chaîne vide
        df = df.withColumn(
            "event_type",
            when(col("event_type").isNull(), "").otherwise(col("event_type")),
        )

        initial_count = df.count()

        # Filtrer uniquement les valeurs dans la liste valide
        df = df.filter(col("event_type").isin(self.valid_event_types))

        filtered_count = df.count()
        logger.info(f"Événements filtrés : {initial_count - filtered_count}")

        return df

    def clean_prices(self, df: DataFrame) -> DataFrame:
        """
        Nettoie la colonne price en remplaçant les valeurs négatives ou nulles par 0.0.
        """
        logger.info("Nettoyage des prix")

        # Conversion de la colonne price en DoubleType
        df = df.withColumn("price", col("price").cast(DoubleType()))

        # Remplacement des valeurs négatives ou nulles par 0.0
        df = df.withColumn(
            "price",
            when((col("price").isNull()) | (col("price") <= 0), 0.0).otherwise(
                col("price")
            ),
        )

        return df

    def clean_brands(self, df: DataFrame) -> DataFrame:
        """
        Nettoie la colonne brand en standardisant la casse et en remplaçant NULL par 'unknown'.
        """
        logger.info("Nettoyage de la colonne brand")
        df_clean = df.withColumn(
            "brand",
            when(col("brand").isNotNull(), trim(lower(col("brand")))).otherwise(
                lit("unknown")
            ),
        )
        return df_clean

    def clean_product_categories(self, df: DataFrame) -> DataFrame:
        """Nettoie et structure les catégories de produits."""
        logger.info("Nettoyage des catégories de produits")

        # Séparation de la catégorie en niveaux (electronics.smartphone.android -> [electronics, smartphone, android])
        df = df.withColumn(
            "category_levels",
            when(
                col("category_code").isNotNull(), split(col("category_code"), "\\.")
            ).otherwise(None),
        )
        # Extraction des principales catégories
        for i in range(3):
            df = df.withColumn(
                f"category_level_{i+1}",
                when(
                    size(col("category_levels")) > i, col("category_levels")[i]
                ).otherwise("unknown"),
            )

        return df.drop("category_levels")

    def handle_nulls_in_non_critical_columns(self, df: DataFrame) -> DataFrame:
        """Remplit ou marque les valeurs NULL dans les colonnes non critiques."""
        logger.info("Traitement des NULLs dans les colonnes non critiques")

        # Colonnes à remplir explicitement
        replacements = {
            "brand": "unknown",
            "category_level_1": "unknown",
            "category_level_2": "unknown",
            "category_level_3": "unknown",
            "category_code": "unknown",
            "user_session": "unknown",
        }

        for col_name, default_value in replacements.items():
            if col_name in df.columns:
                df = df.withColumn(
                    col_name,
                    when(col(col_name).isNull(), lit(default_value)).otherwise(
                        col(col_name)
                    ),
                )

        return df

    def remove_outliers(self, df: DataFrame) -> DataFrame:
        """
        Nettoie le DataFrame en :
        - supprimant les lignes où 'price' est null ou négatif,
        - supprimant les outliers de prix selon la méthode de l'IQR.
        """
        try:
            logger.info("Début du nettoyage des prix")

            # Suppression des valeurs nulles ou négatives
            df_clean = df.filter((col("price").isNotNull()) & (col("price") >= 0))
            count_after_clean = df_clean.count()
            logger.info(
                f"Nombre de lignes après suppression des prix nuls/négatifs : {count_after_clean}"
            )

            # Calcul des quartiles Q1 et Q3
            quantiles = df_clean.approxQuantile("price", [0.25, 0.75], 0.001)
            if len(quantiles) < 2:
                logger.warning(
                    "Impossible de calculer les quartiles Q1 et Q3. Pas de suppression d'outliers."
                )
                return df_clean

            Q1, Q3 = quantiles
            IQR = Q3 - Q1
            seuil_min = Q1 - 1.5 * IQR
            seuil_max = Q3 + 1.5 * IQR

            logger.info(f"Q1: {Q1}, Q3: {Q3}, IQR: {IQR}")
            logger.info(f"Seuils d'outliers : < {seuil_min} ou > {seuil_max}")

            # Filtrer les valeurs dans les seuils (hors outliers)
            df_filtered = df_clean.filter(
                (col("price") >= seuil_min) & (col("price") <= seuil_max)
            )
            count_after_filter = df_filtered.count()
            logger.info(
                f"Nombre de lignes après suppression des outliers : {count_after_filter}"
            )

            logger.info("Nettoyage des prix terminé")
            return df_filtered

        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des outliers : {e}", exc_info=True)
            # En cas d'erreur, on retourne le DataFrame original sans modification
            return df

    def drop_duplicates(self, df: DataFrame) -> DataFrame:
        """Supprime les doublons basés sur des colonnes clés."""
        logger.info("Suppression des doublons")

        critical_columns = [
            "event_time",
            "event_type",
            "product_id",
            "user_id",
            "user_session",
        ]
        initial_count = df.count()
        df = df.dropDuplicates(critical_columns)
        after_count = df.count()

        logger.info(f"Doublons supprimés : {initial_count - after_count}")
        return df

    def anonymize_sensitive_data(self, df: DataFrame) -> DataFrame:
        """Anonymise les données sensibles avec SHA-256 et salage pour conformité RGPD."""
        logger.info("Anonymisation sécurisée des données sensibles")

        # Génération d'un sel unique (peut être stocké de manière sécurisée ailleurs)
        salt = os.urandom(32).hex()  # Sel de 32 bytes

        # Anonymisation avec SHA-256 + sel
        df = df.withColumn(
            "user_id", sha2(concat(col("user_id").cast("string"), lit(salt)), 256)
        )

        df = df.withColumn(
            "user_session", sha2(concat(col("user_session"), lit(salt)), 256)
        )

        logger.info("Données sensibles anonymisées avec SHA-256 et salage")
        return df

    def clean_pipeline(self, df: DataFrame) -> DataFrame:
        """Exécute le pipeline complet de nettoyage des données."""
        logger.info("Début du pipeline de nettoyage")

        # Journalisation des statistiques initiales
        initial_count = df.count()
        logger.info(f"Nombre initial d'enregistrements : {initial_count}")

        try:
            # Étape 1: Nettoyage de base
            df = self.drop_invalid_rows(df)
            df = self.clean_event_types(df)
            df = self.clean_prices(df)
            df = self.clean_brands(df)

            # Étape 2: Nettoyage avancé
            df = self.clean_product_categories(df)
            df = self.handle_nulls_in_non_critical_columns(df)
            df = self.remove_outliers(df)
            df = self.drop_duplicates(df)

            # Étape 3: Conformité RGPD
            df = self.anonymize_sensitive_data(df)

            # Journalisation des résultats
            final_count = df.count()
            logger.info(f"Nombre final d'enregistrements : {final_count}")
            logger.info(f"Taux de rétention : {(final_count/initial_count)*100:.2f}%")

            return df
        except Exception as e:
            logger.error(f"Erreur dans le pipeline de nettoyage : {e}", exc_info=True)
            raise

    def calculate_data_quality_metrics(self, df: DataFrame) -> Dict[str, float]:
        """
        Calcule et retourne des métriques complètes de qualité des données pour un DataFrame Spark.

        Args:
            df (DataFrame): Le DataFrame Spark à analyser

        Returns:
            Dict[str, float]: Dictionnaire des métriques de qualité avec:
                - Taux de complétude par colonne (colonne_completeness)
                - Distribution des types d'événements (si colonne event_type présente)

        Logs:
            - Affiche un tableau détaillé de la complétude par colonne
            - Affiche la distribution des types d'événements (si applicable)
        """
        logger.info("Début de l'analyse de qualité des données")
        metrics = {}
        total_rows = df.count()

        # 1. Analyse de complétude des colonnes
        completeness_table = PrettyTable()
        completeness_table.field_names = [
            "Colonne",
            "Taux Complétude (%)",
            "Valeurs Manquantes",
        ]

        for column in df.columns:
            non_null_count = df.filter(df[column].isNotNull()).count()
            null_count = total_rows - non_null_count
            completeness = (
                (non_null_count / total_rows) * 100 if total_rows > 0 else 0.0
            )

            metrics[f"{column}_completeness"] = completeness
            completeness_table.add_row([column, f"{completeness:.2f}%", null_count])

        logger.info("\nAnalyse de complétude:\n" + completeness_table.get_string())

        # 2. Analyse de distribution des événements
        if "event_type" in df.columns:
            event_dist = (
                df.groupBy("event_type")
                .count()
                .orderBy("count", ascending=False)
                .collect()
            )
            event_table = PrettyTable()
            event_table.field_names = ["Event Type", "Count", "Percentage (%)"]

            for row in event_dist:
                percent = (row["count"] / total_rows) * 100
                metrics[f"event_type_{row['event_type']}_percent"] = percent
                event_table.add_row(
                    [row["event_type"], row["count"], f"{percent:.2f}%"]
                )

            logger.info("\nDistribution des événements:\n" + event_table.get_string())

        return metrics

    def display_descriptive_statistics(self, df: DataFrame, max_display_rows: int = 20):
        """
        Affiche les statistiques descriptives complètes d'un DataFrame Spark.

        Args:
            df (DataFrame): Le DataFrame à analyser
            max_display_rows (int): Nombre maximum de lignes à afficher pour éviter les logs trop longs

        Logs:
            - Statistiques numériques (moyenne, écart-type, min, max, etc.)
            - Distribution des types d'événements (si colonne event_type présente)
        """
        # 1. Statistiques pour les colonnes numériques
        numeric_columns = [
            f.name
            for f in df.schema.fields
            if isinstance(f.dataType, (DoubleType, LongType, IntegerType))
        ]

        if numeric_columns:
            logger.info("Statistiques descriptives des colonnes numériques:")
            stats = df.select(numeric_columns).describe().collect()

            # Affichage formaté
            stats_table = PrettyTable()
            stats_table.field_names = ["Statistique"] + numeric_columns

            for i, row in enumerate(stats):
                if i >= max_display_rows:
                    logger.info(f"... (affichage limité à {max_display_rows} lignes)")
                    break
                stats_table.add_row(
                    [row["summary"]] + [row[col] for col in numeric_columns]
                )

            logger.info("\n" + stats_table.get_string())

            # Affichage brut (pour debug)
            df.select(numeric_columns).describe().show(truncate=False)

        # 2. Analyse des catégories (si colonne event_type présente)
        if "event_type" in df.columns:
            logger.info("Distribution des catégories d'événements:")
            distribution = (
                df.groupBy("event_type").count().orderBy(col("count").desc()).collect()
            )

            dist_table = PrettyTable()
            dist_table.field_names = ["Type d'Événement", "Nombre", "Pourcentage"]
            total_count = df.count()

            for i, row in enumerate(distribution):
                if i >= max_display_rows:
                    logger.info(f"... (affichage limité à {max_display_rows} lignes)")
                    break
                percentage = (
                    (row["count"] / total_count) * 100 if total_count > 0 else 0
                )
                dist_table.add_row(
                    [row["event_type"], row["count"], f"{percentage:.2f}%"]
                )

            logger.info("\n" + dist_table.get_string())

    def compare_data_quality(
        self, before_df: DataFrame, after_df: DataFrame, dataset_name: str = "Dataset"
    ):
        """
        Compare la qualité des données avant et après un traitement.

        Args:
            before_df (DataFrame): DataFrame avant traitement
            after_df (DataFrame): DataFrame après traitement
            dataset_name (str): Nom du dataset pour identification dans les logs

        Logs:
            - Tableau comparatif des métriques avant/après
            - Gains relatifs pour chaque métrique
        """
        logger.info(f"\nCOMPARAISON DE QUALITÉ - {dataset_name}")

        # Analyse avant traitement
        logger.info("\nÉTAT INITIAL:")
        self.display_descriptive_statistics(before_df)
        initial_metrics = self.calculate_data_quality_metrics(before_df)

        # Analyse après traitement
        logger.info("\nÉTAT FINAL:")
        self.display_descriptive_statistics(after_df)
        final_metrics = self.calculate_data_quality_metrics(after_df)

        # Tableau comparatif
        logger.info("\nANALYSE DES AMÉLIORATIONS:")
        comparison_table = PrettyTable()
        comparison_table.field_names = [
            "Métrique",
            "Avant",
            "Après",
            "Différence Absolue",
            "Amélioration Relative (%)",
        ]

        for metric in final_metrics:
            before_value = initial_metrics.get(metric, 0)
            after_value = final_metrics[metric]
            absolute_diff = after_value - before_value
            relative_improvement = (
                (absolute_diff / before_value * 100)
                if before_value != 0
                else float("inf")
            )

            comparison_table.add_row(
                [
                    metric,
                    f"{before_value:.2f}",
                    f"{after_value:.2f}",
                    f"{absolute_diff:+.2f}",
                    f"{relative_improvement:+.2f}%" if before_value != 0 else "N/A",
                ]
            )

        logger.info("\n" + comparison_table.get_string())

    def run_pipeline_cleaning(self):
        """Exécute le pipeline de traitement des données."""
        logger.info("Démarrage du pipeline de traitement des données.")

        try:
            # 1. Convertir les fichiers csv en parquet
            valid_files = self.detect_valid_files_in_directory(
                directory_path=self.input_parquet
            )

            for file_path in valid_files:
                file_name = Path(file_path).name
                s3_object_key = f"{self.output_s3}/{file_name}"

                logger.info(f"\nTraitement du fichier : {file_name}")

                if self.file_exists_in_minio(
                    self.s3_client, self.s3_bucket, s3_object_key
                ):
                    logger.info(f"Existe déjà dans MinIO : {file_name}")
                    continue

                df_raw = self.load_data(file_path)
                df_clean = self.clean_pipeline(df_raw)
                self.compare_data_quality(df_raw, df_clean, file_name)
                # self.save_cleaned_data(df_clean, self.output_data, file_name)
                self.save_cleaned_data_minio(
                    df_clean,
                    bucket_name=self.s3_bucket,
                    output_dir=self.output_s3,
                    file_name=file_name,
                    s3_client=self.s3_client,
                )

            logger.info(f"Pipeline de traitement de données terminé avec succès.")

        except Exception as e:
            logger.error("Échec du pipeline de traitement de données", exc_info=True)
            raise

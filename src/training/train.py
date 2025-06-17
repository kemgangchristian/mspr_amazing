#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 3.0        ############
#######################################################

from encodings import ptcp154
import os
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import tempfile
import shutil
import zipfile
from py4j.protocol import Py4JJavaError
from minio import Minio
from minio.error import S3Error
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import (
    col,
    lit,
    count,
    sum as _sum,
    max as _max,
    datediff,
    countDistinct,
    when,
)
import polars as pl
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans as PySparkKMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline, PipelineModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CustomerSegmentation:
    def __init__(
        self,
        spark: SparkSession,
        temp_dir: str,
        s3_client: Minio,
        s3_bucket: str,
        s3_prefix_in: str,
        s3_prefix_out: str,
        s3_prefix_img: str,
        n_clusters: int = 5,
    ):
        """
        Initialise le module de segmentation des clients.

        Args:
            spark: Session Spark configurée
            n_clusters: Nombre initial de clusters pour K-Means
        """
        self.spark = spark
        self.temp_dir = temp_dir
        self.s3_client = s3_client
        self.s3_bucket = s3_bucket
        self.s3_prefix_in = s3_prefix_in
        self.s3_prefix_out = s3_prefix_out
        self.s3_prefix_img = s3_prefix_img
        self.n_clusters = n_clusters
        self.model: Optional[PipelineModel] = None
        self.scaler = StandardScaler(
            inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True
        )

    def list_parquet_files_from_minio(self) -> List[str]:
        """
        Retourne la liste des chemins de fichiers .parquet stockés dans un bucket MinIO sous un préfixe.
        """
        try:
            if not self.s3_client.bucket_exists(self.s3_bucket):
                raise ValueError(f"Le bucket '{self.s3_bucket}' n'existe pas.")

            objects = list(
                self.s3_client.list_objects(
                    self.s3_bucket, self.s3_prefix_in, recursive=True
                )
            )

            parquet_files = [
                obj.object_name
                for obj in objects
                if obj.object_name.endswith(".parquet")
            ]

            if not parquet_files:
                raise ValueError(
                    f"Aucun fichier .parquet trouvé dans {self.s3_bucket}/{self.s3_prefix_in}"
                )
            return parquet_files

        except S3Error as e:
            logger.error(f"Erreur accès MinIO: {e}", exc_info=True)
            raise

    def load_spark_dataframes_from_parquet_files(
        self, parquet_files: List[str]
    ) -> List[DataFrame]:
        """
        Charge des fichiers .parquet depuis MinIO via S3A et retourne une liste de DataFrames PySpark.
        """
        dfs = []

        for file in parquet_files:
            path = f"s3a://{self.s3_bucket}/{file}"
            try:
                df = self.spark.read.parquet(path)
                dfs.append(df)
            except Exception as e:
                logger.error(
                    f"Erreur de lecture: '{path}' : {e}",
                    exc_info=True,
                )
                raise
        return dfs

    def merge_dataframes_with_schema_check(
        self, dataframes: List[DataFrame]
    ) -> DataFrame:
        """
        Fusionne une liste de DataFrames PySpark en un seul après avoir vérifié la compatibilité des schémas.

        Args:
            dataframes (List[DataFrame]): Liste de DataFrames à fusionner.

        Returns:
            DataFrame: DataFrame fusionné.

        Raises:
            ValueError: Si la liste est vide ou si les schémas sont incompatibles.
        """
        if not dataframes:
            raise ValueError("Liste de DataFrames vide")

        base_cols = set(dataframes[0].columns)
        for df in dataframes[1:]:
            if set(df.columns) != base_cols:
                logger.warning("Incohérence de schéma détectée")
        merged_df = dataframes[0]
        for df in dataframes[1:]:
            merged_df = merged_df.unionByName(df, allowMissingColumns=True)
        logger.info(
            f"Fusion de {len(dataframes)} DataFrames réussie : {merged_df.count()} lignes, {len(merged_df.columns)} colonnes"
        )
        return merged_df  # .limit(100000)

    def calculate_rfm(self, df: DataFrame, reference_date=None) -> DataFrame:
        """
        Calcule les métriques RFM (Recency, Frequency, Monetary) à partir d'événements d'achat.

        Args:
            df (DataFrame): Données d'événements utilisateur avec colonnes ["user_id", "event_type", "event_time", "price", "product_id"]
            reference_date (datetime, optional): Date de référence pour le calcul de la recency. Par défaut = max(event_time)

        Returns:
            DataFrame: DataFrame avec les colonnes [user_id, recency, frequency, monetary, unique_products]

        # Exemple : forcer la date de référence au 31 mai 2025
        # rfm_df = calculate_rfm(df, reference_date=datetime(2025, 5, 31))
        """

        logger.info("🚀 Début du calcul RFM")

        try:
            start_time = datetime.now()

            # Vérifie si des achats existent
            purchase_df = df.filter(col("event_type") == "purchase")
            if purchase_df.isEmpty():
                logger.warning("Aucun achat trouvé dans les données. RFM non calculé.")
                return (
                    df.select("user_id")
                    .distinct()
                    .withColumn("recency", lit(None))
                    .withColumn("frequency", lit(0))
                    .withColumn("monetary", lit(0.0))
                    .withColumn("unique_products", lit(0))
                )

            # Déterminer la date de référence
            if reference_date is None:
                reference_date = purchase_df.agg(_max("event_time")).collect()[0][0]
                # reference_date = purchase_df.agg(_max("event_time").alias("max_date")).collect()[0]["max_date"]

            # Calcul des métriques RFM
            rfm = purchase_df.groupBy("user_id").agg(
                datediff(lit(reference_date), _max("event_time")).alias("recency"),
                count("*").alias("frequency"),
                _sum("price").alias("monetary"),
                countDistinct("product_id").alias("unique_products"),
            )

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"RFM calculé en {duration:.2f}s. {rfm.count()} utilisateurs avec achat."
            )

            return rfm

        except Exception:
            logger.error("Erreur dans calculate_rfm", exc_info=True)
            raise

    def prepare_features(self, df: DataFrame) -> DataFrame:
        """Prépare les features utilisateurs avec PySpark."""
        logger.info("Début de la préparation des features")

        try:

            rfm = self.calculate_rfm(df)

            behavior = df.groupBy("user_id").agg(
                _sum(when(col("event_type") == "view", 1).otherwise(0)).alias("views"),
                _sum(when(col("event_type") == "cart", 1).otherwise(0)).alias("carts"),
                _sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias(
                    "purchases"
                ),
                (
                    _sum(when(col("event_type") == "purchase", 1).otherwise(0))
                    / count("*")
                ).alias("conversion_rate"),
            )

            features = (
                rfm.join(behavior, on="user_id", how="left")
                .fillna(0)
                .filter(col("frequency") > 0)
            )

            df_features = features.repartition(6)
            logger.info(f"Features préparées. Nombre de lignes : {df_features.count()}")
            return df_features

        except Exception:
            logger.error("Erreur dans prepare_features", exc_info=True)
            raise

    def determine_optimal_clusters(self, data: DataFrame, max_k: int = 8) -> int:
        """Détermine le nombre optimal de clusters avec validation croisée,
        en affichant silhouette et méthode du coude (WSS) sur le même graphique,
        avec légende claire."""
        logger.info("Début de la détermination du nombre optimal de clusters")

        try:
            assembler = VectorAssembler(
                inputCols=[c for c in data.columns if c != "user_id"],
                outputCol="features",
            )
            scaler = StandardScaler(
                inputCol="features",
                outputCol="scaledFeatures",
                withStd=True,
                withMean=False,
            )
            evaluator = ClusteringEvaluator(
                featuresCol="scaledFeatures",
                predictionCol="prediction",
                metricName="silhouette",
                distanceMeasure="squaredEuclidean",
            )

            silhouette_scores = []
            wss = []
            k_values = range(2, max_k + 1)

            for k in k_values:
                kmeans = PySparkKMeans(
                    featuresCol="scaledFeatures",
                    predictionCol="prediction",
                    k=k,
                    seed=42,
                    maxIter=20,
                    tol=1e-4,
                )
                pipeline = Pipeline(stages=[assembler, scaler, kmeans])
                model = pipeline.fit(data)
                predictions = model.transform(data)

                predictions.select("scaledFeatures", "prediction").show(
                    5, truncate=False
                )

                score = evaluator.evaluate(predictions)
                silhouette_scores.append(score)

                wss.append(model.stages[-1].summary.trainingCost)

                logger.info(f"K={k} - Silhouette: {score:.4f} - WSS: {wss[-1]:.4f}")

            # Graphique avec double axe Y et légende claire
            with tempfile.NamedTemporaryFile(
                dir=self.temp_dir, suffix=".png", delete=False
            ) as tmp:
                fig, ax1 = plt.subplots(figsize=(10, 6))

                color_sil = "tab:blue"
                ax1.set_xlabel("Nombre de clusters (K)")
                ax1.set_ylabel("Score de silhouette", color=color_sil)
                l1 = ax1.plot(
                    k_values,
                    silhouette_scores,
                    "o-",
                    color=color_sil,
                    label="Silhouette Score",
                )
                ax1.tick_params(axis="y", labelcolor=color_sil)
                ax1.grid(True, which="both", linestyle="--", alpha=0.6)

                ax2 = ax1.twinx()
                color_wss = "tab:red"
                ax2.set_ylabel("WSS (Within-Cluster SSE)", color=color_wss)
                l2 = ax2.plot(
                    k_values, wss, "s--", color=color_wss, label="WSS (Elbow Method)"
                )
                ax2.tick_params(axis="y", labelcolor=color_wss)

                plt.title(
                    "Optimisation du nombre de clusters : Silhouette & Méthode du coude"
                )

                # Légende combinée
                lines = l1 + l2
                labels = [line.get_label() for line in lines]
                ax1.legend(lines, labels, loc="best", fontsize=10)

                fig.tight_layout()
                plt.savefig(tmp.name)
                plt.close()

                logger.info(f"Image temporairement sauvegardée : {tmp.name}")

                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                s3_path_img = (
                    f"{self.s3_prefix_img}/cluster_optimization_{timestamp}.png"
                )
                self.s3_client.fput_object(self.s3_bucket, s3_path_img, tmp.name)
                os.remove(tmp.name)  # ✅ suppression du fichier temporaire
                logger.info(
                    f"Image uploadée vers s3://{self.s3_bucket}/{s3_path_img} et suppression du fichier temporaire: {tmp.name}"
                )

            optimal_k = k_values[np.argmax(silhouette_scores)]
            logger.info(f"Nombre optimal de clusters déterminé: {optimal_k}")

            return optimal_k

        except Exception as e:
            logger.error("Erreur dans determine_optimal_clusters", exc_info=True)
            raise

    def train_model(self, df: DataFrame) -> PipelineModel:
        logger.info("Entraînement du modèle de segmentation")
        try:
            start_time = datetime.now()
            self.n_clusters = self.determine_optimal_clusters(df)

            pipeline = Pipeline(
                stages=[
                    VectorAssembler(
                        inputCols=[c for c in df.columns if c != "user_id"],
                        outputCol="features",
                    ),
                    self.scaler,
                    PySparkKMeans(
                        featuresCol="scaledFeatures", k=self.n_clusters, seed=42
                    ),
                ]
            )

            self.model = pipeline.fit(df)
            predictions = self.model.transform(df)

            evaluator = ClusteringEvaluator(featuresCol="scaledFeatures")
            silhouette = evaluator.evaluate(predictions)

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Modèle entraîné en {duration:.2f}s avec silhouette={silhouette:.4f}"
            )
            return self.model

        except Exception:
            logger.error("Erreur dans train_model", exc_info=True)
            raise

    def save_model_to_s3a(self, model: PipelineModel) -> None:
        """
        Sauvegarde le modèle directement dans un chemin s3a://bucket/path/model
        en supposant que Spark est configuré avec Hadoop S3A.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3a_model_path = (
                f"s3a://{self.s3_bucket}/{self.s3_prefix_out}/model_{timestamp}"
            )
            model.save(s3a_model_path)
            logger.info(f"Modèle sauvegardé avec succès sur S3A : {s3a_model_path}")
        except Exception as e:
            logger.error(
                f"Erreur lors de la sauvegarde du modèle sur S3A: {str(e)}",
                exc_info=True,
            )
            raise

    def save_model_to_minio_zip(self, model: PipelineModel) -> None:
        """
        Sauvegarde le modèle Spark localement, zippe le dossier, puis upload le zip sur S3 via s3a://.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. Sauvegarder localement dans un dossier temporaire
            with tempfile.TemporaryDirectory(dir=self.temp_dir, delete=False) as tmpdir:
                local_model_path = f"{tmpdir}/model_{timestamp}"
                model.save(local_model_path)
                logger.info(f"Modèle sauvegardé localement dans : {local_model_path}")

                # 2. Compresser le dossier en zip
                zip_filename = f"{tmpdir}/model_{timestamp}.zip"
                with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
                    # Parcourir tous les fichiers du dossier model pour les zipper
                    for root, dirs, files in os.walk(local_model_path):
                        for file in files:
                            filepath = os.path.join(root, file)
                            # Ajouter au zip avec chemin relatif pour garder la structure
                            arcname = os.path.relpath(filepath, start=local_model_path)
                            zipf.write(filepath, arcname)
                logger.info(f"Modèle compressé dans le fichier : {zip_filename}")

                # 3. Upload du zip sur S3 via hadoop fs s3a (si self.s3_client est un client compatible)
                s3_path = f"{self.s3_prefix_out}/model_{timestamp}.zip"

                # Exemple avec le client MinIO / s3 compatible utilisé dans ton code (fput_object)
                self.s3_client.fput_object(
                    self.s3_bucket,
                    s3_path,
                    zip_filename,
                )
                os.remove(zip_filename)  # ✅ suppression du fichier zip temporaire
                logger.info(
                    f"Fichier zip du modèle uploadé vers s3a://{self.s3_bucket}/{s3_path} et suppression du fichier zip temporaire: {zip_filename}"
                )

        except Py4JJavaError as e:
            logger.error(
                "Erreur Java Spark lors de la sauvegarde du modèle", exc_info=True
            )
            raise
        except Exception as e:
            logger.error(
                f"Erreur lors de la sauvegarde du modèle zippé sur S3A: {str(e)}",
                exc_info=True,
            )
            raise

    def analyze_and_visualize(self, predictions: DataFrame) -> Dict:
        logger.info("Analyse des clusters")

        try:
            pdf = predictions.select(
                "user_id",
                "prediction",
                "recency",
                "frequency",
                "monetary",
                "conversion_rate",
            ).toPandas()

            cluster_stats = (
                pdf.groupby("prediction")
                .agg(
                    {
                        "recency": ["mean", "std"],
                        "frequency": ["mean", "std"],
                        "monetary": ["mean", "std"],
                        "user_id": "count",
                    }
                )
                .reset_index()
            )

            plt.figure(figsize=(10, 6))
            sns.countplot(x="prediction", data=pdf)
            plt.title("Distribution des clusters")

            with tempfile.NamedTemporaryFile(
                dir=self.temp_dir, suffix=".png", delete=False
            ) as tmp:
                plt.savefig(tmp.name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_path_img = (
                    f"{self.s3_prefix_img}/cluster_distribution{timestamp}.png"
                )
                self.s3_client.fput_object(
                    self.s3_bucket,
                    s3_path_img,
                    tmp.name,
                )
                os.remove(tmp.name)  # ✅ suppression du fichier temporaire
                logger.info(
                    f"Image uploadée vers s3://{self.s3_bucket}/{s3_path_img} et suppression du fichier temporaire: {tmp.name}"
                )
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                cluster_stats.set_index("prediction").xs("mean", axis=1, level=1),
                annot=True,
                fmt=".1f",
            )
            plt.title("Caractéristiques par cluster")

            with tempfile.NamedTemporaryFile(
                dir=self.temp_dir, suffix=".png", delete=False
            ) as tmp:
                plt.savefig(tmp.name)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                s3_path_img = (
                    f"{self.s3_prefix_img}/cluster_characteristics{timestamp}.png"
                )
                self.s3_client.fput_object(
                    self.s3_bucket,
                    s3_path_img,
                    tmp.name,
                )
                os.remove(tmp.name)  # ✅ suppression du fichier temporaire
                logger.info(
                    f"Image uploadée vers s3://{self.s3_bucket}/{s3_path_img} et suppression du fichier temporaire: {tmp.name}"
                )

            descriptions = {}
            for cluster in sorted(pdf["prediction"].unique()):
                subset = pdf[pdf["prediction"] == cluster]
                descriptions[f"Cluster {cluster}"] = {
                    "size": len(subset),
                    "percentage": f"{(len(subset)/len(pdf))*100:.1f}%",
                    "characteristics": {
                        "recency_mean": subset["recency"].mean(),
                        "frequency_mean": subset["frequency"].mean(),
                        "monetary_mean": subset["monetary"].mean(),
                        "conversion_rate_mean": subset["conversion_rate"].mean(),
                    },
                }

            logger.info("Analyse terminée")
            return descriptions

        except Exception:
            logger.error("Erreur dans analyze_and_visualize", exc_info=True)
            raise

    def generate_segment_names(self, descriptions: Dict) -> Dict:
        logger.info("Génération des noms de segments")

        try:
            segment_names = {}
            for cluster, stats in descriptions.items():
                c = stats["characteristics"]

                if c["recency_mean"] < 30 and c["monetary_mean"] > 500:
                    segment_names[cluster] = "Clients Premium"
                elif c["frequency_mean"] > 10 and c["monetary_mean"] < 200:
                    segment_names[cluster] = "Chasseurs de Promos"
                elif c["recency_mean"] > 90:
                    segment_names[cluster] = "Clients Dormants"
                elif c["conversion_rate_mean"] < 0.1:
                    segment_names[cluster] = "Window Shoppers"
                else:
                    segment_names[cluster] = f"Cluster Moyen {cluster}"

            logger.info(f"Noms de segments générés: {segment_names}")
            return segment_names

        except Exception:
            logger.error("Erreur dans generate_segment_names", exc_info=True)
            raise

    def run_pipeline_model(self) -> DataFrame:  # Dict:
        """Exécute le pipeline complet de segmentation."""
        logger.info("Démarrage du pipeline de segmentation client")

        try:
            # 1. Récupérer la liste des fichiers parquet
            parquet_files = self.list_parquet_files_from_minio()

            # 2. Charger les fichiers parquet en DataFrames Spark
            dfs = self.load_spark_dataframes_from_parquet_files(parquet_files)

            # 3. Fusionner les DataFrames
            merged_df = self.merge_dataframes_with_schema_check(dfs)

            # 4. Préparer les features pour la segmentation
            features_df = self.prepare_features(merged_df)

            # 5. Entraîner le modèle KMeans
            model = self.train_model(features_df)

            # 6. Sauvegarder le modèle dans Minio (S3)
            # self.save_model_to_s3a(model)
            self.save_model_to_minio_zip(model)

            # 7. Appliquer le modèle sur les données pour obtenir les clusters
            predictions = model.transform(features_df)

            # 8. Analyser les clusters et générer des visualisations
            descriptions = self.analyze_and_visualize(predictions)

            # 9. Générer des noms lisibles pour chaque segment
            segment_names = self.generate_segment_names(descriptions)

            # 10. Associer les noms aux descriptions
            for cluster, name in segment_names.items():
                descriptions[cluster]["segment_name"] = name

            results = {
                "model": model,
                "prediction": predictions,
                "descriptions": descriptions,
                "segment_names": segment_names,
            }
            logger.info(f"Pipeline terminé avec succès : {results}")
            return results

        except Exception as e:
            logger.error("Échec du pipeline de segmentation", exc_info=True)
            raise

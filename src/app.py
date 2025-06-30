#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 3.0        ############
#######################################################

import streamlit as st
import os
import tempfile
import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import PipelineModel
from minio import Minio
from typing import List, Dict
from datetime import datetime
import plotly.express as px

# --- Configuration MinIO
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minio")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minio123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET_NAME", "mspr")

MODEL_PREFIX = "models"
IMAGE_PREFIX = "images"

# --- Données exactes extraites des logs
MODEL_METRICS = {
    "training_time": "24h 6min 10s",
    "silhouette_score": 0.2364,  # Score final
    "silhouette_score_ksearch": 0.9551,  # Score pendant la recherche K
    "optimal_clusters": 2,
    "total_customers": 2_443_708,
    "model_saved_date": "26/06/2025 14:03",
    "features": [
        "user_id",
        "recency",
        "frequency",
        "monetary",
        "views",
        "carts",
        "purchases",
        "conversion_rate",
        "unique_products",
    ],
    "image_optimization": "cluster_optimization_20250626-120756.png",
    "image_distribution": "cluster_distribution20250626_142010.png",
    "image_characteristics": "cluster_characteristics20250626_142010.png",
}

CLUSTER_DESCRIPTIONS = {
    0: {
        "name": "Clients Dormants",
        "size": "1,394,752 (57.1%)",
        "description": "Clients inactifs depuis longtemps (128 jours en moyenne) avec une fréquence d'achat faible (1.4/mois) mais un panier moyen intéressant (269€).",
        "strategy": """
        - Campagnes de reactivation ciblées
        - Offres spéciales pour le retour
        - Enquêtes de satisfaction
        """,
        "color": "#FF6B6B",
        "icon": "💤",
        "characteristics": {
            "recency_mean": 128.43,
            "frequency_mean": 1.39,
            "monetary_mean": 269.49,
            "conversion_rate_mean": 0.10,
        },
    },
    1: {
        "name": "Clients Actifs Moyens",
        "size": "1,048,956 (42.9%)",
        "description": "Clients avec une activité récente (46 jours en moyenne) et une fréquence d'achat élevée (3.2/mois), avec un panier moyen important (704€).",
        "strategy": """
        - Programmes de fidélisation
        - Recommandations personnalisées
        - Offres pour augmenter le panier moyen
        """,
        "color": "#2EC4B6",
        "icon": "🛒",
        "characteristics": {
            "recency_mean": 45.58,
            "frequency_mean": 3.22,
            "monetary_mean": 704.60,
            "conversion_rate_mean": 0.12,
        },
    },
}

# --- Style CSS personnalisé
custom_css = """
<style>
    /* Titres */
    h1, h2, h3, h4 {
        color: #2C3E50 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Cartes */
    .card {
        border-radius: 12px;
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        border-left: 5px solid;
    }
    
    /* Boutons */
    .stButton>button {
        border-radius: 10px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s;
        background-color: #4B8BBE;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Métriques */
    .stMetric {
        border-radius: 10px;
        padding: 15px;
        background-color: #F8F9FA;
    }
    
    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        padding: 12px 24px !important;
        transition: all 0.3s !important;
    }
    
    /* Couleurs des clusters */
    .cluster-0 { border-color: #FF6B6B; }
    .cluster-1 { border-color: #2EC4B6; }
</style>
"""


@st.cache_resource
def get_spark() -> SparkSession:
    return (
        SparkSession.builder.appName("Streamlit Cluster Viewer")
        .config("spark.driver.memory", "2g")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.access.key", MINIO_ACCESS_KEY)
        .config("spark.hadoop.fs.s3a.secret.key", MINIO_SECRET_KEY)
        .config("spark.hadoop.fs.s3a.endpoint", f"http://{MINIO_ENDPOINT}")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.11.1026",
        )
        .getOrCreate()
    )


def init_minio_client() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def list_models(client: Minio) -> List[str]:
    objects = client.list_objects(MINIO_BUCKET, MODEL_PREFIX, recursive=True)
    return [obj.object_name for obj in objects if obj.object_name.endswith(".zip")]


def download_and_unzip_model(client: Minio, object_name: str) -> str:
    data = client.get_object(MINIO_BUCKET, object_name)
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "model.zip")

    with open(zip_path, "wb") as f:
        f.write(data.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    return temp_dir


def load_model_from_minio(client: Minio, model_path: str) -> PipelineModel:
    temp_model_dir = download_and_unzip_model(client, model_path)
    return PipelineModel.load(temp_model_dir)


def display_model_metrics(minio_client):
    st.markdown("### ⚙️ Métriques du Modèle")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Silhouette (modèle final)",
            f"{MODEL_METRICS['silhouette_score']:.4f}",
            help="Score obtenu après l'entraînement complet avec K=2",
        )
        st.metric(
            "Silhouette (recherche K)",
            f"{MODEL_METRICS['silhouette_score_ksearch']:.4f}",
            help="Score optimal identifié pendant la recherche du nombre de clusters",
        )

    with col2:
        st.metric(
            "Temps d'entraînement",
            MODEL_METRICS["training_time"],
            help="Durée totale pour entraîner le modèle final",
        )
        st.metric(
            "Date du modèle",
            MODEL_METRICS["model_saved_date"],
            help="Dernière date de sauvegarde du modèle",
        )

    with col3:
        st.metric(
            "Clients analysés",
            f"{MODEL_METRICS['total_customers']:,}",
            help="Nombre total de clients inclus dans l'analyse",
        )
        st.metric(
            "Variables utilisées",
            len(MODEL_METRICS["features"]),
            help="Nombre de caractéristiques incluses dans le modèle",
        )

    st.markdown("---")

    # Image de l'optimisation
    st.markdown("### 📈 Optimisation du Nombre de Clusters")
    try:
        optimization_img = minio_client.get_object(
            MINIO_BUCKET, f"{IMAGE_PREFIX}/{MODEL_METRICS['image_optimization']}"
        )
        st.image(
            optimization_img.read(),
            caption="Recherche du nombre optimal de clusters (méthode du coude / silhouette)",
        )
    except Exception as e:
        st.warning(f"Impossible de charger l'image d'optimisation: {str(e)}")

    st.markdown(
        f"""
    **Résumé** :
    - 📌 Pendant la recherche, le **score silhouette le plus élevé** était **{MODEL_METRICS['silhouette_score_ksearch']:.4f}**, obtenu avec **K = {MODEL_METRICS['optimal_clusters']}**.
    - 🚀 Après entraînement final sur toutes les données, le modèle atteint un **score silhouette global** de **{MODEL_METRICS['silhouette_score']:.4f}**.
    - 👥 Le modèle a été entraîné sur **{MODEL_METRICS['total_customers']:,} clients**.
    """
    )

    st.markdown("---")

    # Images de visualisation
    st.markdown("### 🧭 Répartition et Caractéristiques des Clusters")
    col1, col2 = st.columns(2)

    with col1:
        try:
            distribution_img = minio_client.get_object(
                MINIO_BUCKET, f"{IMAGE_PREFIX}/{MODEL_METRICS['image_distribution']}"
            )
            st.image(distribution_img.read(), caption="Répartition des clients")
        except Exception as e:
            st.warning(f"Erreur de chargement (distribution): {str(e)}")

    with col2:
        try:
            characteristics_img = minio_client.get_object(
                MINIO_BUCKET, f"{IMAGE_PREFIX}/{MODEL_METRICS['image_characteristics']}"
            )
            st.image(characteristics_img.read(), caption="Caractéristiques moyennes")
        except Exception as e:
            st.warning(f"Erreur de chargement (caractéristiques): {str(e)}")


def display_cluster_radar(cluster_id: int):
    if cluster_id not in CLUSTER_DESCRIPTIONS:
        return

    characteristics = CLUSTER_DESCRIPTIONS[cluster_id]["characteristics"]
    df = pd.DataFrame(
        dict(
            r=[
                characteristics["recency_mean"] / 150,
                characteristics["frequency_mean"] / 5,
                characteristics["monetary_mean"] / 800,
                characteristics["conversion_rate_mean"] * 10,
            ],
            theta=["Ancienneté", "Fréquence", "Panier Moyen", "Taux Conversion"],
        )
    )

    fig = px.line_polar(
        df,
        r="r",
        theta="theta",
        line_close=True,
        color_discrete_sequence=[CLUSTER_DESCRIPTIONS[cluster_id]["color"]],
        template="plotly_white",
    )
    fig.update_traces(fill="toself")
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=f"Profil du Cluster {cluster_id}",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def display_cluster_comparison():
    clusters = []
    for cluster_id, info in CLUSTER_DESCRIPTIONS.items():
        char = info["characteristics"]
        clusters.append(
            {
                "Cluster": f"{cluster_id} - {info['name']}",
                "Ancienneté (jours)": char["recency_mean"],
                "Fréquence (achats/mois)": char["frequency_mean"],
                "Panier Moyen (€)": char["monetary_mean"],
                "Taux Conversion": char["conversion_rate_mean"],
                "Taille": info["size"],
            }
        )

    df = pd.DataFrame(clusters)
    fig = px.bar(
        df,
        x="Cluster",
        y=["Ancienneté (jours)", "Fréquence (achats/mois)", "Panier Moyen (€)"],
        barmode="group",
        color_discrete_sequence=["#FF6B6B", "#2EC4B6", "#4B8BBE"],
        title="Comparaison des Clusters",
    )

    st.plotly_chart(fig, use_container_width=True)


def display_cluster_info(cluster_id: int):
    if cluster_id not in CLUSTER_DESCRIPTIONS:
        st.warning(f"Aucune information disponible pour le cluster {cluster_id}")
        return

    info = CLUSTER_DESCRIPTIONS[cluster_id]

    st.markdown(
        f"""
    <div class="card cluster-{cluster_id}">
        <h2>{info['icon']} Cluster {cluster_id}: {info['name']} <span style="float: right; font-size: 16px; color: #7F8C8D;">{info['size']} des clients</span></h2>
        <p style="color: {info['color']}; font-weight: bold;">{info['description']}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 📈 Caractéristiques Moyennes")
        st.dataframe(
            pd.DataFrame.from_dict(
                info["characteristics"], orient="index", columns=["Valeur"]
            ).style.format(
                {
                    "recency_mean": "{:.1f} jours",
                    "frequency_mean": "{:.1f}/mois",
                    "monetary_mean": "€{:.1f}",
                    "conversion_rate_mean": "{:.1%}",
                }
            ),
            use_container_width=True,
        )

    with col2:
        st.markdown(f"#### 🎯 Stratégie Recommandée")
        st.markdown(
            f"""
        <div style="background-color: #F8F9FA; padding: 15px; border-radius: 10px; border-left: 4px solid {info['color']}">
            {info['strategy']}
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    display_cluster_radar(cluster_id)


def build_client_form():
    st.markdown("## 🧑‍💼 Profil du Client")
    st.markdown(
        "Complétez les informations ci-dessous pour personnaliser l’analyse du comportement client."
    )

    st.divider()

    user_id = st.text_input("🆔 ID Client", value="CLT_12345")

    col1, col2 = st.columns(2)

    with col1:
        recency = st.slider(
            "📆 Dernier achat (jours)", min_value=0, max_value=365, value=128
        )
        frequency = st.slider(
            "🔁 Fréquence d'achat (par mois)", min_value=0, max_value=10, value=1
        )
        monetary = st.slider(
            "💰 Panier moyen (€)", min_value=0, max_value=2000, value=270
        )
        conversion_rate = (
            st.slider(
                "📈 Taux de conversion (%)", min_value=0.0, max_value=100.0, value=10.0
            )
            / 100.0
        )

    with col2:
        views = st.slider("👀 Pages vues/mois", min_value=0, max_value=500, value=10)
        carts = st.slider("🛒 Paniers créés/mois", min_value=0, max_value=50, value=1)
        purchases = st.slider("✅ Achats/mois", min_value=0, max_value=20, value=0)
        unique_products = st.slider(
            "📦 Produits uniques consultés", min_value=0, max_value=100, value=5
        )

    st.divider()

    user_data = {
        "user_id": user_id,
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "views": views,
        "carts": carts,
        "purchases": purchases,
        "conversion_rate": conversion_rate,
        "unique_products": unique_products,
    }

    return user_data


def predict_client_segment(spark: SparkSession, model: PipelineModel, user_data: Dict):
    try:
        # Ne pas inclure user_id dans les features pour la prédiction
        features = [f for f in MODEL_METRICS["features"] if f != "user_id"]
        values = [user_data[f] for f in features]

        user_df = spark.createDataFrame([values], features)
        prediction = model.transform(user_df)
        cluster_id = prediction.select("prediction").collect()[0][0]
        return int(cluster_id)
    except Exception as e:
        st.error(f"Erreur de prédiction : {str(e)}")
        return None


def display_client_prediction(user_data: Dict, cluster_id: int):
    if cluster_id not in CLUSTER_DESCRIPTIONS:
        st.warning("Cluster non reconnu")
        return

    info = CLUSTER_DESCRIPTIONS[cluster_id]

    st.markdown(
        f"""
    <div style="background-color: {info['color']}20; padding: 20px; border-radius: 10px; border-left: 5px solid {info['color']}; margin-bottom: 20px;">
        <h2 style="color: {info['color']}; margin: 0;">{info['icon']} Cluster Prédit: {cluster_id} - {info['name']}</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Métriques clés
    st.markdown("### 📊 Profil Client")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Ancienneté", f"{user_data['recency']} jours")
        st.metric("Fréquence", f"{user_data['frequency']}/mois")

    with col2:
        st.metric("Panier moyen", f"{user_data['monetary']:.0f} €")
        st.metric("Produits uniques", user_data["unique_products"])

    with col3:
        st.metric("Taux conversion", f"{user_data['conversion_rate']:.1%}")
        st.metric("Pages vues", user_data["views"])

    # Recommandations
    st.markdown("### 💡 Recommandations")
    st.markdown(
        f"""
    <div style="background-color: #F8F9FA; padding: 15px; border-radius: 10px;">
        {info['strategy']}
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="MSPR-AMAZING",
        page_icon="🛍️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Injecter le CSS personnalisé
    st.markdown(custom_css, unsafe_allow_html=True)

    # Initialisation des clients
    spark = get_spark()
    minio_client = init_minio_client()

    # Header
    st.markdown(
        """
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #2C3E50; font-weight: 700; margin-bottom: 10px;">
            🛍️ Customer Segmentation Dashboard
        </h1>
        <p style="font-size: 18px; color: #7F8C8D;">
            Optimisez votre stratégie marketing grâce à l'analyse prédictive des segments clients
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.markdown(
        """
    <div style="margin-bottom: 30px;">
        <h2 style="color: #2C3E50;">⚙️ Configuration</h2>
    </div>
    """,
        unsafe_allow_html=True,
    )

    models_list = list_models(minio_client)
    if not models_list:
        st.error("Aucun modèle disponible dans MinIO")
        return

    selected_model = st.sidebar.selectbox(
        "Modèle de Segmentation",
        models_list,
        help="Sélectionnez le modèle à utiliser pour la prédiction",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ℹ️ À propos")
    st.sidebar.info(
        f"""
    **Données analysées** : {MODEL_METRICS['total_customers']:,} clients  
    **Dernier entraînement** : {MODEL_METRICS['model_saved_date']}  
    **Score Silhouette** : {MODEL_METRICS['silhouette_score']:.4f}  
    **Clusters** : {MODEL_METRICS['optimal_clusters']}
    """
    )

    # Chargement du modèle
    with st.spinner("Chargement du modèle..."):
        model = load_model_from_minio(minio_client, selected_model)

    # Onglets
    tab1, tab2, tab3 = st.tabs(["🔍 Analyse Client", "📊 Segments", "⚙️ Modèle"])

    with tab1:
        st.markdown("### 🔍 Analyse d'un Client")
        user_data = build_client_form()

        if st.button("✨ Prédire le Segment", type="primary"):
            with st.spinner("Analyse en cours..."):
                cluster_id = predict_client_segment(spark, model, user_data)
                if cluster_id is not None:
                    st.success("Prédiction terminée avec succès")
                    display_client_prediction(user_data, cluster_id)

    with tab2:
        st.markdown("### 📊 Analyse des Segments")
        selected_cluster = st.selectbox(
            "Sélectionnez un segment à explorer",
            options=list(CLUSTER_DESCRIPTIONS.keys()),
            format_func=lambda x: f"Cluster {x}: {CLUSTER_DESCRIPTIONS[x]['name']}",
        )

        display_cluster_info(selected_cluster)
        display_cluster_comparison()

    with tab3:
        st.markdown("### ⚙️ Performance du Modèle")
        display_model_metrics(minio_client)


if __name__ == "__main__":
    main()

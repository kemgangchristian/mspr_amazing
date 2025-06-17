#!/bin/bash

#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 1.1        ############
#######################################################

# 🔧 Définition du répertoire racine du projet
ROOT_DIR="TEST"  # Vous pouvez changer "TEST" par "." pour utiliser le dossier courant

echo "🔧 Initialisation de la structure du projet dans : $ROOT_DIR"
echo "---------------------------------------------------------------"

# 💡 Fonction utilitaire pour créer un dossier s’il n’existe pas encore
create_dir() {
  local dir_path="$1"
  if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
    echo "✅ Dossier créé : $dir_path"
  else
    echo "ℹ️  Dossier déjà existant : $dir_path"
  fi
}

# 💡 Fonction utilitaire pour créer un fichier s’il n’existe pas encore
create_file() {
  local file_path="$1"
  if [ ! -f "$file_path" ]; then
    touch "$file_path"
    echo "✅ Fichier créé : $file_path"
  else
    echo "ℹ️  Fichier déjà existant : $file_path"
  fi
}

# 📁 Liste des répertoires à créer (organisée par composants du projet)
DIRS=(
  "$ROOT_DIR/data/processed"                     # Données traitées au format Parquet
  "$ROOT_DIR/data/raw/csv"                       # Données brutes en CSV à ingérer
  "$ROOT_DIR/data/raw/parquet"                   # Données brutes converties en Parquet
  "$ROOT_DIR/data/tmp"                           # Fichiers temporaires
  "$ROOT_DIR/logs"                               # Fichiers de logs applicatifs
  "$ROOT_DIR/minio"                              # Répertoire local simulant un stockage S3 (MinIO)
  "$ROOT_DIR/models"                             # Répertoire local du modèle
  "$ROOT_DIR/monitoring/elasticsearch/config"    # Configurations pour Elasticsearch
  "$ROOT_DIR/monitoring/elasticsearch/data"      # Données persistées par Elasticsearch
  "$ROOT_DIR/monitoring/grafana/data"            # Données persistées par Grafana
  "$ROOT_DIR/monitoring/logstash/config"         # Configs de Logstash
  "$ROOT_DIR/monitoring/logstash/pipeline"       # Pipelines Logstash (ingestion de données)
  "$ROOT_DIR/src/conf"                           # Fichiers de configuration Python
  "$ROOT_DIR/src/processing"                     # Scripts de traitement des données
  "$ROOT_DIR/src/training"                       # Scripts de génération du modèle
  "$ROOT_DIR/src/utils"                          # Fonctions utilitaires (logger, helpers)
)

# 📄 Liste des fichiers à créer (fichiers de configuration, code, documentation, etc.)
FILES=(
  "$ROOT_DIR/.env"                                             # Fichier d'environnement
  "$ROOT_DIR/.gitignore"                                       # Fichier pour exclure des fichiers de Git
  "$ROOT_DIR/docker-compose.yml"                               # Orchestration des services Docker
  "$ROOT_DIR/Dockerfile"                                       # Image Docker de l'application
  "$ROOT_DIR/README.md"                                        # Documentation du projet
  "$ROOT_DIR/monitoring/elasticsearch/config/elasticsearch.yml" # Config ES
  "$ROOT_DIR/monitoring/logstash/config/logstash.yml"           # Config Logstash
  "$ROOT_DIR/monitoring/logstash/config/pipelines.yml"          # Config pipelines Logstash
  "$ROOT_DIR/monitoring/logstash/pipeline/spark.conf"           # Pipeline personnalisé Logstash
  "$ROOT_DIR/requirements.txt"                                  # Dépendances Python
  "$ROOT_DIR/src/conf/settings.py"                              # Variables de config Python
  "$ROOT_DIR/src/processing/converter.py"                       # Script de conversion CSV → Parquet
  "$ROOT_DIR/src/processing/data_cleaning.py"                   # Nettoyage et prétraitement des données
  "$ROOT_DIR/src/training/train.py"                             # création et génération du modèle
  "$ROOT_DIR/src/utils/logger.py"                               # Utilitaire de logging Python
  "$ROOT_DIR/src/main.py"                                       # Point d'entrée de l'application
)

# 🚀 Création des dossiers un par un via la fonction `create_dir`
echo -e "\n📦 Création des répertoires..."
for dir in "${DIRS[@]}"; do
  create_dir "$dir"
done

# 📝 Création des fichiers un par un via la fonction `create_file`
echo -e "\n📝 Création des fichiers..."
for file in "${FILES[@]}"; do
  create_file "$file"
done

# 📂 Affichage de la structure du projet une fois la création terminée
# Utilise `tree` si installé, sinon utilise `find`
echo -e "\n📁 Structure actuelle dans $ROOT_DIR :"
command -v tree &> /dev/null && tree "$ROOT_DIR" -L 3 || find "$ROOT_DIR" -maxdepth 3

# ✅ Message de succès final
echo -e "\n✅ Structure du projet créée avec succès. 🚀"

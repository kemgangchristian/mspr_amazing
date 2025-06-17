#######################################################
############## EPSI (2025): MSPR AMAZING ##############
##############         Version: 2.0        ############
#######################################################

# Étape 1: Image de base avec version spécifique
FROM apache/spark:4.0.0

# Étape 2: Configuration système
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        libcurl4-openssl-dev \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Étape 3: Installation des dépendances Python
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Étape 4: Téléchargement des JARs requis
ENV HADOOP_VERSION=3.3.6
ENV AWS_SDK_VERSION=1.12.262
ENV SPARK_JARS_DIR=/opt/spark/jars

RUN mkdir -p ${SPARK_JARS_DIR} && \
    curl -sL -o ${SPARK_JARS_DIR}/hadoop-aws-${HADOOP_VERSION}.jar \
      https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/${HADOOP_VERSION}/hadoop-aws-${HADOOP_VERSION}.jar && \
    curl -sL -o ${SPARK_JARS_DIR}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar \
      https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/${AWS_SDK_VERSION}/aws-java-sdk-bundle-${AWS_SDK_VERSION}.jar

# Ajoutez ces lignes après l'installation des dépendances
#COPY conf/spark-defaults.conf /opt/spark/conf/
#COPY conf/log4j2.properties /opt/spark/conf/

# Étape 5: Copie du code source
COPY ./src /opt/spark/src

# Étape 6: Configuration finale
WORKDIR /opt/spark

ENV PYTHONPATH=/opt/spark:$PYTHONPATH

USER spark

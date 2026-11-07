FROM apache/airflow:2.9.3-python3.11

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Java 17 for PySpark, procps for 'ps'
# The JDK folder name depends on the CPU (java-17-openjdk-amd64 on Intel,
# -arm64 on Apple Silicon), so link it to one fixed path that works on both.
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sfn /usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture) /usr/lib/jvm/java-17

ENV JAVA_HOME=/usr/lib/jvm/java-17
ENV PATH=$PATH:$JAVA_HOME/bin

RUN mkdir -p /opt/airflow/datamart /opt/airflow/model_bank /opt/airflow/data /opt/airflow/utils && \
    chown -R airflow:root /opt/airflow

WORKDIR /opt/airflow
COPY requirements.txt ./

USER airflow
RUN pip install --no-cache-dir -r requirements.txt

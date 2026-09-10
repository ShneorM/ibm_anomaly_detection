FROM python:3.11.0-slim

WORKDIR /app

# Install curl dependency
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN mkdir -p conf src data/massaged data/labels results/model_experiments trained_models

# Download dataset dependencies
RUN curl -L -o data/labels/anomaly_windows.csv https://zenodo.org/record/14062900/files/anomaly_windows.csv?download=1 && \
    curl -L -o data/massaged/pivoted_data_all.parquet https://zenodo.org/record/14062900/files/pivoted_data_all.parquet

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Run anomaly detector
CMD ["python", "src/run_experiment__multi_models_GRU_ANN.py"]

DATASET_HOME=/mnt/datasets

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d ${DATASET_HOME}/llama3.1-8b \
  https://inference.mlcommons-storage.org/metadata/llama3-1-8b-cnn-eval.uri

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d ${DATASET_HOME}/deepseek-r1 \
  https://inference.mlcommons-storage.org/metadata/deepseek-r1-datasets-fp8-eval.uri

bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) \
  -d ${DATASET_HOME}/whisper \
  https://inference.mlcommons-storage.org/metadata/whisper-dataset.uri

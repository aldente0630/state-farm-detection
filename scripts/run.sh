PROJECT_DIR=~/projects/state-farm-detection; VERSION=1
PROJECT_DIR=${PROJECT_DIR}/models/serving/${VERSION}

docker pull tensorflow/serving

docker run -t --rm -p 8501:8501 \
    -v ${PROJECT_DIR}:/models/state-farm-detection/${VERSION} \
    -e MODEL_NAME=state-farm-detection \
    tensorflow/serving &
docker pull tensorflow/serving

docker run -t --rm -p 8501:8501 \
    -v "../saved_model:/models/state-farm-detection" \
    -e MODEL_NAME=state-farm-detection \
    tensorflow/serving &
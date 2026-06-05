FROM tensorflow/tensorflow:1.4.0-py3

WORKDIR /workspace

RUN pip install --no-cache-dir \
    numpy==1.16.4 \
    scipy==1.2.3 \
    scikit-learn==0.20.4 \
    pandas==0.24.2 \
    networkx==1.11 \
    configparser \
    Theano==0.9.0

ENV PYTHONPATH=/workspace/models/RelationPrediction-master/code:/workspace/models/RelationPrediction-master/code/optimization:/workspace/models/RelationPrediction-master/code/optimization/tensorflow_backend:/workspace/models/RelationPrediction-master/code/optimization/theano_backend
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONIOENCODING=utf-8
FROM nvidia/cuda:12.5.0-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install flwr torch torchvision numpy matplotlib pandas

COPY server.py /app/server.py
COPY lstm_ae.py /app/lstm_ae.py
COPY training.py /app/training.py

WORKDIR /app

CMD ["python3", "server.py"]
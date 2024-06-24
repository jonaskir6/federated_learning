FROM nvidia/cuda:11.0-runtime

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install flwr torch numpy matplotlib

COPY server.py /app/server.py
COPY network.py /app/network.py
COPY train.py /app/train.py

WORKDIR /app

CMD ["python3", "client.py"]
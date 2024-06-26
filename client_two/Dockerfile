FROM nvidia/cuda:12.5.0-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip3 install flwr torch torchvision numpy sys matplotlib

COPY client_two.py /app/client_two.py
COPY network.py /app/network.py
COPY train.py /app/train.py

WORKDIR /app

CMD ["python3", "client.py"]
import os
import re
import yaml
import shutil
import subprocess

# local - max_clients will be tested
print("Info: The number of available data files needs to be at least as high the maximum number of clients that should be tested.")
print("Info: The data files need to be named \"client_num\" and placed in the \"all_data\" folder")
max_clients = input("Enter the maximum number of clients that is tested: ")
data_path = input("Enter data path")
assert(max_clients <= len([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])), "Not enough data for the maximum client number"


def prepare_data():
    print("Info: Preparing data...")
    for i in range (1, max_clients + 1):
        os.makedirs(f"./client/data/client_{i}")
        shutil.copy(f"./all_data/client_{i}.csv", f"./client/data/client_{i}/fl_data.csv")
    print("Info: Data prepared successfully")


def generate_dockerfiles():
    print("Info: Generating dockerfiles...")
    for i in range(1, max_clients + 1):
        content = f"""FROM nvidia/cuda:12.5.0-runtime-ubuntu20.04

            RUN apt-get update && apt-get install -y python3 python3-pip

            RUN pip3 install flwr torch torchvision numpy matplotlib pandas seaborn scikit-learn scipy

            COPY client.py /app/client.py
            COPY lstm_ae.py /app/lstm_ae.py
            COPY training.py /app/training.py
            COPY data/client{i}/fl_data.csv /app/fl_data.csv

            WORKDIR /app

            CMD ["python3", "client.py"]
            """
        with open(f"./client/Dockerfile_{i}", "w") as dockerfile:
            dockerfile.write(content)

    print(f"Info: {max_clients} dockerfiles successfully generated")


def generate_docker_compose(num_clients):
    print("Info: Generating docker-compose file...")
    compose = {
        "version": "3.8",
        "services": {
            "server": {
                "build": {
                    "context": "./server",
                    "dockerfile": "Dockerfile",
                },
                "ports": ["5000:5000"],
                "volumes": ["./server:/app"],
            }
        }
    }

    for i in range(1, num_clients + 1):
        client_name = f"client_{i}"
        compose["services"][client_name] = {
            "build": {
                "context": "./client",
                "dockerfile": f"Dockerfile_{i}",
            },
            "ports": [f"500{i}:500{i}"],
            "volumes": [f"./d_data/{client_name}:/app/data"],
            "runtime": "nvidia",
            "deploy": {
                "resources": {
                    "reservations": {
                        "devices": [
                            {
                                "driver": "nvidia",
                                "count": 1,
                                "capabilities": ["gpu"]
                            }
                        ]
                    }
                }
            },
            "depends_on": ["server"]
        }

    with open("docker-compose.yml", "w") as file:
        yaml.dump(compose, file, default_flow_style=False, sort_keys=False)

    print(f"Info: docker-compose.yml generated successfully with {num_clients} clients")


def run_experiment(num_clients, output_dir):
    print(f"Info: Starting experiment with {num_clients} clients...")
    generate_docker_compose(num_clients)
    
    subprocess.run(["docker-compose", "up", "-d"], check=True)

    # get server container ID
    result = subprocess.run(["docker", "ps", "--format", "{{.ID}} {{.Names}}"], 
                            stdout=subprocess.PIPE, text=True, check=True)
    container_info = result.stdout.strip().split("\n")

    server_id = None
    for line in container_info:
        if "server" in line.split()[1]:
            server_id = line.split()[0]
            break

    # copy model
    model_path = "/app/models/model_round_10.pth"
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["docker", "cp", f"{server_id}:{model_path}", f"{output_dir}/model_{num_clients}.pth"], check=True)
    print(f"Info: Model copied from server container {server_id} to {output_dir}")

    subprocess.run(["docker-compose", "down"], check=True)
    print(f"Info: Finished experiment with {num_clients} clients successfully")


prepare_data()
generate_dockerfiles()

for i in range(2, max_clients + 1):
    run_experiment(i, "results/models")


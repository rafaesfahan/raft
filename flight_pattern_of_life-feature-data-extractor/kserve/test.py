import torch
import requests
example_input = torch.rand(3, 5, 100)

# Local test for Jupyter notebook in Kurrent
cluster_ip = "aircraft-trajectory.demo.svc.cluster.local"
model_name = "aircraft-trajectory"
predict_url = f"http://{cluster_ip}/v1/models/{model_name}:predict"

# KServe expects a specific payload format
payload = {"instances": example_input.tolist()}

headers = {"Content-Type": "application/json"}

response = requests.post(predict_url,
              headers=headers,
              json=payload)

response.json()
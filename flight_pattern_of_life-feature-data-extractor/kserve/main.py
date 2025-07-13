import base64
import io
import os
from typing import Dict

import kserve
import requests
import torch
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error

load_dotenv("docker.env")


class AircraftTrajectoryPredictor(kserve.Model):
    def __init__(self, name: str, model_file: str):
        super().__init__(name)
        self.name = name
        self.model_file = model_file
        self.model = None
        self.ready = False
        self.minio_client = Minio(
            os.environ.get("MINIO_ENDPOINT", "minio-service.kubeflow:9000"),
            access_key=os.environ.get("MINIO_ACCESS_KEY"),
            secret_key=os.environ.get("MINIO_SECRET_KEY"),
            secure=False,  # Set to True if using HTTPS
        )
        self.load()

    def load(self):
        try:
            # First, check if the model file exists locally
            if os.path.exists(self.model_file):
                model_path = self.model_file
            else:
                # If not local, check if the model file exists in the MinIO bucket
                bucket_name = os.environ.get("MINIO_BUCKET", "aanikevich")
                self.minio_client.stat_object(bucket_name, self.model_file)

                # If the file exists in MinIO, download it to a temporary location
                model_path = f"/tmp/{self.model_file}"
                self.minio_client.fget_object(bucket_name, self.model_file, model_path)

            # Load the model using the file path
                self.model = torch.jit.load(model_path)
            # Clean up the temporary file if it was downloaded from MinIO
            if model_path.startswith("/tmp/"):
                os.remove(model_path)
            
            # Model ready
            self.ready = True
        except S3Error as e:
            raise RuntimeError(f"Error accessing model file in MinIO: {str(e)}")
        except FileNotFoundError:
            raise RuntimeError(f"Model file not found: {self.model_file}")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def predict(self, request: Dict, headers: Dict) -> Dict:
        # Implement prediction logic here
        if not self.ready:
            raise RuntimeError("Model is not loaded")

        # Process input and make predictions
        input_data = request["instances"]

        # Raw input data
        print(input_data)

        # Convert input_data to tensor
        format_data = torch.tensor(input_data)

        out_loaded = self.model(format_data)

        results = out_loaded.detach().cpu().numpy().tolist()

        # Return the JSON response
        return {"predictions": results}


if __name__ == "__main__":
    model_file = os.environ.get("MODEL_FILE", "path/to/default/model.pt")
    model_name = os.environ.get("MODEL_NAME", "aircraft-trajectory")
    predictor = AircraftTrajectoryPredictor(
        model_name,
        model_file
    )
    kserve.ModelServer().start([predictor])

# kserve-template

This repo shows an example of how to set up kserve for a Pytorch model. First the model is saved as a torch.jit .pt file. This file is uploaded to a minio bucket.

Then the file can be used as a basis for kServe model and served in Kurrent. For building the latest image, run the following commands:

```bash
cd ./kserve
```

```bash
docker login ghcr.io
```

```bash
docker build --platform linux/amd64 -t ghcr.io/raft-tech/kserve-example/aircraft-trajectory:0.1.0 .
```

```bash
docker push --platform linux/amd64 ghcr.io/raft-tech/kserve-example/aircraft-trajectory:0.1.0
```

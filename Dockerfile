FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "scripts/deploy_api.py"]

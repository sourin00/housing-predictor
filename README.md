# MLOps Pipeline - California Housing Prediction

## Overview
This project implements an end-to-end MLOps pipeline for California Housing price prediction using:
- Scikit-learn Linear Regression
- PyTorch Neural Network
- Docker containerization
- GitHub Actions CI/CD
- Manual model quantization

## Repository Links
- GitHub: [https://github.com/sourin00/housing-predictor]
- Docker Hub: [https://hub.docker.com/repository/docker/sourin00/housing-predictor/general]

## Branches
- `main`: Initial setup and documentation
- `dev`: Model development and training
- `docker_ci`: Docker containerization and CI/CD pipeline
- `quantization`: Model quantization and optimization

## Setup Instructions
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python src/train.py`
4. Build Docker image: `docker build -t housing-predictor .`
5. Run container: `docker run housing-predictor python src/predict.py`

## Results Comparison
| Metric | Original Sklearn Model | Quantized Model |
|--------|------------------------|-----------------|
| R² Score | 0.575788 | -0.179863 |
| Model Size | 0.035 KB | 0.020 KB |

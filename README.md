# MLOps Pipeline - California Housing Prediction

## Overview
This project implements an end-to-end MLOps pipeline for California Housing price prediction using:
- Scikit-learn Linear Regression
- PyTorch Neural Network
- Docker containerization
- GitHub Actions CI/CD
- Manual model quantization

## Repository Links
- GitHub: [Your GitHub Repository URL]
- Docker Hub: [Your Docker Hub Repository URL]

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
| R² Score | [To be filled] | [To be filled] |
| Model Size | [To be filled] KB | [To be filled] KB |
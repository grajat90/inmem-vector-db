#!/bin/bash

# Exit on any error
set -e

echo "Building Docker image for stackai-vector-db in Minikube's Docker environment..."
eval $(minikube docker-env)
docker build -t stackai-vector-db:latest .

echo "Checking for existing Helm release..."
if helm list | grep -q "vector-db"; then
    echo "Upgrading existing Helm release..."
    helm upgrade vector-db ./stackai-vector-db
else
    echo "Installing new Helm release..."
    helm install vector-db ./stackai-vector-db
fi

echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/vector-db-stackai-vector-db
kubectl port-forward svc/vector-db-stackai-vector-db 8000:8000 &

echo "Deployment successful! Here's how to access the application:"
echo "Run: kubectl port-forward svc/vector-db-stackai-vector-db 8000:8000"
echo "Then visit: http://localhost:8000 in your browser"
echo "API documentation is available at: http://localhost:8000/docs" 


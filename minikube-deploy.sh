#!/bin/bash

# Exit on any error
set -e

echo "Building Docker image for vector-db in Minikube's Docker environment..."
eval $(minikube docker-env)
docker build -t vector-db:latest .

echo "Checking for existing Helm release..."
if helm list | grep -q "vector-db"; then
    echo "Upgrading existing Helm release..."
    helm upgrade vector-db ./vector-db
else
    echo "Installing new Helm release..."
    helm install vector-db ./vector-db
fi

echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/vector-db
kubectl port-forward svc/vector-db 8000:8000 &
# Store the PID of the port-forwarding process
PORT_FORWARD_PID=$!

echo "Port forwarding started with PID: $PORT_FORWARD_PID"
echo "To stop port forwarding, run: kill $PORT_FORWARD_PID"


echo "Deployment successful! Here's how to access the application:"
echo "Run: kubectl port-forward svc/vector-db 8000:8000"
echo "Then visit: http://localhost:8000 in your browser"
echo "API documentation is available at: http://localhost:8000/docs" 


#!/bin/bash

# Exit on any error
set -e

# Set script directory as working directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.."

echo "Building Docker image for vector-db in Minikube's Docker environment..."
# Check if minikube is installed
if ! command -v minikube &> /dev/null; then
    echo "Error: minikube is not installed. Please install minikube first."
    exit 1
fi

# Check if minikube is running
if ! minikube status | grep -q "Running"; then
    echo "Error: minikube is not running. Please start minikube with 'minikube start'."
    exit 1
fi

eval $(minikube docker-env)
docker build -t vector-db:latest -f deployment/Dockerfile .

# Check if COHERE_API_KEY is set
if [ -z "$COHERE_API_KEY" ]; then
    echo "Warning: COHERE_API_KEY environment variable is not set."
    echo "The application may not function correctly without it."
    echo "Set it with: export COHERE_API_KEY=your_api_key"
    COHERE_API_KEY_PARAM=""
else
    COHERE_API_KEY_PARAM="--set secrets.cohereApiKey=$COHERE_API_KEY"
fi

echo "Checking for existing Helm release..."
if helm list | grep -q "vector-db"; then
    echo "Upgrading existing Helm release..."
    helm upgrade vector-db deployment/vector-db $COHERE_API_KEY_PARAM
else
    echo "Installing new Helm release..."
    helm install vector-db deployment/vector-db $COHERE_API_KEY_PARAM
fi

echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/vector-db

echo "Deployment successful! Here's how to access the application:"
echo "Run: kubectl port-forward svc/vector-db 8000:8000"
echo "Then visit: http://localhost:8000 in your browser"
echo "API documentation is available at: http://localhost:8000/docs"

# Start port forwarding in the background
kubectl port-forward svc/vector-db 8000:8000 &
PORT_FORWARD_PID=$!

echo "Port forwarding started with PID: $PORT_FORWARD_PID"
echo "To stop port forwarding, run: kill $PORT_FORWARD_PID" 


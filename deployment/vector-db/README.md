# Vector Database Helm Chart

This Helm chart deploys the Vector Database application on Kubernetes.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- PV provisioner support in the underlying infrastructure (for persistence)
- Cohere API key

## Installing the Chart

To install the chart with the release name `vector-db`:

```bash
# Set your Cohere API key
export COHERE_API_KEY=your_cohere_api_key

# Install the chart
helm install vector-db ./deployment/vector-db --set secrets.cohereApiKey=$COHERE_API_KEY
```

Alternatively, you can use the provided script for Minikube:

```bash
export COHERE_API_KEY=your_cohere_api_key
./deployment/minikube-deploy.sh
```

## Configuration

The following table lists the configurable parameters of the Vector Database chart and their default values.

| Parameter                   | Description                                       | Default                 |
|-----------------------------|---------------------------------------------------|-------------------------|
| `replicaCount`              | Number of replicas                                | `1`                     |
| `image.repository`          | Image repository                                  | `vector-db`             |
| `image.tag`                 | Image tag                                         | `latest`                |
| `image.pullPolicy`          | Image pull policy                                 | `Never`                 |
| `service.type`              | Kubernetes Service type                           | `ClusterIP`             |
| `service.port`              | Service port                                      | `8000`                  |
| `persistence.enabled`       | Enable persistence using PVC                      | `true`                  |
| `persistence.storageClass`  | Storage class for PVC                             | `""`                    |
| `persistence.accessMode`    | Access mode for PVC                               | `ReadWriteOnce`         |
| `persistence.size`          | Size of PVC                                       | `1Gi`                   |
| `persistence.mountPath`     | Mount path for persistent volume                  | `/app/data`             |
| `secrets.cohereApiKey`      | Cohere API key                                    | `""`                    |
| `env`                       | Environment variables for the application         | See `values.yaml`       |
| `resources`                 | CPU/Memory resource requests/limits               | See `values.yaml`       |

Specify each parameter using the `--set key=value[,key=value]` argument to `helm install`.

For example:

```bash
helm install vector-db ./deployment/vector-db \
  --set replicaCount=2 \
  --set persistence.size=2Gi \
  --set secrets.cohereApiKey=your_cohere_api_key
```

## Accessing the Application

After deploying the chart, you can access the application by port-forwarding the service:

```bash
kubectl port-forward svc/vector-db 8000:8000
```

Then visit:
- Application: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Persistence

This chart mounts a Persistent Volume at `/app/data`. The volume is created using dynamic volume provisioning.

## Uninstalling the Chart

To uninstall/delete the `vector-db` deployment:

```bash
helm delete vector-db
```
#  Vector Database Helm Chart

This Helm chart deploys the  Vector Database API in a Kubernetes cluster.

## Prerequisites

- Kubernetes 1.16+
- Helm 3.0+
- Docker (to build the application image)

## Building the Application Image

Before deploying the chart, you need to build the Docker image for the application:

```bash
# From the root of the project
docker build -t vector-db:latest .
```

If deploying to a remote Kubernetes cluster, you'll need to push the image to a container registry:

```bash
# Tag the image for your registry
docker tag vector-db:latest your-registry.io/vector-db:latest

# Push the image
docker push your-registry.io/vector-db:latest

# Update the values.yaml file to use your registry
# image:
#   repository: your-registry.io/vector-db
#   tag: latest
```

## Installing the Chart

To install the chart with the release name `vector-db`:

```bash
helm install vector-db ./vector-db
```

## Uninstalling the Chart

To uninstall/delete the `vector-db` deployment:

```bash
helm delete vector-db
```

## Configuration

The following table lists the configurable parameters of the  Vector Database chart and their default values.

| Parameter | Description | Default |
|-----------|-------------|---------|
| `replicaCount` | Number of replicas | `1` |
| `image.repository` | Image repository | `vector-db` |
| `image.tag` | Image tag | `latest` |
| `image.pullPolicy` | Image pull policy | `IfNotPresent` |
| `service.type` | Kubernetes Service type | `ClusterIP` |
| `service.port` | Service port | `8000` |
| `resources.limits.cpu` | CPU limits | `500m` |
| `resources.limits.memory` | Memory limits | `512Mi` |
| `resources.requests.cpu` | CPU requests | `100m` |
| `resources.requests.memory` | Memory requests | `256Mi` |
| `autoscaling.enabled` | Enable autoscaling | `false` |
| `autoscaling.minReplicas` | Minimum replicas | `1` |
| `autoscaling.maxReplicas` | Maximum replicas | `5` |
| `env` | Environment variables | See `values.yaml` |

Specify each parameter using the `--set key=value[,key=value]` argument to `helm install`.

Alternatively, a YAML file that specifies the values for the parameters can be provided while installing the chart:

```bash
helm install vector-db ./vector-db -f values.yaml
``` 
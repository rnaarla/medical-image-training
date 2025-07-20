#!/bin/bash
"""
Production Training Job Deployment Script

Deploys distributed medical image training jobs to AWS EKS cluster
with comprehensive monitoring, scaling, and fault tolerance.

Author: Medical AI Platform Team  
Version: 2.0.0
"""

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
CLUSTER_NAME=${CLUSTER_NAME:-medical-ai-cluster}
NAMESPACE=${NAMESPACE:-medical-training}
AWS_REGION=${AWS_REGION:-us-west-2}

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
check_prerequisites() {
    print_status "Checking deployment prerequisites..."
    
    # Check required commands
    for cmd in kubectl helm aws terraform; do
        if ! command -v $cmd &> /dev/null; then
            print_error "$cmd is required but not installed"
            exit 1
        fi
    done
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured"
        exit 1
    fi
    
    # Check cluster access
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot access Kubernetes cluster. Run: aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Setup namespace and RBAC
setup_namespace() {
    print_status "Setting up namespace and RBAC..."
    
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Create service account with required permissions
    cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: medical-training-sa
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: medical-training-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "endpoints"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: medical-training-binding
subjects:
- kind: ServiceAccount
  name: medical-training-sa
  namespace: $NAMESPACE
roleRef:
  kind: ClusterRole
  name: medical-training-role
  apiGroup: rbac.authorization.k8s.io
EOF
    
    print_success "Namespace and RBAC configured"
}

# Deploy NVIDIA GPU Operator
deploy_gpu_operator() {
    print_status "Deploying NVIDIA GPU Operator..."
    
    # Add NVIDIA Helm repository
    helm repo add nvidia https://nvidia.github.io/gpu-operator
    helm repo update
    
    # Install GPU Operator
    helm upgrade --install gpu-operator nvidia/gpu-operator \
        --namespace gpu-operator \
        --create-namespace \
        --set driver.enabled=true \
        --set toolkit.enabled=true \
        --set dcgmExporter.enabled=true \
        --set nodeStatusExporter.enabled=true \
        --wait
    
    print_success "GPU Operator deployed"
}

# Deploy monitoring stack
deploy_monitoring() {
    print_status "Deploying monitoring stack..."
    
    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
        --set grafana.adminPassword=admin123 \
        --wait
    
    # Deploy DCGM Exporter for GPU metrics
    cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: dcgm-exporter
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: dcgm-exporter
  template:
    metadata:
      labels:
        app: dcgm-exporter
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: dcgm-exporter
        image: nvcr.io/nvidia/k8s/dcgm-exporter:3.1.7-3.1.4-ubuntu20.04
        ports:
        - name: metrics
          containerPort: 9400
        resources:
          limits:
            nvidia.com/gpu: 1
        env:
        - name: DCGM_EXPORTER_LISTEN
          value: ":9400"
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
EOF
    
    print_success "Monitoring stack deployed"
}

# Deploy training job
deploy_training_job() {
    print_status "Deploying medical image training job..."
    
    # Get ECR repository URI
    ECR_URI=$(aws ecr describe-repositories --repository-names medical-image-training --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text 2>/dev/null || echo "")
    
    if [ -z "$ECR_URI" ]; then
        print_error "ECR repository not found. Please run terraform apply first."
        exit 1
    fi
    
    # Create training job manifest
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: medical-training-job-$(date +%s)
  namespace: $NAMESPACE
spec:
  parallelism: 2
  completions: 1
  backoffLimit: 3
  template:
    metadata:
      labels:
        app: medical-training
        version: v2.0.0
    spec:
      serviceAccountName: medical-training-sa
      restartPolicy: Never
      containers:
      - name: training
        image: $ECR_URI:latest
        command: ["/bin/bash"]
        args: ["-c", "cd /app && python train.py --distributed --mixed-precision --epochs 100 --batch-size 32"]
        resources:
          requests:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
          limits:
            memory: "32Gi"
            cpu: "8"
            nvidia.com/gpu: 1
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NCCL_DEBUG
          value: "INFO"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: training-secrets
              key: wandb-api-key
              optional: true
        volumeMounts:
        - name: data-volume
          mountPath: /data
        - name: model-cache
          mountPath: /app/models
        - name: shm
          mountPath: /dev/shm
      volumes:
      - name: data-volume
        emptyDir:
          sizeLimit: 100Gi
      - name: model-cache
        emptyDir:
          sizeLimit: 50Gi
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: 8Gi
      nodeSelector:
        accelerator: nvidia-tesla-k80
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
EOF
    
    print_success "Training job deployed"
}

# Deploy inference service
deploy_inference_service() {
    print_status "Deploying inference service..."
    
    # Get ECR repository URI
    ECR_URI=$(aws ecr describe-repositories --repository-names medical-image-training --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text 2>/dev/null || echo "")
    
    # Deploy Triton inference server using Helm
    helm upgrade --install triton-inference charts/triton \
        --namespace $NAMESPACE \
        --set image.repository=$ECR_URI \
        --set image.tag=latest \
        --set replicaCount=2 \
        --set resources.limits.nvidia\\.com/gpu=1 \
        --set autoscaling.enabled=true \
        --set autoscaling.minReplicas=1 \
        --set autoscaling.maxReplicas=5 \
        --wait
    
    print_success "Inference service deployed"
}

# Setup monitoring dashboards
setup_dashboards() {
    print_status "Setting up Grafana dashboards..."
    
    # Create ConfigMap with custom dashboard
    kubectl create configmap medical-training-dashboard \
        --from-file=monitoring/grafana-dashboard.json \
        --namespace monitoring \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Label for auto-discovery by Grafana
    kubectl label configmap medical-training-dashboard \
        grafana_dashboard=1 \
        --namespace monitoring
    
    print_success "Grafana dashboards configured"
}

# Display deployment status
show_deployment_status() {
    print_status "Deployment Status Summary"
    echo "=========================="
    
    echo ""
    echo "🚀 Cluster Information:"
    kubectl cluster-info
    
    echo ""
    echo "📊 Node Status:"
    kubectl get nodes -o wide
    
    echo ""
    echo "🏃 Running Pods:"
    kubectl get pods -n $NAMESPACE -o wide
    
    echo ""
    echo "📈 Services:"
    kubectl get svc -n $NAMESPACE
    
    echo ""
    echo "🎯 Ingress/LoadBalancer endpoints:"
    kubectl get ingress -n $NAMESPACE 2>/dev/null || echo "No ingress configured"
    
    echo ""
    echo "🔍 GPU Resources:"
    kubectl describe node | grep -A 5 "nvidia.com/gpu" || echo "No GPU nodes found"
    
    echo ""
    print_success "Deployment completed successfully!"
    echo ""
    echo "Access URLs:"
    echo "- Grafana: kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    echo "- Prometheus: kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090"
    echo "- Training Jobs: kubectl logs -n $NAMESPACE -l app=medical-training -f"
}

# Cleanup function
cleanup() {
    print_warning "Cleaning up deployment..."
    
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
    helm uninstall triton-inference -n $NAMESPACE 2>/dev/null || true
    helm uninstall prometheus -n monitoring 2>/dev/null || true
    helm uninstall gpu-operator -n gpu-operator 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Main deployment function
main() {
    print_status "Starting AWS EKS deployment..."
    
    check_prerequisites
    setup_namespace
    deploy_gpu_operator
    deploy_monitoring
    deploy_training_job
    deploy_inference_service
    setup_dashboards
    show_deployment_status
    
    print_success "🎉 Deployment pipeline completed successfully!"
}

# Handle command line arguments
case "${1:-}" in
    "deploy")
        main
        ;;
    "cleanup")
        cleanup
        ;;
    "status")
        show_deployment_status
        ;;
    *)
        echo "Usage: $0 {deploy|cleanup|status}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy the complete training platform"
        echo "  cleanup - Remove all deployed resources"
        echo "  status  - Show current deployment status"
        exit 1
        ;;
esac

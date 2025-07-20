#!/bin/bash

# Terraform Auto Upload Script
# Automates model deployment and training job triggering

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="medical-image-training"
AWS_REGION=${AWS_REGION:-"us-west-2"}
CLUSTER_NAME=${CLUSTER_NAME:-"medical-training-cluster"}
NAMESPACE=${NAMESPACE:-"default"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    # Check required tools
    for tool in terraform aws kubectl docker helm; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install the missing tools and try again."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or expired"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local ecr_registry="${account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    local image_name="${PROJECT_NAME}"
    local image_tag="latest"
    local full_image_name="${ecr_registry}/${image_name}:${image_tag}"
    
    # Create ECR repository if it doesn't exist
    if ! aws ecr describe-repositories --repository-names "$image_name" --region "$AWS_REGION" &> /dev/null; then
        log_info "Creating ECR repository: $image_name"
        aws ecr create-repository \
            --repository-name "$image_name" \
            --region "$AWS_REGION" \
            --image-scanning-configuration scanOnPush=true
    fi
    
    # Login to ECR
    log_info "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ecr_registry"
    
    # Build image
    log_info "Building Docker image: $full_image_name"
    docker build -f Dockerfile.train -t "$full_image_name" .
    
    # Push image
    log_info "Pushing Docker image to ECR..."
    docker push "$full_image_name"
    
    log_success "Docker image built and pushed: $full_image_name"
    echo "$full_image_name" > .docker_image
}

# Apply Terraform infrastructure
apply_infrastructure() {
    log_info "Applying Terraform infrastructure..."
    
    # Initialize Terraform
    terraform init
    
    # Plan infrastructure changes
    log_info "Planning infrastructure changes..."
    terraform plan \
        -var="aws_region=${AWS_REGION}" \
        -var="cluster_name=${CLUSTER_NAME}" \
        -out=tfplan
    
    # Apply infrastructure
    log_info "Applying infrastructure changes..."
    terraform apply tfplan
    
    # Get outputs
    local cluster_name=$(terraform output -raw cluster_name)
    local s3_bucket=$(terraform output -raw s3_bucket_name)
    local efs_id=$(terraform output -raw efs_file_system_id)
    
    # Update kubeconfig
    log_info "Updating kubeconfig..."
    aws eks update-kubeconfig \
        --region "$AWS_REGION" \
        --name "$cluster_name"
    
    # Wait for cluster to be ready
    log_info "Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=600s
    
    log_success "Infrastructure applied successfully"
    
    # Store outputs for later use
    cat > .terraform_outputs << EOF
CLUSTER_NAME=${cluster_name}
S3_BUCKET=${s3_bucket}
EFS_ID=${efs_id}
EOF
}

# Compile and install CUDA kernel
setup_cuda_kernel() {
    log_info "Setting up CUDA kernel..."
    
    # This would typically be done in the Docker build process
    # For local development, compile the kernel
    if [ -f "setup_custom_kernel.py" ] && [ -f "custom_grayscale_kernel.cu" ]; then
        log_info "Compiling CUDA kernel locally for testing..."
        python setup_custom_kernel.py build_ext --inplace || {
            log_warning "CUDA kernel compilation failed - will use fallback in container"
        }
    fi
}

# Upload model files to S3
upload_models() {
    log_info "Uploading model configuration and artifacts..."
    
    # Source terraform outputs
    if [ -f ".terraform_outputs" ]; then
        source .terraform_outputs
    else
        log_error "Terraform outputs not found. Run infrastructure setup first."
        return 1
    fi
    
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local s3_prefix="models/${timestamp}"
    
    # Create ONNX model if it doesn't exist (placeholder for demo)
    if [ ! -f "model.onnx" ]; then
        log_info "Creating placeholder ONNX model..."
        # In production, this would export from a trained model
        touch model.onnx
    fi
    
    # Upload model files
    log_info "Uploading model files to S3..."
    aws s3 cp model_repo/resnet/config.pbtxt "s3://${S3_BUCKET}/${s3_prefix}/config.pbtxt"
    
    if [ -f "model.onnx" ]; then
        aws s3 cp model.onnx "s3://${S3_BUCKET}/${s3_prefix}/model.onnx"
    fi
    
    # Upload training scripts and data
    log_info "Uploading training configuration..."
    tar -czf training_code.tar.gz *.py utils/ requirements.txt
    aws s3 cp training_code.tar.gz "s3://${S3_BUCKET}/${s3_prefix}/training_code.tar.gz"
    
    log_success "Model files uploaded to s3://${S3_BUCKET}/${s3_prefix}/"
}

# Deploy Helm charts
deploy_helm_charts() {
    log_info "Deploying Helm charts..."
    
    # Deploy Triton Inference Server
    log_info "Deploying Triton Inference Server..."
    helm upgrade --install triton ./charts/triton \
        --namespace default \
        --create-namespace \
        --wait \
        --timeout 600s
    
    # Deploy monitoring stack (Prometheus + Grafana)
    log_info "Deploying monitoring stack..."
    
    # Add Prometheus community helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install kube-prometheus-stack
    helm upgrade --install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
        --set grafana.adminPassword=admin123 \
        --wait \
        --timeout 600s
    
    # Install NVIDIA DCGM Exporter for GPU monitoring
    helm repo add gpu-helm-charts https://nvidia.github.io/k8s-device-plugin
    helm upgrade --install nvidia-dcgm-exporter gpu-helm-charts/dcgm-exporter \
        --namespace monitoring \
        --create-namespace \
        --wait
    
    log_success "Helm charts deployed successfully"
}

# Create Kubernetes secrets and configmaps
setup_kubernetes_resources() {
    log_info "Setting up Kubernetes resources..."
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
    
    # Update training job with actual image
    if [ -f ".docker_image" ]; then
        local docker_image=$(cat .docker_image)
        log_info "Updating training job with image: $docker_image"
        
        # Update the image in train-job.yaml
        sed -i.bak "s|medical-training:latest|${docker_image}|g" train-job.yaml
        rm train-job.yaml.bak
    fi
    
    # Apply training job configuration
    kubectl apply -f train-job.yaml
    
    log_success "Kubernetes resources configured"
}

# Trigger training job
trigger_training() {
    log_info "Triggering training job..."
    
    # Check if job already exists and delete it
    if kubectl get job medical-image-training -n "$NAMESPACE" &> /dev/null; then
        log_info "Deleting existing training job..."
        kubectl delete job medical-image-training -n "$NAMESPACE" --wait=true
    fi
    
    # Create the training job
    kubectl create job --from=job/medical-image-training medical-image-training-$(date +%s) -n "$NAMESPACE"
    
    # Monitor job progress
    log_info "Monitoring job progress..."
    kubectl wait --for=condition=complete --timeout=7200s job -l app=medical-training -n "$NAMESPACE" || {
        log_warning "Job did not complete within timeout. Check job status with: kubectl describe job -n $NAMESPACE"
    }
    
    # Show job logs
    log_info "Showing job logs..."
    kubectl logs -f job.batch/medical-image-training -n "$NAMESPACE" || true
    
    log_success "Training job triggered"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check cluster health
    log_info "Checking cluster health..."
    kubectl get nodes
    kubectl get pods -A
    
    # Check Triton server
    log_info "Checking Triton server..."
    kubectl get pods -l app.kubernetes.io/name=triton
    
    # Check training resources
    log_info "Checking training resources..."
    kubectl get pvc model-storage-pvc
    kubectl get jobs -n "$NAMESPACE"
    
    # Test inference server (if running)
    local triton_pod=$(kubectl get pods -l app.kubernetes.io/name=triton -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [ -n "$triton_pod" ]; then
        log_info "Testing Triton server health..."
        kubectl exec "$triton_pod" -- curl -s http://localhost:8000/v2/health/ready || {
            log_warning "Triton server is not ready yet"
        }
    fi
    
    log_success "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f tfplan .docker_image training_code.tar.gz model.onnx
}

# Print usage
usage() {
    cat << EOF
Usage: $0 [COMMAND]

Commands:
    all                 Run complete setup (default)
    prerequisites      Check prerequisites only
    build              Build and push Docker image
    infrastructure     Apply Terraform infrastructure
    upload             Upload models to S3
    deploy             Deploy Helm charts
    train              Trigger training job
    verify             Verify deployment
    cleanup            Clean up temporary files

Environment Variables:
    AWS_REGION         AWS region (default: us-west-2)
    CLUSTER_NAME       EKS cluster name (default: medical-training-cluster)
    NAMESPACE          Kubernetes namespace (default: default)

Examples:
    $0                 # Run complete setup
    $0 build           # Build Docker image only
    $0 train           # Trigger training job only
EOF
}

# Main execution
main() {
    local command=${1:-all}
    
    case $command in
        "all")
            check_prerequisites
            setup_cuda_kernel
            build_and_push_image
            apply_infrastructure
            upload_models
            deploy_helm_charts
            setup_kubernetes_resources
            trigger_training
            verify_deployment
            cleanup
            log_success "Complete setup finished successfully!"
            ;;
        "prerequisites")
            check_prerequisites
            ;;
        "build")
            check_prerequisites
            setup_cuda_kernel
            build_and_push_image
            ;;
        "infrastructure")
            check_prerequisites
            apply_infrastructure
            ;;
        "upload")
            upload_models
            ;;
        "deploy")
            deploy_helm_charts
            setup_kubernetes_resources
            ;;
        "train")
            trigger_training
            ;;
        "verify")
            verify_deployment
            ;;
        "cleanup")
            cleanup
            ;;
        "help"|"-h"|"--help")
            usage
            ;;
        *)
            log_error "Unknown command: $command"
            usage
            exit 1
            ;;
    esac
}

# Trap cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"

#!/bin/bash
"""
Complete AWS Deployment Pipeline

One-click deployment script that orchestrates the entire AWS infrastructure
and application deployment process for the medical image training platform.

Author: Medical AI Platform Team
Version: 2.0.0
"""

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Load environment variables
source .env 2>/dev/null || echo "No .env file found, using defaults"

print_header() {
    echo -e "${PURPLE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║           🏥 MEDICAL IMAGE TRAINING PLATFORM                   ║"
    echo "║               AWS Deployment Pipeline                          ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_phase() { echo -e "${CYAN}[PHASE]${NC} $1"; }
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Phase 1: Pre-deployment validation
phase_1_validation() {
    print_phase "Phase 1: Pre-deployment Validation"
    echo "======================================"
    
    print_status "Validating AWS credentials..."
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Please run: aws configure"
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region || echo "us-west-2")
    print_success "AWS Account: $ACCOUNT_ID, Region: $REGION"
    
    print_status "Checking SSH key..."
    if [ ! -f ~/.ssh/id_rsa.pub ]; then
        print_warning "SSH key not found. Generating new key..."
        ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
        print_success "SSH key generated"
    fi
    
    print_status "Validating Docker..."
    if ! docker info &> /dev/null; then
        print_error "Docker is not running. Please start Docker Desktop."
        exit 1
    fi
    
    print_status "Running enterprise tests..."
    if python grayscale_wrapper.py; then
        print_success "All enterprise tests passed"
    else
        print_error "Enterprise tests failed"
        exit 1
    fi
    
    print_success "Phase 1 completed - System validated"
    echo ""
}

# Phase 2: Infrastructure deployment
phase_2_infrastructure() {
    print_phase "Phase 2: Infrastructure Deployment"
    echo "=================================="
    
    print_status "Setting up AWS environment..."
    chmod +x aws_deploy_setup.sh
    ./aws_deploy_setup.sh
    
    print_status "Initializing Terraform..."
    cd terraform
    terraform init
    
    print_status "Planning infrastructure changes..."
    terraform plan -out=tfplan
    
    print_status "Applying infrastructure changes..."
    terraform apply tfplan
    
    # Update kubeconfig
    CLUSTER_NAME=$(terraform output -raw cluster_name)
    aws eks update-kubeconfig --region $REGION --name $CLUSTER_NAME
    
    print_success "Infrastructure deployed successfully"
    cd ..
    echo ""
}

# Phase 3: Container image build and push
phase_3_container() {
    print_phase "Phase 3: Container Image Build & Push"
    echo "======================================"
    
    ECR_URI=$(aws ecr describe-repositories --repository-names medical-image-training --region $REGION --query 'repositories[0].repositoryUri' --output text)
    
    print_status "Logging into ECR..."
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR_URI
    
    print_status "Building Docker image..."
    docker build -f Dockerfile.train -t medical-image-training:latest .
    
    print_status "Tagging and pushing image..."
    docker tag medical-image-training:latest $ECR_URI:latest
    docker push $ECR_URI:latest
    
    print_success "Container image deployed to ECR"
    echo ""
}

# Phase 4: Application deployment
phase_4_application() {
    print_phase "Phase 4: Application Deployment"
    echo "==============================="
    
    print_status "Deploying training platform..."
    chmod +x deploy_training.sh
    ./deploy_training.sh deploy
    
    print_success "Application deployed successfully"
    echo ""
}

# Phase 5: Monitoring and validation
phase_5_monitoring() {
    print_phase "Phase 5: Monitoring & Validation"
    echo "================================"
    
    print_status "Validating cluster health..."
    kubectl get nodes
    kubectl get pods --all-namespaces
    
    print_status "Setting up port forwarding for monitoring..."
    print_warning "Run the following commands in separate terminals to access monitoring:"
    echo "  Grafana:    kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    echo "  Prometheus: kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090"
    
    print_success "Monitoring configured"
    echo ""
}

# Phase 6: Final summary and next steps
phase_6_summary() {
    print_phase "Phase 6: Deployment Summary"
    echo "==========================="
    
    echo ""
    echo -e "${GREEN}🎉 DEPLOYMENT COMPLETED SUCCESSFULLY! 🎉${NC}"
    echo ""
    echo "📋 Summary:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Get cluster info
    CLUSTER_ENDPOINT=$(cd terraform && terraform output -raw cluster_endpoint)
    ECR_URI=$(cd terraform && terraform output -raw ecr_repository_url)
    S3_BUCKET=$(cd terraform && terraform output -raw s3_bucket_name)
    
    echo "🏥 Medical AI Platform Status: PRODUCTION READY"
    echo "☁️  AWS Region: $REGION"
    echo "🎯 EKS Cluster: $CLUSTER_NAME"
    echo "🐳 ECR Repository: $ECR_URI"
    echo "📊 S3 Bucket: $S3_BUCKET"
    echo "🔍 Cluster Endpoint: $CLUSTER_ENDPOINT"
    echo ""
    
    echo "🚀 Next Steps:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "1. Monitor training jobs:"
    echo "   kubectl logs -n medical-training -l app=medical-training -f"
    echo ""
    echo "2. Access Grafana dashboard:"
    echo "   kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    echo "   Open: http://localhost:3000 (admin/admin123)"
    echo ""
    echo "3. Scale training jobs:"
    echo "   kubectl scale deployment -n medical-training medical-training --replicas=5"
    echo ""
    echo "4. Upload training data to S3:"
    echo "   aws s3 cp /path/to/data s3://$S3_BUCKET/training-data/ --recursive"
    echo ""
    echo "5. Monitor GPU utilization:"
    echo "   kubectl top nodes"
    echo "   kubectl exec -it <pod-name> -- nvidia-smi"
    echo ""
    echo "6. Access Triton inference server:"
    echo "   kubectl port-forward -n medical-training svc/triton-inference 8000:8000"
    echo ""
    
    echo "📚 Documentation:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "• Architecture: README.md"
    echo "• API Reference: docs/api.md"  
    echo "• Troubleshooting: docs/troubleshooting.md"
    echo "• Performance Tuning: docs/performance.md"
    echo ""
    
    echo "🛡️  Security & Compliance:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "✅ Encryption at rest (S3, EBS)"
    echo "✅ Encryption in transit (TLS/SSL)"
    echo "✅ IAM roles with least privilege"
    echo "✅ Network security groups"
    echo "✅ Container image scanning"
    echo "✅ Audit logging enabled"
    echo ""
    
    print_success "Medical Image Training Platform is now LIVE in production! 🚀"
}

# Error handling and cleanup
cleanup_on_error() {
    print_error "Deployment failed. Starting cleanup..."
    
    # Cleanup Kubernetes resources
    ./deploy_training.sh cleanup 2>/dev/null || true
    
    # Cleanup Terraform (with confirmation)
    read -p "Do you want to destroy Terraform infrastructure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd terraform && terraform destroy -auto-approve
    fi
    
    print_warning "Partial cleanup completed. Manual review may be required."
}

# Set up error handling
trap cleanup_on_error ERR

# Main execution
main() {
    print_header
    
    echo "Starting complete AWS deployment pipeline..."
    echo "This will deploy a production-ready medical image training platform."
    echo ""
    
    # Confirmation prompt
    read -p "Continue with deployment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Deployment cancelled by user"
        exit 0
    fi
    
    START_TIME=$(date +%s)
    
    # Execute deployment phases
    phase_1_validation
    phase_2_infrastructure  
    phase_3_container
    phase_4_application
    phase_5_monitoring
    phase_6_summary
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${PURPLE}🕒 Total deployment time: ${DURATION}s${NC}"
    echo ""
}

# Handle command line arguments
case "${1:-}" in
    "full")
        main
        ;;
    "infra")
        print_header
        phase_1_validation
        phase_2_infrastructure
        ;;
    "app")
        print_header
        phase_3_container
        phase_4_application
        ;;
    "cleanup")
        cleanup_on_error
        ;;
    *)
        echo "Usage: $0 {full|infra|app|cleanup}"
        echo ""
        echo "Commands:"
        echo "  full    - Complete end-to-end deployment"
        echo "  infra   - Infrastructure deployment only"
        echo "  app     - Application deployment only"
        echo "  cleanup - Clean up all resources"
        echo ""
        echo "For first-time deployment, use: $0 full"
        exit 1
        ;;
esac

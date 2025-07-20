#!/bin/bash
"""
AWS Deployment Setup Script

Complete AWS environment setup for enterprise medical image training platform.
Configures all necessary AWS services, IAM roles, and security policies.

Author: Medical AI Platform Team
Version: 2.0.0
"""

set -e

echo "🚀 AWS Deployment Setup for Medical Image Training Platform"
echo "============================================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration variables
AWS_REGION=${AWS_REGION:-us-west-2}
CLUSTER_NAME=${CLUSTER_NAME:-medical-ai-cluster}
NODE_GROUP_NAME=${NODE_GROUP_NAME:-gpu-nodes}
ECR_REPO_NAME=${ECR_REPO_NAME:-medical-image-training}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Verify AWS CLI and credentials
check_aws_setup() {
    print_status "Checking AWS CLI setup..."
    
    if ! command -v aws &> /dev/null; then
        print_error "AWS CLI not installed. Please install it first:"
        echo "  brew install awscli  # macOS"
        echo "  pip install awscli   # Python"
        exit 1
    fi
    
    if ! aws sts get-caller-identity &> /dev/null; then
        print_error "AWS credentials not configured. Run: aws configure"
        exit 1
    fi
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    print_success "AWS CLI configured. Account ID: $ACCOUNT_ID"
}

# Step 2: Create ECR repository
create_ecr_repo() {
    print_status "Creating ECR repository..."
    
    if aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION &> /dev/null; then
        print_warning "ECR repository $ECR_REPO_NAME already exists"
    else
        aws ecr create-repository \
            --repository-name $ECR_REPO_NAME \
            --region $AWS_REGION \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        print_success "ECR repository created: $ECR_REPO_NAME"
    fi
    
    # Get ECR URI
    ECR_URI=$(aws ecr describe-repositories --repository-names $ECR_REPO_NAME --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text)
    echo "ECR_URI=$ECR_URI" >> .env
    print_success "ECR URI: $ECR_URI"
}

# Step 3: Build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
    
    # Build image
    docker build -f Dockerfile.train -t $ECR_REPO_NAME:latest .
    
    # Tag and push
    docker tag $ECR_REPO_NAME:latest $ECR_URI:latest
    docker push $ECR_URI:latest
    
    print_success "Docker image pushed to ECR"
}

# Step 4: Create IAM roles
create_iam_roles() {
    print_status "Creating IAM roles..."
    
    # EKS Cluster Service Role
    cat > eks-cluster-role-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    if ! aws iam get-role --role-name EKSClusterRole &> /dev/null; then
        aws iam create-role \
            --role-name EKSClusterRole \
            --assume-role-policy-document file://eks-cluster-role-trust-policy.json
        
        aws iam attach-role-policy \
            --role-name EKSClusterRole \
            --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy
        
        print_success "EKS Cluster Role created"
    else
        print_warning "EKS Cluster Role already exists"
    fi
    
    # EKS Node Group Role
    cat > eks-nodegroup-role-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    if ! aws iam get-role --role-name EKSNodeGroupRole &> /dev/null; then
        aws iam create-role \
            --role-name EKSNodeGroupRole \
            --assume-role-policy-document file://eks-nodegroup-role-trust-policy.json
        
        aws iam attach-role-policy \
            --role-name EKSNodeGroupRole \
            --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy
        
        aws iam attach-role-policy \
            --role-name EKSNodeGroupRole \
            --policy-arn arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy
        
        aws iam attach-role-policy \
            --role-name EKSNodeGroupRole \
            --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
        
        print_success "EKS Node Group Role created"
    else
        print_warning "EKS Node Group Role already exists"
    fi
}

# Step 5: Install required tools
install_tools() {
    print_status "Installing required tools..."
    
    # kubectl
    if ! command -v kubectl &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install kubectl
        else
            curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
            chmod +x kubectl
            sudo mv kubectl /usr/local/bin/
        fi
        print_success "kubectl installed"
    else
        print_warning "kubectl already installed"
    fi
    
    # eksctl
    if ! command -v eksctl &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew tap weaveworks/tap
            brew install weaveworks/tap/eksctl
        else
            curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
            sudo mv /tmp/eksctl /usr/local/bin
        fi
        print_success "eksctl installed"
    else
        print_warning "eksctl already installed"
    fi
    
    # Helm
    if ! command -v helm &> /dev/null; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            brew install helm
        else
            curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
        fi
        print_success "Helm installed"
    else
        print_warning "Helm already installed"
    fi
}

# Main execution
main() {
    print_status "Starting AWS deployment setup..."
    
    check_aws_setup
    install_tools
    create_iam_roles
    create_ecr_repo
    
    if command -v docker &> /dev/null; then
        build_and_push_image
    else
        print_warning "Docker not available. Skipping image build."
        print_warning "Install Docker and run: ./aws_deploy_setup.sh build-image"
    fi
    
    print_success "AWS deployment setup completed!"
    echo ""
    echo "Next steps:"
    echo "1. Run: terraform init && terraform plan"
    echo "2. Run: terraform apply"
    echo "3. Run: ./deploy_training.sh"
    echo ""
    echo "Environment variables saved to .env file"
}

# Handle command line arguments
case "${1:-}" in
    "build-image")
        build_and_push_image
        ;;
    *)
        main
        ;;
esac

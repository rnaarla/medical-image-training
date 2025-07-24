# Medical Image Training Platform - Terraform Configuration
# Enterprise-grade AWS infrastructure for distributed ML training

terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "cluster_name" {
  description = "EKS cluster name"
  type        = string
  default     = "medical-ai-cluster"
}

variable "node_instance_types" {
  description = "EC2 instance types for GPU nodes"
  type        = list(string)
  default     = ["p3.2xlarge", "p3.8xlarge", "g4dn.xlarge"]
}

variable "min_nodes" {
  description = "Minimum number of nodes in cluster"
  type        = number
  default     = 1
}

variable "max_nodes" {
  description = "Maximum number of nodes in cluster"
  type        = number
  default     = 10
}

variable "desired_nodes" {
  description = "Desired number of nodes in cluster"
  type        = number
  default     = 2
}

# Provider configuration
provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Configuration
resource "aws_vpc" "medical_ai_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name        = "medical-ai-vpc"
    Environment = "production"
    Project     = "medical-image-training"
  }
}

resource "aws_internet_gateway" "medical_ai_igw" {
  vpc_id = aws_vpc.medical_ai_vpc.id

  tags = {
    Name = "medical-ai-igw"
  }
}

resource "aws_subnet" "public_subnets" {
  count = 3

  vpc_id                  = aws_vpc.medical_ai_vpc.id
  cidr_block              = "10.0.${count.index + 1}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "medical-ai-public-subnet-${count.index + 1}"
    Type = "public"
  }
}

resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.medical_ai_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.medical_ai_igw.id
  }

  tags = {
    Name = "medical-ai-public-rt"
  }
}

resource "aws_route_table_association" "public_rta" {
  count = length(aws_subnet.public_subnets)

  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

# Security Groups
resource "aws_security_group" "eks_cluster_sg" {
  name_prefix = "medical-ai-eks-cluster-"
  vpc_id      = aws_vpc.medical_ai_vpc.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "medical-ai-eks-cluster-sg"
  }
}

resource "aws_security_group" "eks_nodes_sg" {
  name_prefix = "medical-ai-eks-nodes-"
  vpc_id      = aws_vpc.medical_ai_vpc.id

  ingress {
    description = "Allow nodes to communicate with each other"
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    self        = true
  }

  ingress {
    description = "Allow pods to communicate with the cluster API Server"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "medical-ai-eks-nodes-sg"
  }
}

# IAM Roles
resource "aws_iam_role" "eks_cluster_role" {
  name = "medical-ai-eks-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "medical-ai-eks-cluster-role"
  }
}

resource "aws_iam_role_policy_attachment" "eks_cluster_AmazonEKSClusterPolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role.name
}

resource "aws_iam_role" "eks_nodegroup_role" {
  name = "medical-ai-eks-nodegroup-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "medical-ai-eks-nodegroup-role"
  }
}

resource "aws_iam_role_policy_attachment" "eks_nodegroup_AmazonEKSWorkerNodePolicy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_nodegroup_role.name
}

resource "aws_iam_role_policy_attachment" "eks_nodegroup_AmazonEKS_CNI_Policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_nodegroup_role.name
}

resource "aws_iam_role_policy_attachment" "eks_nodegroup_AmazonEC2ContainerRegistryReadOnly" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_nodegroup_role.name
}

# S3 Bucket for data storage
resource "aws_s3_bucket" "medical_data_bucket" {
  bucket = "medical-ai-data-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "medical-ai-data-bucket"
    Environment = "production"
    Project     = "medical-image-training"
  }
}

resource "aws_s3_bucket_versioning" "medical_data_versioning" {
  bucket = aws_s3_bucket.medical_data_bucket.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "medical_data_encryption" {
  bucket = aws_s3_bucket.medical_data_bucket.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# EKS Cluster
resource "aws_eks_cluster" "medical_ai_cluster" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = "1.27"

  vpc_config {
    subnet_ids              = aws_subnet.public_subnets[*].id
    security_group_ids      = [aws_security_group.eks_cluster_sg.id]
    endpoint_private_access = true
    endpoint_public_access  = true
  }

  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_AmazonEKSClusterPolicy,
  ]

  tags = {
    Name        = var.cluster_name
    Environment = "production"
    Project     = "medical-image-training"
  }
}

# EKS Node Group
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.medical_ai_cluster.name
  node_group_name = "gpu-nodes"
  node_role_arn   = aws_iam_role.eks_nodegroup_role.arn
  subnet_ids      = aws_subnet.public_subnets[*].id
  instance_types  = var.node_instance_types

  scaling_config {
    desired_size = var.desired_nodes
    max_size     = var.max_nodes
    min_size     = var.min_nodes
  }

  # Use conditional key pair reference
  dynamic "remote_access" {
    for_each = fileexists("~/.ssh/id_rsa.pub") ? [1] : []
    content {
      ec2_ssh_key = aws_key_pair.medical_ai_key[0].key_name
    }
  }

  dynamic "remote_access" {
    for_each = fileexists("~/.ssh/id_rsa.pub") ? [] : [1]
    content {
      ec2_ssh_key = aws_key_pair.medical_ai_key_generated[0].key_name
    }
  }

  # Launch template for additional security
  launch_template {
    name    = aws_launch_template.medical_ai_nodes.name
    version = aws_launch_template.medical_ai_nodes.latest_version
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_nodegroup_AmazonEKSWorkerNodePolicy,
    aws_iam_role_policy_attachment.eks_nodegroup_AmazonEKS_CNI_Policy,
    aws_iam_role_policy_attachment.eks_nodegroup_AmazonEC2ContainerRegistryReadOnly,
  ]

  tags = {
    Name        = "medical-ai-gpu-nodes"
    Environment = "production"
    Project     = "medical-image-training"
  }
}

# Launch template for additional security and configuration
resource "aws_launch_template" "medical_ai_nodes" {
  name_prefix   = "medical-ai-nodes-"
  image_id      = data.aws_ami.eks_worker.id
  instance_type = var.node_instance_types[0]

  vpc_security_group_ids = [aws_security_group.eks_nodes_sg.id]

  block_device_mappings {
    device_name = "/dev/xvda"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      encrypted             = true
      delete_on_termination = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 2
    instance_metadata_tags      = "enabled"
  }

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "medical-ai-worker"
      Environment = "production"
      Project     = "medical-image-training"
    }
  }

  user_data = base64encode(templatefile("${path.module}/user_data.sh", {
    cluster_name        = var.cluster_name
    cluster_endpoint    = aws_eks_cluster.medical_ai_cluster.endpoint
    cluster_ca          = aws_eks_cluster.medical_ai_cluster.certificate_authority[0].data
    bootstrap_arguments = "--container-runtime containerd"
  }))

  tags = {
    Name = "medical-ai-nodes-template"
  }
}

# Data source for EKS worker AMI
data "aws_ami" "eks_worker" {
  filter {
    name   = "name"
    values = ["amazon-eks-node-${aws_eks_cluster.medical_ai_cluster.version}-v*"]
  }

  most_recent = true
  owners      = ["602401143452"] # Amazon EKS AMI Account ID
}

# Additional IAM policy for enhanced permissions
resource "aws_iam_role_policy" "eks_nodegroup_additional" {
  name = "medical-ai-nodegroup-additional"
  role = aws_iam_role.eks_nodegroup_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.medical_data_bucket.arn,
          "${aws_s3_bucket.medical_data_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ]
        Resource = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:medical-ai/*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# EC2 Key Pair (conditionally create if public key exists)
resource "aws_key_pair" "medical_ai_key" {
  count      = fileexists("~/.ssh/id_rsa.pub") ? 1 : 0
  key_name   = "medical-ai-key"
  public_key = file("~/.ssh/id_rsa.pub")

  tags = {
    Name = "medical-ai-key"
  }
}

# Generate key pair if none exists
resource "tls_private_key" "medical_ai_key" {
  count     = fileexists("~/.ssh/id_rsa.pub") ? 0 : 1
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "medical_ai_key_generated" {
  count      = fileexists("~/.ssh/id_rsa.pub") ? 0 : 1
  key_name   = "medical-ai-key-generated"
  public_key = tls_private_key.medical_ai_key[0].public_key_openssh

  tags = {
    Name = "medical-ai-key-generated"
  }
}

# Store private key in Secrets Manager
resource "aws_secretsmanager_secret" "medical_ai_private_key" {
  count       = fileexists("~/.ssh/id_rsa.pub") ? 0 : 1
  name        = "medical-ai/ssh-private-key"
  description = "SSH private key for medical AI cluster access"
}

resource "aws_secretsmanager_secret_version" "medical_ai_private_key" {
  count     = fileexists("~/.ssh/id_rsa.pub") ? 0 : 1
  secret_id = aws_secretsmanager_secret.medical_ai_private_key[0].id
  secret_string = jsonencode({
    private_key = tls_private_key.medical_ai_key[0].private_key_pem
    public_key  = tls_private_key.medical_ai_key[0].public_key_openssh
  })
}

# ECR Repository
resource "aws_ecr_repository" "medical_image_training" {
  name                 = "medical-image-training"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = {
    Name        = "medical-image-training"
    Environment = "production"
    Project     = "medical-image-training"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "eks_cluster_logs" {
  name              = "/aws/eks/${var.cluster_name}/cluster"
  retention_in_days = 7

  tags = {
    Name        = "medical-ai-eks-logs"
    Environment = "production"
    Project     = "medical-image-training"
  }
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = aws_eks_cluster.medical_ai_cluster.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ids attached to the cluster control plane"
  value       = aws_eks_cluster.medical_ai_cluster.vpc_config[0].cluster_security_group_id
}

output "cluster_name" {
  description = "EKS cluster name"
  value       = aws_eks_cluster.medical_ai_cluster.name
}

output "cluster_arn" {
  description = "EKS cluster ARN"
  value       = aws_eks_cluster.medical_ai_cluster.arn
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.medical_image_training.repository_url
}

output "s3_bucket_name" {
  description = "S3 bucket name for data storage"
  value       = aws_s3_bucket.medical_data_bucket.bucket
}

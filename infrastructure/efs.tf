# EFS (Elastic File System) for shared model storage across nodes
# This enables model sharing and checkpoint persistence

# EFS File System
resource "aws_efs_file_system" "model_storage" {
  creation_token   = "${var.cluster_name}-model-storage"
  performance_mode = "generalPurpose"
  throughput_mode  = "provisioned"
  
  # High performance settings for ML workloads
  provisioned_throughput_in_mibps = 500
  
  # Encryption at rest
  encrypted = true
  kms_key_id = aws_kms_key.efs.arn
  
  lifecycle_policy {
    transition_to_ia                    = "AFTER_30_DAYS"
    transition_to_primary_storage_class = "AFTER_1_ACCESS"
  }
  
  tags = {
    Name        = "${var.cluster_name}-model-storage"
    Environment = var.environment
    Purpose     = "ML-Models-Checkpoints"
  }
}

# EFS Mount Targets (one per AZ for high availability)
resource "aws_efs_mount_target" "model_storage" {
  count = length(module.vpc.private_subnets)
  
  file_system_id  = aws_efs_file_system.model_storage.id
  subnet_id       = module.vpc.private_subnets[count.index]
  security_groups = [aws_security_group.efs.id]
}

# Security Group for EFS
resource "aws_security_group" "efs" {
  name_prefix = "${var.cluster_name}-efs-"
  vpc_id      = module.vpc.vpc_id
  description = "Security group for EFS mount targets"
  
  ingress {
    description = "NFS from EKS nodes"
    from_port   = 2049
    to_port     = 2049
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-efs-sg"
  }
}

# KMS Key for EFS encryption
resource "aws_kms_key" "efs" {
  description             = "EFS encryption key for ${var.cluster_name}"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = {
    Name = "${var.cluster_name}-efs-key"
  }
}

resource "aws_kms_alias" "efs" {
  name          = "alias/${var.cluster_name}-efs-key"
  target_key_id = aws_kms_key.efs.key_id
}

# EFS CSI Driver IRSA
module "irsa-efs-csi" {
  source = "terraform-aws-modules/iam/aws//modules/iam-assumable-role-with-oidc"
  
  create_role                   = true
  role_name                     = "${var.cluster_name}-efs-csi-driver"
  provider_url                  = module.eks.cluster_oidc_issuer_url
  role_policy_arns              = [aws_iam_policy.efs_csi.arn]
  oidc_fully_qualified_subjects = ["system:serviceaccount:kube-system:efs-csi-controller-sa"]
}

# EFS CSI Driver IAM Policy
resource "aws_iam_policy" "efs_csi" {
  name        = "${var.cluster_name}-EFSCSIDriverPolicy"
  path        = "/"
  description = "IAM policy for EFS CSI driver"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "elasticfilesystem:DescribeAccessPoints",
          "elasticfilesystem:DescribeFileSystems",
          "elasticfilesystem:DescribeMountTargets",
          "ec2:DescribeAvailabilityZones"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "elasticfilesystem:CreateAccessPoint"
        ]
        Resource = "*"
        Condition = {
          StringLike = {
            "aws:RequestedRegion" = var.aws_region
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "elasticfilesystem:DeleteAccessPoint"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:ResourceTag/efs.csi.aws.com/cluster" = "true"
          }
        }
      }
    ]
  })
}

# Install EFS CSI Driver
resource "helm_release" "efs_csi_driver" {
  depends_on = [module.eks, module.irsa-efs-csi]
  
  name       = "aws-efs-csi-driver"
  repository = "https://kubernetes-sigs.github.io/aws-efs-csi-driver/"
  chart      = "aws-efs-csi-driver"
  namespace  = "kube-system"
  version    = "2.4.6"
  
  set {
    name  = "controller.serviceAccount.create"
    value = "true"
  }
  
  set {
    name  = "controller.serviceAccount.name"
    value = "efs-csi-controller-sa"
  }
  
  set {
    name  = "controller.serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = module.irsa-efs-csi.iam_role_arn
  }
  
  set {
    name  = "node.serviceAccount.create"
    value = "false"
  }
  
  set {
    name  = "node.serviceAccount.name"
    value = "efs-csi-node-sa"
  }
}

# Storage Class for dynamic EFS provisioning
resource "kubernetes_storage_class" "efs" {
  depends_on = [helm_release.efs_csi_driver]
  
  metadata {
    name = "efs-sc"
    annotations = {
      "storageclass.kubernetes.io/is-default-class" = "false"
    }
  }
  
  storage_provisioner    = "efs.csi.aws.com"
  allow_volume_expansion = true
  reclaim_policy         = "Retain"
  volume_binding_mode    = "Immediate"
  
  parameters = {
    provisioningMode = "efs-ap"
    fileSystemId     = aws_efs_file_system.model_storage.id
    directoryPerms   = "0755"
    gidRangeStart    = "1000"
    gidRangeEnd      = "2000"
    basePath         = "/dynamic_provisioning"
  }
}

# Persistent Volume for model storage (static provisioning)
resource "kubernetes_persistent_volume" "model_storage" {
  depends_on = [helm_release.efs_csi_driver]
  
  metadata {
    name = "model-storage-pv"
  }
  
  spec {
    capacity = {
      storage = "1000Gi"  # EFS is elastic, this is just for k8s
    }
    
    volume_mode                      = "Filesystem"
    access_modes                     = ["ReadWriteMany"]
    persistent_volume_reclaim_policy = "Retain"
    storage_class_name               = "efs-sc"
    
    persistent_volume_source {
      csi {
        driver        = "efs.csi.aws.com"
        volume_handle = aws_efs_file_system.model_storage.id
        
        volume_attributes = {
          path = "/"
        }
      }
    }
    
    mount_options = [
      "tls",
      "_netdev",
      "fsc"  # Enable local caching
    ]
  }
}

# Persistent Volume Claim for model storage
resource "kubernetes_persistent_volume_claim" "model_storage" {
  depends_on = [kubernetes_persistent_volume.model_storage]
  
  metadata {
    name      = "model-storage-pvc"
    namespace = "default"
  }
  
  spec {
    access_modes       = ["ReadWriteMany"]
    storage_class_name = "efs-sc"
    volume_name        = kubernetes_persistent_volume.model_storage.metadata.0.name
    
    resources {
      requests = {
        storage = "1000Gi"
      }
    }
  }
}

# S3 Bucket for model artifacts and datasets
resource "aws_s3_bucket" "model_artifacts" {
  bucket        = "${var.cluster_name}-model-artifacts-${random_string.bucket_suffix.result}"
  force_destroy = false
  
  tags = {
    Name        = "Model Artifacts"
    Environment = var.environment
    Purpose     = "ML-Training-Artifacts"
  }
}

resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# S3 Bucket versioning
resource "aws_s3_bucket_versioning" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# S3 Bucket encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  rule {
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.s3.arn
      sse_algorithm     = "aws:kms"
    }
    bucket_key_enabled = true
  }
}

# S3 KMS Key
resource "aws_kms_key" "s3" {
  description             = "S3 encryption key for ${var.cluster_name}"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  
  tags = {
    Name = "${var.cluster_name}-s3-key"
  }
}

resource "aws_kms_alias" "s3" {
  name          = "alias/${var.cluster_name}-s3-key"
  target_key_id = aws_kms_key.s3.key_id
}

# S3 Bucket public access block
resource "aws_s3_bucket_public_access_block" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy for cost optimization
resource "aws_s3_bucket_lifecycle_configuration" "model_artifacts" {
  bucket = aws_s3_bucket.model_artifacts.id
  
  rule {
    id     = "model_artifacts_lifecycle"
    status = "Enabled"
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
    
    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

# IRSA for S3 access from pods
module "irsa-s3-access" {
  source = "terraform-aws-modules/iam/aws//modules/iam-assumable-role-with-oidc"
  
  create_role                   = true
  role_name                     = "${var.cluster_name}-s3-access"
  provider_url                  = module.eks.cluster_oidc_issuer_url
  role_policy_arns              = [aws_iam_policy.s3_access.arn]
  oidc_fully_qualified_subjects = ["system:serviceaccount:default:training-service-account"]
}

resource "aws_iam_policy" "s3_access" {
  name        = "${var.cluster_name}-S3AccessPolicy"
  path        = "/"
  description = "IAM policy for S3 access from training pods"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:PutObject",
          "s3:PutObjectAcl",
          "s3:DeleteObject",
          "s3:DeleteObjectVersion"
        ]
        Resource = [
          "${aws_s3_bucket.model_artifacts.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket",
          "s3:GetBucketLocation",
          "s3:ListBucketVersions"
        ]
        Resource = [
          aws_s3_bucket.model_artifacts.arn
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = [
          aws_kms_key.s3.arn
        ]
      }
    ]
  })
}

# Service Account for training pods
resource "kubernetes_service_account" "training" {
  depends_on = [module.eks]
  
  metadata {
    name      = "training-service-account"
    namespace = "default"
    annotations = {
      "eks.amazonaws.com/role-arn" = module.irsa-s3-access.iam_role_arn
    }
  }
}

# Outputs
output "efs_file_system_id" {
  description = "EFS file system ID for model storage"
  value       = aws_efs_file_system.model_storage.id
}

output "efs_file_system_dns_name" {
  description = "EFS file system DNS name"
  value       = aws_efs_file_system.model_storage.dns_name
}

output "s3_bucket_name" {
  description = "S3 bucket name for model artifacts"
  value       = aws_s3_bucket.model_artifacts.id
}

output "s3_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.model_artifacts.arn
}

output "training_service_account_arn" {
  description = "Training service account IAM role ARN"
  value       = module.irsa-s3-access.iam_role_arn
}

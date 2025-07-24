# Production Deployment Guide - Medical Image Training Platform

## 🚀 Production Readiness Checklist

This guide provides a comprehensive checklist and procedures for deploying the medical image training platform to production on AWS.

### ✅ Pre-Deployment Checklist

#### Infrastructure Requirements
- [ ] AWS account with appropriate permissions
- [ ] VPC with public/private subnets across multiple AZs
- [ ] EKS cluster with GPU-enabled nodes (p3.2xlarge, p3.8xlarge)
- [ ] ECR repository for container images
- [ ] S3 bucket with encryption enabled
- [ ] RDS PostgreSQL instance (Multi-AZ for production)
- [ ] EFS for shared model storage
- [ ] Route53 for DNS management

#### Security Requirements
- [ ] IAM roles with least privilege access
- [ ] Security groups properly configured
- [ ] Network ACLs configured
- [ ] WAF rules for API protection
- [ ] SSL/TLS certificates provisioned
- [ ] Secrets Manager for sensitive data
- [ ] KMS keys for encryption
- [ ] VPN or Direct Connect for secure access

#### Monitoring & Logging
- [ ] CloudWatch logs aggregation
- [ ] Prometheus monitoring stack
- [ ] Grafana dashboards configured
- [ ] AlertManager rules set up
- [ ] PagerDuty/Slack integration
- [ ] Audit logging enabled
- [ ] Application performance monitoring (APM)

#### Compliance & Governance
- [ ] HIPAA compliance review completed
- [ ] GDPR compliance verified
- [ ] Data retention policies configured
- [ ] Backup and recovery procedures tested
- [ ] Incident response plan documented
- [ ] Change management process established

## 🏗️ Deployment Procedures

### Phase 1: Infrastructure Deployment (30-45 minutes)

```bash
# 1. Set up AWS credentials and region
export AWS_REGION=us-west-2
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# 2. Deploy infrastructure using Terraform
cd infrastructure/
terraform init
terraform plan -var="aws_region=$AWS_REGION"
terraform apply -auto-approve

# 3. Update kubeconfig
aws eks update-kubeconfig --region $AWS_REGION --name medical-ai-cluster
```

### Phase 2: Application Deployment (15-20 minutes)

```bash
# 1. Build and push production container
docker build -f Dockerfile.production -t medical-image-training:v2.0.0 .
ECR_URI=$(aws ecr describe-repositories --repository-names medical-image-training --query 'repositories[0].repositoryUri' --output text)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
docker tag medical-image-training:v2.0.0 $ECR_URI:v2.0.0
docker push $ECR_URI:v2.0.0

# 2. Create namespace and apply configurations
kubectl create namespace medical-training
kubectl apply -f k8s-production.yaml

# 3. Verify deployment
kubectl get pods -n medical-training
kubectl get services -n medical-training
```

### Phase 3: Monitoring Setup (10-15 minutes)

```bash
# 1. Install monitoring stack
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring --create-namespace

# 2. Configure service monitors
kubectl apply -f monitoring/service-monitors.yaml

# 3. Import Grafana dashboards
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80 &
# Access http://localhost:3000 (admin/prom-operator)
```

### Phase 4: Security Hardening (15-20 minutes)

```bash
# 1. Apply network policies
kubectl apply -f security/network-policies.yaml

# 2. Configure pod security standards
kubectl label namespace medical-training pod-security.kubernetes.io/enforce=restricted

# 3. Set up secrets management
aws secretsmanager create-secret --name medical-ai/database-password --secret-string "your-secure-password"
aws secretsmanager create-secret --name medical-ai/jwt-secret --secret-string "your-jwt-secret"
```

## 📊 Production Monitoring

### Key Metrics to Monitor

#### Application Metrics
- Request rate and latency
- Error rates by endpoint
- Image processing throughput
- Model inference accuracy
- Training job success/failure rates

#### Infrastructure Metrics
- CPU and memory utilization
- GPU utilization and memory
- Network bandwidth and latency
- Storage I/O and capacity
- Pod restart frequency

#### Security Metrics
- Failed authentication attempts
- Unusual API access patterns
- Certificate expiration dates
- Security vulnerability scans

### Grafana Dashboard URLs
```bash
# Grafana (admin/prom-operator)
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80

# Prometheus
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090:9090

# AlertManager  
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-alertmanager 9093:9093
```

### Critical Alerts Configuration

```yaml
groups:
- name: medical-ai-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      
  - alert: ModelInferenceLatencyHigh
    expr: histogram_quantile(0.95, rate(inference_duration_seconds_bucket[5m])) > 2
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Model inference latency is high"
      
  - alert: GPUUtilizationLow
    expr: nvidia_gpu_utilization < 10
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "GPU utilization is unexpectedly low"
```

## 🔧 Operational Procedures

### Scaling Operations

#### Horizontal Scaling
```bash
# Scale API replicas
kubectl scale deployment medical-training-api -n medical-training --replicas=10

# Scale training workers
kubectl scale deployment training-workers -n medical-training --replicas=5

# Auto-scaling is configured via HPA
kubectl get hpa -n medical-training
```

#### Vertical Scaling
```bash
# Update resource limits
kubectl patch deployment medical-training-api -n medical-training -p='{"spec":{"template":{"spec":{"containers":[{"name":"medical-training-api","resources":{"limits":{"memory":"8Gi","cpu":"4000m"}}}]}}}}'
```

### Rolling Updates
```bash
# Update container image
kubectl set image deployment/medical-training-api medical-training-api=$ECR_URI:v2.1.0 -n medical-training

# Monitor rollout
kubectl rollout status deployment/medical-training-api -n medical-training

# Rollback if needed
kubectl rollout undo deployment/medical-training-api -n medical-training
```

### Backup and Recovery

#### Database Backup
```bash
# Automated daily backup using CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: database-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:13
            command:
            - pg_dump
            - -h
            - $(DATABASE_HOST)
            - -U
            - $(DATABASE_USER)
            - $(DATABASE_NAME)
```

#### Model Backup
```bash
# Sync models to S3
aws s3 sync /models s3://medical-ai-models-backup/$(date +%Y%m%d)/ --delete
```

### Disaster Recovery Procedures

#### RTO (Recovery Time Objective): 4 hours
#### RPO (Recovery Point Objective): 1 hour

#### DR Steps:
1. **Assess Impact**: Determine scope of outage
2. **Activate DR Site**: Spin up resources in alternate region
3. **Restore Data**: Restore from latest backups
4. **Update DNS**: Point traffic to DR environment
5. **Validate**: Run production validation tests
6. **Communicate**: Update stakeholders on status

## 🛡️ Security Operations

### Security Scanning
```bash
# Container vulnerability scanning
trivy image $ECR_URI:latest --severity HIGH,CRITICAL

# Kubernetes security audit
kube-bench run --targets node,policies,managedservices

# Network security validation
kubectl auth can-i --list --as=system:serviceaccount:medical-training:default
```

### Access Management
```bash
# Create role for medical staff
kubectl create role medical-reader --verb=get,list,watch --resource=pods,services -n medical-training

# Bind role to users
kubectl create rolebinding medical-staff --role=medical-reader --user=medical-team@company.com -n medical-training
```

### Compliance Auditing
```bash
# Generate compliance report
./scripts/generate-compliance-report.sh

# Audit trail analysis
aws cloudtrail lookup-events --start-time 2023-01-01 --end-time 2023-12-31
```

## 📋 Troubleshooting Guide

### Common Issues

#### Application Won't Start
```bash
# Check pod status
kubectl describe pod -n medical-training -l app=medical-training-api

# Check logs
kubectl logs -n medical-training -l app=medical-training-api --tail=100

# Check resource constraints
kubectl top pods -n medical-training
```

#### High Memory Usage
```bash
# Check memory metrics
kubectl top pods -n medical-training

# Analyze heap dumps (if available)
kubectl exec -it pod-name -n medical-training -- jcmd <PID> GC.run_finalization
```

#### Database Connectivity Issues
```bash
# Test database connection
kubectl run db-test --rm -it --image=postgres:13 -- psql -h $DB_HOST -U $DB_USER -d $DB_NAME

# Check network policies
kubectl describe networkpolicy -n medical-training
```

### Performance Issues

#### High Latency
1. Check resource utilization
2. Analyze application metrics
3. Review database query performance
4. Validate network connectivity
5. Scale resources if needed

#### Low Throughput
1. Check GPU utilization
2. Analyze batch processing efficiency
3. Review I/O bottlenecks
4. Optimize model inference
5. Scale horizontally

## 📞 Support Contacts

### Escalation Matrix
- **L1 Support**: Operations team (24/7)
- **L2 Support**: Development team (business hours)
- **L3 Support**: Platform architects (on-call)
- **Executive**: VP Engineering (critical issues)

### Emergency Procedures
1. **Critical Issues**: Page on-call engineer immediately
2. **Security Incidents**: Activate security incident response team
3. **Data Breaches**: Contact legal and compliance teams
4. **System Outages**: Activate business continuity plan

---

## 🎯 Success Metrics

### SLA Targets
- **Availability**: 99.9% uptime
- **Latency**: < 200ms for API calls
- **Throughput**: > 1000 images/minute processing
- **Recovery**: < 4 hour RTO, < 1 hour RPO

### Business Metrics
- **Training Accuracy**: > 95% for medical classifications
- **Processing Speed**: 10x faster than baseline
- **Cost Efficiency**: 60% reduction in infrastructure costs
- **User Satisfaction**: > 4.5/5 rating from medical professionals

This production deployment guide ensures that your medical image training platform is deployed with enterprise-grade reliability, security, and operational excellence.
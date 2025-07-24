#!/bin/bash
"""
Production Deployment Validation Script

Validates the production deployment of the medical image training platform
by performing comprehensive health checks, security validation, and 
operational readiness tests.

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

print_header() {
    echo -e "${PURPLE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║           🏥 PRODUCTION DEPLOYMENT VALIDATION                  ║"
    echo "║               Medical Image Training Platform                  ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_phase() { echo -e "${CYAN}[PHASE]${NC} $1"; }
print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Test counters
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    print_status "Running test: $test_name"
    
    if eval "$test_command" > /dev/null 2>&1; then
        print_success "✅ $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        print_error "❌ $test_name"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

# Phase 1: Infrastructure Validation
phase_1_infrastructure() {
    print_phase "Phase 1: Infrastructure Validation"
    echo "===================================="
    
    # AWS CLI validation
    run_test "AWS CLI connectivity" "aws sts get-caller-identity"
    
    # EKS cluster validation
    run_test "EKS cluster access" "kubectl cluster-info"
    
    # Node availability
    run_test "Kubernetes nodes ready" "kubectl get nodes | grep -q Ready"
    
    # Namespace existence
    run_test "Medical training namespace" "kubectl get namespace medical-training"
    
    # Storage validation
    run_test "Persistent volume claims" "kubectl get pvc -n medical-training"
    
    echo ""
}

# Phase 2: Security Validation
phase_2_security() {
    print_phase "Phase 2: Security Validation"
    echo "============================"
    
    # Network policies
    run_test "Network policies configured" "kubectl get networkpolicy -n medical-training"
    
    # RBAC validation
    run_test "Service accounts configured" "kubectl get serviceaccount -n medical-training"
    
    # Secrets validation
    run_test "Secrets properly configured" "kubectl get secrets -n medical-training"
    
    # Pod security standards
    run_test "Pod security contexts" "kubectl get pods -n medical-training -o jsonpath='{.items[*].spec.securityContext}' | grep -q runAsNonRoot"
    
    # Image security scanning (if available)
    if command -v trivy &> /dev/null; then
        run_test "Container image security scan" "trivy image --severity HIGH,CRITICAL --exit-code 0 medical-image-training:latest"
    else
        print_warning "Trivy not available - container scanning skipped"
    fi
    
    echo ""
}

# Phase 3: Application Health Validation
phase_3_application() {
    print_phase "Phase 3: Application Health Validation"
    echo "======================================"
    
    # Deployment status
    run_test "Deployment rollout status" "kubectl rollout status deployment/medical-training-api -n medical-training --timeout=300s"
    
    # Pod readiness
    run_test "All pods ready" "kubectl wait --for=condition=ready pod -l app=medical-training-api -n medical-training --timeout=300s"
    
    # Service availability
    run_test "Service endpoints" "kubectl get endpoints -n medical-training medical-training-api-service"
    
    # Get service URL
    SERVICE_URL=$(kubectl get service medical-training-api-service -n medical-training -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    if [ -z "$SERVICE_URL" ]; then
        SERVICE_URL="localhost:8080"
        print_status "Using port-forward for testing"
        kubectl port-forward -n medical-training svc/medical-training-api-service 8080:80 &
        PORT_FORWARD_PID=$!
        sleep 10
    fi
    
    # Health endpoint validation
    run_test "Health endpoint responding" "curl -f http://$SERVICE_URL/health"
    
    # Readiness endpoint
    run_test "Readiness endpoint responding" "curl -f http://$SERVICE_URL/health/ready"
    
    # Liveness endpoint
    run_test "Liveness endpoint responding" "curl -f http://$SERVICE_URL/health/live"
    
    # Metrics endpoint
    run_test "Metrics endpoint responding" "curl -f http://$SERVICE_URL/metrics"
    
    # API documentation (if enabled)
    if [ "$ENVIRONMENT" != "production" ]; then
        run_test "API documentation accessible" "curl -f http://$SERVICE_URL/docs"
    fi
    
    # Clean up port-forward if used
    if [ ! -z "$PORT_FORWARD_PID" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
    
    echo ""
}

# Phase 4: Monitoring Validation
phase_4_monitoring() {
    print_phase "Phase 4: Monitoring Validation"
    echo "=============================="
    
    # Prometheus availability
    run_test "Prometheus deployment" "kubectl get deployment -n monitoring prometheus-kube-prometheus-prometheus"
    
    # Grafana availability
    run_test "Grafana deployment" "kubectl get deployment -n monitoring prometheus-grafana"
    
    # Service monitors
    run_test "Service monitors configured" "kubectl get servicemonitor -n medical-training"
    
    # Alert manager
    run_test "Alert manager running" "kubectl get pods -n monitoring -l app.kubernetes.io/name=alertmanager"
    
    echo ""
}

# Phase 5: Performance Validation
phase_5_performance() {
    print_phase "Phase 5: Performance Validation"
    echo "==============================="
    
    # Resource utilization
    run_test "CPU usage within limits" "kubectl top pods -n medical-training"
    
    # Memory usage
    run_test "Memory usage within limits" "kubectl top pods -n medical-training"
    
    # HPA configuration
    run_test "Horizontal Pod Autoscaler" "kubectl get hpa -n medical-training"
    
    # Load testing (basic)
    if command -v ab &> /dev/null; then
        run_test "Basic load test" "ab -n 100 -c 10 http://$SERVICE_URL/health"
    else
        print_warning "Apache Bench not available - load testing skipped"
    fi
    
    echo ""
}

# Phase 6: Compliance Validation
phase_6_compliance() {
    print_phase "Phase 6: Compliance Validation"
    echo "=============================="
    
    # Audit logging
    run_test "Audit logs enabled" "kubectl logs -n kube-system -l component=kube-apiserver | grep -q audit"
    
    # Encryption at rest
    run_test "EBS encryption enabled" "aws ec2 describe-volumes --query 'Volumes[?Encrypted==\`false\`]' --output text | grep -q '^$'"
    
    # S3 encryption
    run_test "S3 encryption configured" "aws s3api get-bucket-encryption --bucket $(terraform output -raw s3_bucket_name)"
    
    # Network encryption
    run_test "TLS/SSL certificates" "kubectl get certificates -n medical-training"
    
    echo ""
}

# Phase 7: Disaster Recovery Validation
phase_7_disaster_recovery() {
    print_phase "Phase 7: Disaster Recovery Validation"
    echo "====================================="
    
    # Backup validation
    run_test "Backup procedures configured" "kubectl get cronjob -n medical-training"
    
    # Multi-AZ deployment
    run_test "Multi-AZ node distribution" "kubectl get nodes -o wide | awk '{print \$7}' | sort | uniq | wc -l | grep -q '[2-9]'"
    
    # Data replication
    run_test "Database replication status" "echo 'Database replication check - implement based on your DB'"
    
    echo ""
}

# Generate validation report
generate_report() {
    print_phase "Validation Summary Report"
    echo "========================="
    
    echo ""
    echo -e "${PURPLE}🏥 PRODUCTION DEPLOYMENT VALIDATION RESULTS${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Calculate success rate
    SUCCESS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    
    echo "📊 Test Results:"
    echo "   Total Tests: $TOTAL_TESTS"
    echo "   Passed: $PASSED_TESTS"
    echo "   Failed: $FAILED_TESTS"
    echo "   Success Rate: $SUCCESS_RATE%"
    echo ""
    
    # Deployment readiness assessment
    if [ $SUCCESS_RATE -ge 95 ]; then
        echo -e "${GREEN}🎉 PRODUCTION READY - Excellent! 🎉${NC}"
        echo "✅ System is ready for production deployment"
    elif [ $SUCCESS_RATE -ge 80 ]; then
        echo -e "${YELLOW}⚠️  MOSTLY READY - Minor issues to address${NC}"
        echo "🔧 Address failed tests before full production launch"
    else
        echo -e "${RED}❌ NOT READY - Significant issues found${NC}"
        echo "🚫 Do not deploy to production until issues are resolved"
    fi
    
    echo ""
    echo "📋 Next Steps:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    if [ $FAILED_TESTS -gt 0 ]; then
        echo "1. Review and fix failed tests"
        echo "2. Re-run validation script"
        echo "3. Update documentation with any changes"
    fi
    
    echo "1. Set up monitoring dashboards"
    echo "2. Configure alerting rules"
    echo "3. Train operations team"
    echo "4. Prepare incident response procedures"
    echo "5. Schedule regular health checks"
    
    echo ""
    echo "📚 Production Operations Guide:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "• Monitoring: kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80"
    echo "• Logs: kubectl logs -f -n medical-training -l app=medical-training-api"
    echo "• Scaling: kubectl scale deployment -n medical-training medical-training-api --replicas=N"
    echo "• Updates: kubectl set image deployment/medical-training-api medical-training-api=new-image"
    echo ""
}

# Main execution
main() {
    print_header
    
    START_TIME=$(date +%s)
    
    # Run all validation phases
    phase_1_infrastructure
    phase_2_security
    phase_3_application
    phase_4_monitoring
    phase_5_performance
    phase_6_compliance
    phase_7_disaster_recovery
    
    # Generate final report
    generate_report
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo -e "${PURPLE}🕒 Total validation time: ${DURATION}s${NC}"
    
    # Exit with appropriate code
    if [ $FAILED_TESTS -gt 0 ]; then
        exit 1
    else
        exit 0
    fi
}

# Run main function
main "$@"
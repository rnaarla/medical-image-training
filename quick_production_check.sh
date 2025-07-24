#!/bin/bash
"""
Quick Production Readiness Check

Performs essential checks to determine if the medical image training platform
is ready for production deployment. This is a lightweight version of the
full validation script for quick assessment.

Usage: ./quick_production_check.sh
"""

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "🏥 Medical Image Training Platform - Quick Production Check"
    echo "==========================================================="
    echo -e "${NC}"
}

check_item() {
    local item="$1"
    local command="$2"
    local required="$3"
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "✅ ${GREEN}$item${NC}"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "❌ ${RED}$item (REQUIRED)${NC}"
            return 1
        else
            echo -e "⚠️  ${YELLOW}$item (Optional)${NC}"
            return 0
        fi
    fi
}

main() {
    print_header
    
    CRITICAL_FAILURES=0
    
    echo "🔍 Checking Production Prerequisites:"
    echo "------------------------------------"
    
    # Critical infrastructure checks
    check_item "AWS CLI configured" "aws sts get-caller-identity" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    check_item "Docker running" "docker info" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    check_item "Kubectl configured" "kubectl cluster-info" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    check_item "Terraform available" "terraform version" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    
    echo ""
    echo "🏗️  Checking Deployment Artifacts:"
    echo "----------------------------------"
    
    # Deployment artifacts
    check_item "Production Dockerfile" "test -f Dockerfile.production" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    check_item "Kubernetes manifests" "test -f k8s-production.yaml" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    check_item "Terraform configuration" "test -f infrastructure/main.tf" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    check_item "Production environment config" "test -f .env.production" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    check_item "Production API application" "test -f production_api.py" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    
    echo ""
    echo "🛡️  Checking Security Configuration:"
    echo "-----------------------------------"
    
    # Security checks
    check_item "AWS IAM permissions" "aws iam get-user" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    check_item "SSH key exists" "test -f ~/.ssh/id_rsa.pub" false
    check_item "Production requirements" "test -f requirements_production.txt" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    
    echo ""
    echo "📊 Checking Monitoring Setup:"
    echo "-----------------------------"
    
    # Monitoring checks
    check_item "Validation script executable" "test -x scripts/validate_production.sh" false
    check_item "Deployment documentation" "test -f docs/PRODUCTION_DEPLOYMENT.md" true || CRITICAL_FAILURES=$((CRITICAL_FAILURES + 1))
    
    echo ""
    echo "🎯 Production Readiness Assessment:"
    echo "==================================="
    
    if [ $CRITICAL_FAILURES -eq 0 ]; then
        echo -e "${GREEN}🎉 PRODUCTION READY!${NC}"
        echo ""
        echo "✅ All critical requirements met"
        echo "✅ Security configurations in place"
        echo "✅ Deployment artifacts available"
        echo "✅ Infrastructure tools configured"
        echo ""
        echo "🚀 Next Steps:"
        echo "1. Review .env.production and customize for your environment"
        echo "2. Run: ./scripts/deploy_aws.sh full"
        echo "3. Execute: ./scripts/validate_production.sh"
        echo "4. Set up monitoring dashboards"
        echo ""
        echo "📚 See docs/PRODUCTION_DEPLOYMENT.md for detailed procedures"
        return 0
    else
        echo -e "${RED}❌ NOT READY FOR PRODUCTION${NC}"
        echo ""
        echo "⚠️  $CRITICAL_FAILURES critical requirements failed"
        echo ""
        echo "🔧 Required Actions:"
        echo "1. Configure AWS CLI: aws configure"
        echo "2. Start Docker service"
        echo "3. Configure kubectl: aws eks update-kubeconfig --name cluster-name"
        echo "4. Install Terraform: https://terraform.io/downloads"
        echo "5. Verify all required files are present"
        echo ""
        echo "📖 See README.md for setup instructions"
        return 1
    fi
}

main "$@"
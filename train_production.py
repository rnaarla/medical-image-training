#!/usr/bin/env python3
"""
Production Training Script
Optimized for enterprise deployment with monitoring and error handling
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Production training entrypoint with error handling"""
    try:
        # Import after logging setup
        from train import main as train_main, TrainingConfig
        
        logger.info("Starting medical image training in production mode")
        
        # Validate environment
        required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'WANDB_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.warning(f"Missing environment variables: {missing_vars}")
            logger.info("Continuing with available configuration...")
        
        # Run training
        train_main()
        
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == '__main__':
    main()

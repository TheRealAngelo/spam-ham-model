#!/usr/bin/env python3
"""
Deployment script for SMS Spam Detector
This script handles training, testing, and deployment of the application
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f" {description} completed successfully!")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_dependencies():
    """Check if required files exist"""
    required_files = [
        'spam.csv',
        'train.py',
        'main.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return False
    
    print(" All required files found!")
    return True

def install_dependencies():
    """Install required Python packages"""
    if os.path.exists('requirements.txt'):
        return run_command(
            "pip install -r requirements.txt",
            "Installing dependencies"
        )
    else:
        print(" requirements.txt not found!")
        return False

def train_model():
    """Train the spam detection model"""
    return run_command(
        "python train.py",
        "Training the spam detection model"
    )

def test_model():
    """Test the trained model"""
    return run_command(
        "python utils.py",
        "Testing the trained model"
    )

def deploy_streamlit():
    """Deploy the Streamlit application"""
    print("\n Starting Streamlit application...")
    print("The application will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")
    
    try:
        subprocess.run("streamlit run main.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n Application stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f" Failed to start Streamlit: {e}")

def main():
    parser = argparse.ArgumentParser(description='Deploy SMS Spam Detector')
    parser.add_argument('--skip-install', action='store_true', 
                       help='Skip dependency installation')
    parser.add_argument('--skip-train', action='store_true', 
                       help='Skip model training')
    parser.add_argument('--skip-test', action='store_true', 
                       help='Skip model testing')
    parser.add_argument('--only-deploy', action='store_true', 
                       help='Only deploy the application (skip all other steps)')
    
    args = parser.parse_args()
    
    print(" SMS Spam Detector Deployment Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    if args.only_deploy:
        deploy_streamlit()
        return
    
    # Install dependencies
    if not args.skip_install:
        if not install_dependencies():
            print(" Dependency installation failed. Exiting.")
            return
    
    # Train model
    if not args.skip_train:
        if not train_model():
            print(" Model training failed. Exiting.")
            return
    
    # Test model
    if not args.skip_test:
        if not test_model():
            print(" Model testing failed, but continuing with deployment.")
    
    # Deploy application
    deploy_streamlit()

if __name__ == "__main__":
    main()

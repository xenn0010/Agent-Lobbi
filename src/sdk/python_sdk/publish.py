#!/usr/bin/env python3
"""
PyPI Publication Script for Agent Lobbi SDK

This script helps publish the Agent Lobbi Python SDK to PyPI.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\nRELOAD {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"OK {description} completed successfully!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main publication workflow."""
    print("START Agent Lobbi SDK Publication Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("ERROR Error: pyproject.toml not found. Please run this script from the SDK directory.")
        sys.exit(1)
    
    # Step 1: Clean previous builds
    print("\n1️⃣ Cleaning previous builds...")
    if Path("dist").exists():
        run_command("rm -rf dist", "Removing old dist directory")
    if Path("build").exists():
        run_command("rm -rf build", "Removing old build directory")
    
    # Step 2: Install/upgrade build tools
    if not run_command("pip install --upgrade build twine", "Installing/upgrading build tools"):
        sys.exit(1)
    
    # Step 3: Run tests
    print("\n2️⃣ Running tests...")
    if not run_command("python -m pytest tests/ -v", "Running test suite"):
        print("WARNING  Tests failed, but continuing with build...")
    
    # Step 4: Build package
    if not run_command("python -m build", "Building package"):
        sys.exit(1)
    
    # Step 5: Check package
    if not run_command("python -m twine check dist/*", "Checking package"):
        sys.exit(1)
    
    # Step 6: Show built files
    print("\n3️⃣ Built files:")
    run_command("ls -la dist/", "Listing distribution files")
    
    # Step 7: Upload options
    print("\n4️⃣ Upload Options:")
    print("Choose your upload destination:")
    print("1. TestPyPI (recommended for testing)")
    print("2. PyPI (production)")
    print("3. Skip upload (just build)")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            print("\nTEST Uploading to TestPyPI...")
            cmd = "python -m twine upload --repository testpypi dist/*"
            if run_command(cmd, "Uploading to TestPyPI"):
                print("\nOK Successfully uploaded to TestPyPI!")
                print("🔗 Check your package at: https://test.pypi.org/project/agent-lobbi-sdk/")
                print("INSTALL Test installation: pip install -i https://test.pypi.org/simple/ agent-lobbi-sdk")
            break
            
        elif choice == "2":
            print("\nSTART Uploading to PyPI...")
            confirmation = input("WARNING  This will publish to production PyPI. Are you sure? (yes/no): ")
            if confirmation.lower() == "yes":
                cmd = "python -m twine upload dist/*"
                if run_command(cmd, "Uploading to PyPI"):
                    print("\nSUCCESS Successfully published to PyPI!")
                    print("🔗 Check your package at: https://pypi.org/project/agent-lobbi-sdk/")
                    print("INSTALL Install with: pip install agent-lobbi-sdk")
            else:
                print("ERROR Upload cancelled.")
            break
            
        elif choice == "3":
            print("\nOK Build completed. Upload skipped.")
            break
            
        else:
            print("ERROR Invalid choice. Please enter 1, 2, or 3.")
    
    print("\n🏁 Publication script completed!")

if __name__ == "__main__":
    main() 
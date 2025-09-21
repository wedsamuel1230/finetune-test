#!/usr/bin/env python3
"""
Validation script for Google Colab notebook enhancements.
This script checks that all the key Colab features are properly implemented.
"""

import json
import os
import re

def validate_notebook():
    """Validate the notebook has all required Google Colab features."""
    
    print("🔍 Validating Google Colab Notebook Enhancements")
    print("=" * 60)
    
    notebook_path = "notebooks/t5_book_qa_training.ipynb"
    
    if not os.path.exists(notebook_path):
        print("❌ Notebook not found!")
        return False
    
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    validation_results = {}
    
    # Check 1: Environment detection
    env_detection_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            source = ''.join(cell['source'])
            if 'import google.colab' in source and 'IN_COLAB' in source:
                env_detection_found = True
                break
    
    validation_results['Environment Detection'] = env_detection_found
    
    # Check 2: Repository cloning
    repo_clone_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            for line in cell['source']:
                if ('git clone' in line and 
                    'wedsamuel1230/finetune-test' in line and 
                    not line.strip().startswith('#')):  # Not commented out
                    repo_clone_found = True
                    break
            if repo_clone_found:
                break
    
    validation_results['Repository Cloning'] = repo_clone_found
    
    # Check 3: Google Drive integration
    drive_integration_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            source = ''.join(cell['source'])
            if 'drive.mount' in source or '/content/drive' in source:
                drive_integration_found = True
                break
    
    validation_results['Google Drive Integration'] = drive_integration_found
    
    # Check 4: GPU optimization
    gpu_optimization_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            source = ''.join(cell['source'])
            if ('torch.cuda.is_available()' in source and 
                ('batch_size' in source or 'gradient_accumulation' in source)):
                gpu_optimization_found = True
                break
    
    validation_results['GPU Optimization'] = gpu_optimization_found
    
    # Check 5: File upload capability
    file_upload_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            source = ''.join(cell['source'])
            if 'files.upload()' in source:
                file_upload_found = True
                break
    
    validation_results['File Upload Capability'] = file_upload_found
    
    # Check 6: Memory optimization
    memory_optimization_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            source = ''.join(cell['source'])
            if ('PYTORCH_CUDA_ALLOC_CONF' in source or 
                'torch.cuda.empty_cache()' in source or
                'dataloader_num_workers=0' in source):
                memory_optimization_found = True
                break
    
    validation_results['Memory Optimization'] = memory_optimization_found
    
    # Check 7: Error handling
    error_handling_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            source = ''.join(cell['source'])
            if 'try:' in source and 'except' in source and 'load_book_dataset' in source:
                error_handling_found = True
                break
    
    validation_results['Error Handling'] = error_handling_found
    
    # Check 8: Model download feature
    model_download_found = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and cell.get('source'):
            source = ''.join(cell['source'])
            if 'files.download' in source and '.zip' in source:
                model_download_found = True
                break
    
    validation_results['Model Download'] = model_download_found
    
    # Print results
    print("\n📊 Validation Results:")
    print("-" * 40)
    
    all_passed = True
    for feature, passed in validation_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{feature:<30} {status}")
        if not passed:
            all_passed = False
    
    print("-" * 40)
    
    if all_passed:
        print("🎉 All validations passed! Notebook is ready for Google Colab.")
        print("\n📋 Summary of enhancements:")
        print("• Automatic environment detection (Colab vs local)")
        print("• Repository cloning with proper setup")
        print("• Google Drive integration for model persistence")
        print("• T4 GPU optimization and memory management")
        print("• File upload support for custom datasets")
        print("• Robust error handling with fallbacks")
        print("• Easy model download functionality")
        return True
    else:
        print("⚠️  Some validations failed. Please check the notebook.")
        return False

if __name__ == "__main__":
    success = validate_notebook()
    exit(0 if success else 1)
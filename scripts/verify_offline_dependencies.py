import sys
import importlib

def check_package(package_name):
    try:
        mod = importlib.import_module(package_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"[OK] {package_name} is available (version: {version})")
        return True
    except ImportError as e:
        print(f"[FAIL] {package_name} is NOT available: {e}")
        return False

def main():
    print("--- OFFLINE DEPENDENCY VERIFICATION ---")
    
    # Critical system packages
    packages = [
        'torch',
        'torchvision',
        'timm',
        'numpy',
        'pandas',
        'cv2',
        'yaml',
        'tqdm',
        'matplotlib'
    ]
    
    all_ok = True
    for pkg in packages:
        if not check_package(pkg):
            all_ok = False
            
    print("\nChecking segmentation_models_pytorch...")
    try:
        import segmentation_models_pytorch as smp
        print(f"[OK] segmentation_models_pytorch is available via system (version: {smp.__version__})")
    except ImportError:
        import os
        vendor_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vendor'))
        print(f"[INFO] segmentation_models_pytorch not found in system. Checking vendored path: {vendor_path}")
        if vendor_path not in sys.path:
            sys.path.insert(0, vendor_path)
        
        try:
            import segmentation_models_pytorch as smp
            print(f"[OK] segmentation_models_pytorch is available via VENDOR (version: {getattr(smp, '__version__', 'unknown')})")
        except ImportError as e:
            print(f"[FAIL] segmentation_models_pytorch could not be loaded from vendor: {e}")
            all_ok = False
            
    print("\n--- VERIFICATION RESULT ---")
    if all_ok:
        print("ALL REQUIRED DEPENDENCIES ARE AVAILABLE OFFLINE.")
        sys.exit(0)
    else:
        print("SOME DEPENDENCIES ARE MISSING.")
        sys.exit(1)

if __name__ == '__main__':
    main()

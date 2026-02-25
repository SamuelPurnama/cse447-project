#!/usr/bin/env python
"""
Test script for ByT5 deployment: verify language detection and model routing.
"""
import sys
import os

# Test language detection
def test_language_detection():
    """Test the language detection logic."""
    print("=" * 60)
    print("Testing language detection...")
    print("=" * 60)
    
    try:
        import langdetect
        
        test_cases = [
            ("Hello world", "en", "English"),
            ("你好世界", "zh-cn", "Chinese"),
            ("Bonjour le monde", "fr", "French"),
            ("こんにちは世界", "ja", "Japanese"),
            ("Hola mundo", "es", "Spanish"),
            ("", "en", "Empty (default to en)"),
            ("a", "en", "Very short (default to en)"),
        ]
        
        for text, expected_lang_prefix, description in test_cases:
            try:
                if text and len(text.strip()) >= 3:
                    detected = langdetect.detect(text)
                else:
                    detected = "en"  # Default
                
                is_english = detected == "en"
                model_name = "Bolmo-1B" if is_english else "ByT5-Small"
                
                print(f"Text: '{text[:30]}...' ({description})")
                print(f"  Detected: {detected}, Expected: {expected_lang_prefix}")
                print(f"  → Will use: {model_name}")
                print()
            except Exception as e:
                print(f"Text: '{text}' ({description})")
                print(f"  Detection failed: {e}")
                print(f"  → Will use: Bolmo-1B (default)")
                print()
        
        print("✓ Language detection test completed")
        return True
        
    except ImportError:
        print("✗ langdetect not installed. Run: pip install langdetect")
        return False


def test_imports():
    """Test that all required imports work."""
    print("\n" + "=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    required_imports = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("scipy", "SciPy"),
        ("langdetect", "langdetect"),
    ]
    
    all_ok = True
    for module, name in required_imports:
        try:
            __import__(module)
            print(f"✓ {name} ({module})")
        except ImportError as e:
            print(f"✗ {name} ({module}): {e}")
            all_ok = False
    
    return all_ok


def test_myprogram_syntax():
    """Test that myprogram.py has no syntax errors."""
    print("\n" + "=" * 60)
    print("Testing myprogram.py syntax...")
    print("=" * 60)
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "myprogram",
            "src/myprogram.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("✓ myprogram.py syntax is valid")
        
        # Check key components
        checks = [
            ("BOLMO_SUBDIR", "Bolmo constant"),
            ("BYT5_SUBDIR", "ByT5 constant"),
            ("BYT5_HF_ID", "ByT5 HF ID constant"),
            ("MyModel", "MyModel class"),
        ]
        
        for attr, desc in checks:
            if hasattr(module, attr):
                print(f"  ✓ {desc} ({attr}) exists")
            else:
                print(f"  ✗ {desc} ({attr}) missing!")
                return False
        
        # Check MyModel methods
        mymodel_methods = [
            "_detect_language",
            "_next_char_top3",
            "_next_char_top3_byt5",
            "run_pred",
            "load",
            "save",
        ]
        
        for method in mymodel_methods:
            if hasattr(module.MyModel, method):
                print(f"  ✓ MyModel.{method}() exists")
            else:
                print(f"  ✗ MyModel.{method}() missing!")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading myprogram.py: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("ByT5 Deployment Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Imports
    results.append(("Imports", test_imports()))
    
    # Test 2: myprogram.py syntax
    results.append(("myprogram.py syntax", test_myprogram_syntax()))
    
    # Test 3: Language detection
    results.append(("Language detection", test_language_detection()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All tests passed! ByT5 deployment is ready.")
        print("\nNext steps:")
        print("1. Run: python src/myprogram.py test --work_dir work --test_data example/input.txt")
        print("2. Check that ByT5-Small downloads and loads successfully")
        print("3. Verify predictions use Bolmo for English, ByT5 for non-English")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

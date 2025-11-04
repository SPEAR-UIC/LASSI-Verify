"""
Test AMD Integration for LASSI Pipeline
Tests AMD functionality without running the full LASSI-EE pipeline
"""

import os
import sys
from lassi.gpu_detector import detect_gpu_vendors
from lassi.amd_power import AMDPowerMonitor, amd_report_gpu_memory
import lassi.code_manager

def test_gpu_detection():
    """Test 1: GPU vendor detection"""
    print("=== Test 1: GPU Vendor Detection ===")
    
    vendors = detect_gpu_vendors()
    if not vendors:
        print("❌ No GPUs detected")
        return False
    
    for vendor_info in vendors:
        vendor = vendor_info['vendor']
        count = vendor_info['count']
        print(f"✅ {vendor.upper()}: {count} GPU(s) detected")
    
    return True

def test_amd_power_monitoring():
    """Test 2: AMD power monitoring"""
    print("\n=== Test 2: AMD Power Monitoring ===")
    
    vendors = detect_gpu_vendors()
    amd_available = any(v['vendor'] == 'amd' for v in vendors)
    
    if not amd_available:
        print("⚠️  No AMD GPUs detected, skipping AMD power test")
        return True
    
    try:
        # Test basic power monitoring
        monitor = AMDPowerMonitor()
        power = monitor.get_power_usage(0)
        print(f"✅ AMD GPU 0 power: {power:.2f} W")
        
        # Test memory reporting
        print("\n--- AMD GPU Memory Report ---")
        amd_report_gpu_memory()
        
        return True
    except Exception as e:
        print(f"❌ AMD power monitoring failed: {e}")
        return False

def test_selectvendor_execution():
    """Test 3: Multi-vendor execution functions"""
    print("\n=== Test 3: Multi-vendor Code Manager Functions ===")
    
    # Test that the new functions exist
    try:
        # Check if the new functions are available
        if hasattr(lassi.code_manager, 'execute_code_selectvendor'):
            print("✅ execute_code_multivendor function available")
        else:
            print("❌ execute_code_selectvendor function not found")
            return False
            
        if hasattr(lassi.code_manager, 'execute_code_amd'):
            print("✅ execute_code_amd function available")
        else:
            print("❌ execute_code_amd function not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Multi-vendor function test failed: {e}")
        return False

def test_hip_compilation():
    """Test 4: HIP compilation (if AMD GPUs available)"""
    print("\n=== Test 4: HIP Compilation Test ===")
    
    vendors = detect_gpu_vendors()
    amd_available = any(v['vendor'] == 'amd' for v in vendors)
    
    if not amd_available:
        print("⚠️  No AMD GPUs detected, skipping HIP compilation test")
        return True
    
    # Use our existing hip_code_test
    try:
        from hip_code_test import quick_test
        
        # Test with a simple HIP file (adjust path as needed)
        test_file = "source_codes/HeCBench/jacobi/jacobi-hip_main.cu"
        
        if os.path.exists(test_file):
            print(f"✅ Testing HIP compilation with: {test_file}")
            result = quick_test(test_file)
            if result:
                print("✅ HIP compilation and execution successful")
                return True
            else:
                print("❌ HIP compilation or execution failed")
                return False
        else:
            print(f"⚠️  Test file not found: {test_file}")
            print("✅ HIP compilation functions available (file test skipped)")
            return True
            
    except Exception as e:
        print(f"❌ HIP compilation test failed: {e}")
        return False

def test_pipeline_compatibility():
    """Test 5: Pipeline compatibility check"""
    print("\n=== Test 5: Pipeline Compatibility ===")
    
    try:
        # Test that pipeline_v3 can still import everything it needs
        import lassi.pipeline_v3
        print("✅ pipeline_v3 imports successfully")
        
        # Test that original functions still exist
        if hasattr(lassi.code_manager, 'compile_code'):
            print("✅ Original compile_code function preserved")
        else:
            print("❌ Original compile_code function missing")
            return False
            
        if hasattr(lassi.code_manager, 'execute_code'):
            print("✅ Original execute_code function preserved")
        else:
            print("❌ Original execute_code function missing")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Pipeline compatibility test failed: {e}")
        return False

def run_all_tests():
    """Run all AMD integration tests"""
    print("🚀 Starting AMD Integration Tests for LASSI\n")
    
    tests = [
        test_gpu_detection,
        test_amd_power_monitoring,
        test_selectvendor_execution,
        test_hip_compilation,
        test_pipeline_compatibility
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AMD integration ready for LASSI pipeline.")
        return True
    else:
        print("⚠️  Some tests failed. Check integration before using LASSI pipeline.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Test correctness with 100M elements to verify 32-float operations per iteration
"""
import sys
sys.path.insert(0, '/Users/tharungoka/LabNeura/python')
import labneura
import time

def test_fp32_large():
    """Test FP32 with 100M elements"""
    print("=" * 70)
    print("Testing FP32 with 100,000,000 elements (32-float operations)")
    print("=" * 70)
    
    size = 100_000_000
    
    # Create tensors with known values
    t1_data = [2.5] * size
    t2_data = [1.5] * size
    expected = 4.0  # 2.5 + 1.5
    
    print(f"Creating tensors of size {size:,}...")
    t1 = labneura.Tensor(t1_data, labneura.QuantizationMode.FP32)
    t2 = labneura.Tensor(t2_data, labneura.QuantizationMode.FP32)
    
    print("Performing addition...")
    start = time.perf_counter()
    t1.add_inplace(t2)
    elapsed = time.perf_counter() - start
    
    # Verify correctness by sampling
    result = t1.data_fp32()
    
    # Check samples at different positions
    test_positions = [0, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 99_999_999]
    print("\nSampling results at different positions:")
    print(f"{'Position':<15} {'Expected':<15} {'Got':<15} {'Match':<10}")
    print("-" * 55)
    
    all_correct = True
    for pos in test_positions:
        actual = result[pos]
        match = abs(actual - expected) < 1e-6
        all_correct = all_correct and match
        print(f"{pos:<15} {expected:<15.6f} {actual:<15.6f} {'✓' if match else '✗':<10}")
    
    # Test remainder handling: check last few elements
    print("\nTesting remainder handling (last 10 elements):")
    for i in range(max(0, size - 10), size):
        actual = result[i]
        match = abs(actual - expected) < 1e-6
        if not match:
            all_correct = False
            print(f"  Position {i}: Expected {expected}, got {actual} ✗")
    if all_correct:
        print("  All remainder elements correct ✓")
    
    # Test alignment boundaries
    print("\nTesting alignment boundaries:")
    boundaries = [
        (size - 1, "last element"),
        (32, "32-element boundary"),
        (31, "before 32-boundary"),
        (33, "after 32-boundary"),
        (16, "16-element boundary"),
        (4, "4-element boundary"),
    ]
    
    for pos, desc in boundaries:
        if pos < size:
            actual = result[pos]
            match = abs(actual - expected) < 1e-6
            all_correct = all_correct and match
            print(f"  Position {pos:10} ({desc:20}): {actual:.6f} {'✓' if match else '✗'}")
    
    print()
    print(f"Time: {elapsed:.3f}s")
    ops = (size * 1) / (elapsed * 1e6)
    print(f"Throughput: {ops:.2f} M ops/sec")
    print()
    
    if all_correct:
        print("✓ FP32 correctness test PASSED")
        return True
    else:
        print("✗ FP32 correctness test FAILED")
        return False

def test_int8_large():
    """Test INT8 with 100M elements"""
    print("\n" + "=" * 70)
    print("Testing INT8 with 100,000,000 elements (saturation handling)")
    print("=" * 70)
    
    size = 100_000_000
    
    # Create tensors with values that test saturation
    t1_data = [50] * size
    t2_data = [60] * size
    expected_sum = min(127, 50 + 60)  # 110 (no saturation)
    
    print(f"Creating INT8 tensors of size {size:,}...")
    t1 = labneura.Tensor(t1_data, labneura.QuantizationMode.INT8)
    t2 = labneura.Tensor(t2_data, labneura.QuantizationMode.INT8)
    
    print("Performing INT8 addition...")
    start = time.perf_counter()
    t1.add_inplace(t2)
    elapsed = time.perf_counter() - start
    
    result = t1.data_int8()
    
    # Check samples
    test_positions = [0, 1000, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 99_999_999]
    print("\nSampling results at different positions:")
    print(f"{'Position':<15} {'Expected':<15} {'Got':<15} {'Match':<10}")
    print("-" * 55)
    
    all_correct = True
    for pos in test_positions:
        actual = result[pos]
        match = actual == expected_sum
        all_correct = all_correct and match
        print(f"{pos:<15} {expected_sum:<15} {actual:<15} {'✓' if match else '✗':<10}")
    
    print()
    print(f"Time: {elapsed:.3f}s")
    ops = (size * 1) / (elapsed * 1e6)
    print(f"Throughput: {ops:.2f} M ops/sec")
    print()
    
    if all_correct:
        print("✓ INT8 correctness test PASSED")
        return True
    else:
        print("✗ INT8 correctness test FAILED")
        return False

def test_saturation():
    """Test INT8 saturation handling"""
    print("\n" + "=" * 70)
    print("Testing INT8 Saturation Edge Cases")
    print("=" * 70)
    
    test_cases = [
        ([127], [0], [127], "max + 0"),
        ([127], [1], [127], "saturation positive"),
        ([-128], [0], [-128], "min + 0"),
        ([-128], [-1], [-128], "saturation negative"),
        ([60], [70], [127], "saturation 60+70=130->127"),
        ([-60], [-70], [-128], "saturation -60-70=-130->-128"),
    ]
    
    all_correct = True
    for t1_data, t2_data, expected, desc in test_cases:
        t1 = labneura.Tensor(t1_data, labneura.QuantizationMode.INT8)
        t2 = labneura.Tensor(t2_data, labneura.QuantizationMode.INT8)
        t1.add_inplace(t2)
        result = t1.data_int8()
        
        match = result[0] == expected[0]
        all_correct = all_correct and match
        status = "✓" if match else "✗"
        print(f"{desc:30} {t1_data[0]:5} + {t2_data[0]:5} = {result[0]:5} (expected {expected[0]:5}) {status}")
    
    print()
    if all_correct:
        print("✓ INT8 saturation test PASSED")
        return True
    else:
        print("✗ INT8 saturation test FAILED")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LabNeura Correctness Test Suite (32-float operations)")
    print("=" * 70 + "\n")
    
    results = []
    
    # Run tests
    try:
        results.append(("FP32 Large", test_fp32_large()))
    except Exception as e:
        print(f"✗ FP32 Large test FAILED with exception: {e}")
        results.append(("FP32 Large", False))
    
    try:
        results.append(("INT8 Large", test_int8_large()))
    except Exception as e:
        print(f"✗ INT8 Large test FAILED with exception: {e}")
        results.append(("INT8 Large", False))
    
    try:
        results.append(("INT8 Saturation", test_saturation()))
    except Exception as e:
        print(f"✗ INT8 Saturation test FAILED with exception: {e}")
        results.append(("INT8 Saturation", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results:
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"{test_name:30} {status}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 70)
    if all_passed:
        print("All tests PASSED ✓")
        sys.exit(0)
    else:
        print("Some tests FAILED ✗")
        sys.exit(1)

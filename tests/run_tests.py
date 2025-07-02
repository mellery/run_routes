#!/usr/bin/env python3
"""
Test Runner for Route Services
Runs all unit and integration tests
"""

import unittest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """Run all tests and return results"""
    # Discover and run unit tests
    unit_loader = unittest.TestLoader()
    unit_suite = unit_loader.discover('unit', pattern='test_*.py')
    
    # Discover and run integration tests
    integration_loader = unittest.TestLoader()
    integration_suite = integration_loader.discover('integration', pattern='test_*.py')
    
    # Combine all test suites
    all_tests = unittest.TestSuite([unit_suite, integration_suite])
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(all_tests)
    
    return result

def run_unit_tests_only():
    """Run only unit tests"""
    loader = unittest.TestLoader()
    suite = loader.discover('unit', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result

def run_integration_tests_only():
    """Run only integration tests"""
    loader = unittest.TestLoader()
    suite = loader.discover('integration', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result

def print_test_summary(result):
    """Print test summary"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
        
        if test_type == 'unit':
            print("🧪 Running Unit Tests Only...")
            result = run_unit_tests_only()
        elif test_type == 'integration':
            print("🔗 Running Integration Tests Only...")
            result = run_integration_tests_only()
        elif test_type == 'all':
            print("🚀 Running All Tests...")
            result = run_all_tests()
        else:
            print("Usage: python run_tests.py [unit|integration|all]")
            print("Default: runs all tests")
            return
    else:
        print("🚀 Running All Tests...")
        result = run_all_tests()
    
    print_test_summary(result)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()
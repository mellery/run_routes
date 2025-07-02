#!/usr/bin/env python3
"""
Route Services Test Runner
Runs unit and integration tests for the route services
"""

import unittest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_dependencies():
    """Check if required dependencies are available"""
    missing_deps = []
    
    try:
        import networkx
    except ImportError:
        missing_deps.append('networkx')
    
    try:
        import numpy
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import osmnx
    except ImportError:
        missing_deps.append('osmnx')
    
    return missing_deps


def run_unit_tests():
    """Run unit tests for route services (requires dependencies)"""
    print("🧪 Running Unit Tests (Requires Dependencies)")
    print("=" * 50)
    
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("📋 Install with: pip install networkx osmnx numpy matplotlib pandas")
        return None
    
    try:
        # Discover and run unit tests
        loader = unittest.TestLoader()
        suite = loader.discover('unit', pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        
        return result
    except Exception as e:
        print(f"❌ Failed to run unit tests: {e}")
        return None


def run_integration_tests():
    """Run integration tests (requires dependencies)"""
    print("🔗 Running Integration Tests (Requires Dependencies)")
    print("=" * 50)
    
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        return None
    
    try:
        # Discover and run integration tests
        loader = unittest.TestLoader()
        suite = loader.discover('integration', pattern='test_*.py')
        
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        
        return result
    except Exception as e:
        print(f"❌ Failed to run integration tests: {e}")
        return None


def run_smoke_tests():
    """Run smoke tests with real dependencies"""
    print("🔥 Running Smoke Tests (Real Dependencies)")
    print("=" * 50)
    
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("📋 Install with: pip install networkx osmnx numpy matplotlib pandas")
        return None
    
    try:
        # Import and run smoke tests
        import importlib.util
        import os
        smoke_path = os.path.join(os.path.dirname(__file__), "smoke_tests.py")
        spec = importlib.util.spec_from_file_location("smoke_tests", smoke_path)
        smoke_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(smoke_module)
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(smoke_module)
        
        runner = unittest.TextTestRunner(verbosity=2, buffer=True)
        result = runner.run(suite)
        
        return result
    except Exception as e:
        print(f"❌ Failed to run smoke tests: {e}")
        return None


def print_test_summary(result, test_type="tests"):
    """Print test summary"""
    if result is None:
        print(f"\n❌ {test_type.title()} could not be run due to missing dependencies")
        return False
    
    print(f"\n{'='*60}")
    print(f"{test_type.upper()} SUMMARY")
    print(f"{'='*60}")
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
    
    success = result.wasSuccessful()
    if success:
        print(f"✅ All {test_type} passed!")
    else:
        print(f"❌ Some {test_type} failed!")
    
    return success


def main():
    """Main test runner"""
    print("🚀 Route Services Test Runner")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()
    else:
        test_type = 'unit'
    
    all_success = True
    
    if test_type in ['unit', 'all']:
        # Run unit tests if dependencies available
        result = run_unit_tests()
        if result:
            success = print_test_summary(result, "unit tests")
            all_success = all_success and success
        else:
            all_success = False
    
    if test_type in ['integration', 'all']:
        # Run integration tests if dependencies available
        result = run_integration_tests()
        if result:
            success = print_test_summary(result, "integration tests")
            all_success = all_success and success
        else:
            all_success = False
    
    if test_type in ['smoke', 'all']:
        # Run smoke tests if dependencies available
        result = run_smoke_tests()
        if result:
            success = print_test_summary(result, "smoke tests")
            all_success = all_success and success
        else:
            all_success = False
    
    if test_type not in ['unit', 'integration', 'smoke', 'all']:
        print("Usage: python run_tests.py [unit|integration|smoke|all]")
        print("\nTest types:")
        print("  unit        - Run unit tests (mocked, fast)")
        print("  integration - Run integration tests (mocked)")
        print("  smoke       - Run smoke tests (real dependencies, slower)")
        print("  all         - Run all available tests")
        print("\nDefault: unit")
        return
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    if all_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ All route services working correctly")
        print("✅ Applications functioning properly")
    else:
        print("⚠️ SOME ISSUES DETECTED")
        missing_deps = check_dependencies()
        if missing_deps:
            print(f"📋 Missing dependencies may be the cause: {', '.join(missing_deps)}")
            print("   Install with: pip install networkx osmnx numpy matplotlib pandas")
            print("   Then run: python run_tests.py all")
    
    # Exit with error code if tests failed
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
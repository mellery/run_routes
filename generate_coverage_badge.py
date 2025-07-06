#!/usr/bin/env python3
"""
Coverage Badge Generator
Generates coverage badges and summary reports from pytest-cov XML output
"""

import xml.etree.ElementTree as ET
import json
import os
from datetime import datetime
import subprocess

def parse_coverage_xml(xml_file="coverage.xml"):
    """Parse coverage XML file and extract coverage percentage"""
    if not os.path.exists(xml_file):
        return None
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Get overall coverage percentage
        coverage_attr = root.attrib
        line_rate = float(coverage_attr.get('line-rate', 0)) * 100
        branch_rate = float(coverage_attr.get('branch-rate', 0)) * 100
        
        # Get total lines and branches
        lines_covered = int(coverage_attr.get('lines-covered', 0))
        lines_valid = int(coverage_attr.get('lines-valid', 0))
        branches_covered = int(coverage_attr.get('branches-covered', 0))
        branches_valid = int(coverage_attr.get('branches-valid', 0))
        
        # Calculate combined coverage
        if lines_valid > 0 and branches_valid > 0:
            combined_coverage = ((lines_covered + branches_covered) / 
                               (lines_valid + branches_valid)) * 100
        elif lines_valid > 0:
            combined_coverage = line_rate
        else:
            combined_coverage = 0
        
        return {
            'line_rate': round(line_rate, 1),
            'branch_rate': round(branch_rate, 1),
            'combined_rate': round(combined_coverage, 1),
            'lines_covered': lines_covered,
            'lines_valid': lines_valid,
            'branches_covered': branches_covered,
            'branches_valid': branches_valid,
            'timestamp': datetime.now().isoformat()
        }
    
    except Exception as e:
        print(f"Error parsing coverage XML: {e}")
        return None

def get_coverage_color(percentage):
    """Get color for coverage badge based on percentage"""
    if percentage >= 90:
        return "brightgreen"
    elif percentage >= 80:
        return "green"
    elif percentage >= 70:
        return "yellowgreen"
    elif percentage >= 60:
        return "yellow"
    elif percentage >= 50:
        return "orange"
    else:
        return "red"

def generate_badge_url(label, message, color):
    """Generate shields.io badge URL"""
    # URL encode the message
    message = str(message).replace(" ", "%20")
    return f"https://img.shields.io/badge/{label}-{message}-{color}"

def generate_badge_markdown(coverage_data):
    """Generate markdown for coverage badges"""
    if not coverage_data:
        return "[![Coverage](https://img.shields.io/badge/coverage-unknown-lightgrey)]()"
    
    combined_rate = coverage_data['combined_rate']
    line_rate = coverage_data['line_rate']
    branch_rate = coverage_data['branch_rate']
    
    combined_color = get_coverage_color(combined_rate)
    line_color = get_coverage_color(line_rate)
    branch_color = get_coverage_color(branch_rate)
    
    badges = []
    
    # Combined coverage badge
    combined_url = generate_badge_url("coverage", f"{combined_rate}%25", combined_color)
    badges.append(f"[![Coverage]({combined_url})](htmlcov/index.html)")
    
    # Line coverage badge
    line_url = generate_badge_url("lines", f"{line_rate}%25", line_color)
    badges.append(f"[![Lines]({line_url})](htmlcov/index.html)")
    
    # Branch coverage badge
    branch_url = generate_badge_url("branches", f"{branch_rate}%25", branch_color)
    badges.append(f"[![Branches]({branch_url})](htmlcov/index.html)")
    
    return " ".join(badges)

def generate_coverage_summary(coverage_data):
    """Generate detailed coverage summary"""
    if not coverage_data:
        return "Coverage data not available"
    
    summary = f"""## Test Coverage Summary

**Overall Coverage:** {coverage_data['combined_rate']}%

### Coverage Breakdown
- **Line Coverage:** {coverage_data['line_rate']}% ({coverage_data['lines_covered']}/{coverage_data['lines_valid']} lines)
- **Branch Coverage:** {coverage_data['branch_rate']}% ({coverage_data['branches_covered']}/{coverage_data['branches_valid']} branches)

### Coverage Quality Assessment
"""
    
    combined_rate = coverage_data['combined_rate']
    if combined_rate >= 90:
        summary += "🟢 **EXCELLENT** - Comprehensive test coverage\n"
    elif combined_rate >= 80:
        summary += "🟡 **GOOD** - Solid test coverage with room for improvement\n"
    elif combined_rate >= 70:
        summary += "🟠 **MODERATE** - Adequate coverage, should be improved\n"
    elif combined_rate >= 60:
        summary += "🔴 **LOW** - Insufficient coverage, requires attention\n"
    else:
        summary += "🔴 **CRITICAL** - Very low coverage, immediate action needed\n"
    
    summary += f"\n*Last updated: {coverage_data['timestamp']}*\n"
    summary += "\n**View detailed coverage report:** [htmlcov/index.html](htmlcov/index.html)\n"
    
    return summary

def run_coverage_tests():
    """Run tests with coverage and generate reports"""
    print("🧪 Running tests with coverage...")
    
    try:
        # Run pytest with coverage
        cmd = [
            "python", "-m", "pytest",
            "tests/unit",  # Only run unit tests for coverage
            "--cov=route_services",
            "--cov=ga_chromosome", 
            "--cov=ga_population",
            "--cov=ga_operators",
            "--cov=ga_fitness",
            "--cov=genetic_route_optimizer",
            "--cov=ga_visualizer",
            "--cov=ga_parameter_tuning",
            "--cov=ga_performance",
            "--cov=route",
            "--cov=tsp_solver",
            "--cov=tsp_solver_fast",
            "--cov=graph_cache",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term-missing",
            "--cov-branch",
            "-q"  # Quiet output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/mike/src/run_routes")
        
        if result.returncode == 0:
            print("✅ Tests completed successfully")
            return True
        else:
            print(f"❌ Tests failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def main():
    """Main function to generate coverage reports and badges"""
    print("📊 Coverage Badge Generator")
    print("=" * 50)
    
    # Run tests with coverage
    if not run_coverage_tests():
        print("❌ Failed to run coverage tests")
        return
    
    # Parse coverage data
    coverage_data = parse_coverage_xml()
    
    if not coverage_data:
        print("❌ Failed to parse coverage data")
        return
    
    # Generate badge markdown
    badge_markdown = generate_badge_markdown(coverage_data)
    
    # Generate coverage summary
    coverage_summary = generate_coverage_summary(coverage_data)
    
    # Save badge markdown to file
    with open("coverage_badges.md", "w") as f:
        f.write("# Coverage Badges\\n\\n")
        f.write(badge_markdown + "\\n\\n")
        f.write(coverage_summary)
    
    # Save coverage data as JSON
    with open("coverage_data.json", "w") as f:
        json.dump(coverage_data, f, indent=2)
    
    # Print results
    print(f"\\n✅ Coverage analysis complete!")
    print(f"   Combined Coverage: {coverage_data['combined_rate']}%")
    print(f"   Line Coverage: {coverage_data['line_rate']}%")  
    print(f"   Branch Coverage: {coverage_data['branch_rate']}%")
    
    print(f"\\n📁 Generated files:")
    print(f"   📄 coverage_badges.md - Badge markdown")
    print(f"   📊 htmlcov/index.html - Detailed HTML report")
    print(f"   📈 coverage.xml - XML coverage data")
    print(f"   💾 coverage_data.json - JSON coverage data")
    
    print(f"\\n🏷️ Badge Markdown:")
    print(badge_markdown)

if __name__ == "__main__":
    main()
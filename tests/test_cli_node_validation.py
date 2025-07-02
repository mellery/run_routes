#!/usr/bin/env python3
"""
Test CLI Node Validation
Reproduce the exact validation issue from CLI route planner
"""

from cli_route_planner import CLIRoutePlanner

def test_node_validation():
    """Test the exact validation logic from CLI"""
    
    print("🧪 Testing CLI Node Validation")
    print("=" * 40)
    
    # Create CLI planner instance (same as in interactive mode)
    planner = CLIRoutePlanner()
    
    # Load network (same as CLI does)
    print("1️⃣ Loading network...")
    success = planner.load_network()
    
    if not success:
        print("❌ Failed to load network")
        return
    
    print(f"✅ Network loaded: {len(planner.graph.nodes)} nodes")
    
    # Test the problematic node
    test_node_str = "12629717481"
    print(f"\n2️⃣ Testing node validation for: {test_node_str}")
    
    try:
        # This is the exact same logic as in CLI route planner line 329
        start_node = int(test_node_str.strip())
        print(f"   Parsed as integer: {start_node}")
        
        # This is the exact same check as line 331
        is_valid = start_node in planner.graph.nodes
        print(f"   start_node in planner.graph.nodes: {is_valid}")
        
        if is_valid:
            print("✅ Node validation should PASS")
            
            # Get node info
            node_data = planner.graph.nodes[start_node]
            print(f"   Location: {node_data['y']:.6f}, {node_data['x']:.6f}")
            print(f"   Elevation: {node_data.get('elevation', 0):.0f}m")
        else:
            print("❌ Node validation FAILS")
            
    except ValueError as e:
        print(f"❌ Failed to parse as integer: {e}")
    
    # List the actual starting points to compare
    print(f"\n3️⃣ Listing starting points (same as CLI option 1):")
    selected_nodes = planner.list_starting_points(10)
    
    print(f"\n4️⃣ Checking if target node is in the list:")
    target_node = int(test_node_str)
    if target_node in selected_nodes:
        print(f"✅ Node {target_node} is in the displayed list")
    else:
        print(f"❌ Node {target_node} is NOT in the displayed list")
        print("   This could explain the confusion!")
    
    # Check the graph object identity
    print(f"\n5️⃣ Graph object verification:")
    print(f"   Graph type: {type(planner.graph)}")
    print(f"   Graph nodes type: {type(planner.graph.nodes)}")
    print(f"   Sample node types: {[type(n) for n in list(planner.graph.nodes)[:3]]}")

def main():
    """Main test function"""
    test_node_validation()

if __name__ == "__main__":
    main()
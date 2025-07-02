#!/usr/bin/env python3
"""
Test Improved CLI
Test the enhanced node validation and selection in CLI
"""

from cli_route_planner import CLIRoutePlanner

def test_node_selection():
    """Test the improved node selection logic"""
    
    print("🧪 Testing Improved CLI Node Selection")
    print("=" * 50)
    
    # Create CLI planner
    planner = CLIRoutePlanner()
    
    # Load network
    print("1️⃣ Loading network...")
    success = planner.load_network()
    
    if not success:
        print("❌ Failed to load network")
        return
    
    print(f"✅ Network loaded: {len(planner.graph.nodes)} nodes")
    
    # Test the enhanced list display
    print("\n2️⃣ Testing enhanced starting points display:")
    available_nodes = planner.list_starting_points(10)
    
    # Test direct node ID input
    test_node = "12629717481"
    print(f"\n3️⃣ Testing direct node ID input: {test_node}")
    
    # Simulate the improved validation logic
    try:
        input_num = int(test_node)
        
        # Check if it's an option number (1-10)
        if 1 <= input_num <= len(available_nodes):
            start_node = available_nodes[input_num - 1]
            print(f"✅ Interpreted as option {input_num}: Node {start_node}")
        else:
            # Treat as direct node ID
            start_node = input_num
            print(f"✅ Interpreted as direct node ID: {start_node}")
        
        # Validate the final node ID
        if start_node in planner.graph.nodes:
            print(f"✅ Node validation PASSED for {start_node}")
            
            # Get node info
            node_data = planner.graph.nodes[start_node]
            print(f"   Location: {node_data['y']:.6f}, {node_data['x']:.6f}")
            print(f"   Elevation: {node_data.get('elevation', 0):.0f}m")
        else:
            print(f"❌ Node validation FAILED for {start_node}")
            
    except Exception as e:
        print(f"❌ Error in validation logic: {e}")
    
    # Test option number selection
    print(f"\n4️⃣ Testing option number selection:")
    for i in range(1, min(4, len(available_nodes) + 1)):  # Test first 3 options
        option_node = available_nodes[i - 1]
        print(f"   Option {i} -> Node {option_node}")
        
        # Verify this node exists in graph
        if option_node in planner.graph.nodes:
            print(f"   ✅ Option {i} is valid")
        else:
            print(f"   ❌ Option {i} is invalid")

def main():
    """Main test function"""
    test_node_selection()
    
    print("\n" + "=" * 50)
    print("🎯 CLI Enhancement Test Complete")
    print("\n📋 Improvements made:")
    print("   • Enhanced error messages with debugging info")
    print("   • Support for option numbers (1-10)")
    print("   • Better input validation and feedback")
    print("   • Numbered list display for easier selection")

if __name__ == "__main__":
    main()
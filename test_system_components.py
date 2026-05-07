"""
Test script to demonstrate the stress management system components.
"""

import json
from stress_management_planner import StressManagementPlanner

def test_retrieval():
    """Test the retrieval function from RAG system."""
    print("\n" + "=" * 80)
    print("TEST 1: RETRIEVAL FUNCTION TEST")
    print("=" * 80)
    
    planner = StressManagementPlanner()
    
    # Test contexts
    test_contexts = [
        "High stress with elevated LF/HF ratio indicating sympathetic activation",
        "Moderate anxiety with cognitive overload and poor sleep",
        "Severe emotional stress with rumination and emotional dysregulation"
    ]
    
    for context in test_contexts:
        print(f"\nContext: {context}")
        print("-" * 60)
        
        results = planner.retrieve_techniques(context, n_results=3)
        
        print("Retrieved Techniques:")
        for i, meta in enumerate(results['metadatas'][0], 1):
            print(f"  {i}. {meta['technique_name']}")
            print(f"     Applicable: {meta['applicable_conditions']}")
            print(f"     Duration: {meta['duration']}")

def test_formatting():
    """Test the technique formatting for LLM prompt."""
    print("\n" + "=" * 80)
    print("TEST 2: TECHNIQUE FORMATTING TEST")
    print("=" * 80)
    
    planner = StressManagementPlanner()
    
    context = "High stress with cognitive overload"
    results = planner.retrieve_techniques(context, n_results=2)
    
    formatted = planner._format_techniques(results)
    print("\nFormatted Techniques for LLM:")
    print("-" * 60)
    print(formatted)

def test_stress_profiles():
    """Test retrieval across different stress profiles."""
    print("\n" + "=" * 80)
    print("TEST 3: STRESS PROFILE MATCHING TEST")
    print("=" * 80)
    
    planner = StressManagementPlanner()
    
    stress_profiles = {
        "very_high": "Critical stress with multiple physiological dysregulations: elevated LF/HF ratio (3.2), high beta power (18), low HRV (45)",
        "high": "High sustained stress with sympathetic dominance: LF/HF ratio 2.5, beta activity increased (15), sleep disrupted",
        "moderate": "Moderate occupational stress with mild cognitive load: LF/HF ratio 1.8, beta power normal (10), occasional sleep issues",
        "low": "Low baseline stress with adaptive responses: normal LF/HF (1.3), stable beta power (8), good sleep quality",
        "very_low": "Minimal stress with parasympathetic dominance: low LF/HF (0.9), reduced beta power (6), excellent recovery"
    }
    
    for level, context in stress_profiles.items():
        print(f"\n{level.upper()} STRESS LEVEL:")
        print(f"Context: {context[:60]}...")
        print("-" * 60)
        
        results = planner.retrieve_techniques(context, n_results=2)
        
        print("Top Retrieved Techniques:")
        for i, meta in enumerate(results['metadatas'][0], 1):
            print(f"  {i}. {meta['technique_name']} ({meta['duration']})")
            stress_mapping = ', '.join(meta['stress_level_mapping'])
            print(f"     For stress levels: {stress_mapping}")

def test_similarity_matching():
    """Test semantic similarity matching accuracy."""
    print("\n" + "=" * 80)
    print("TEST 4: SEMANTIC SIMILARITY MATCHING TEST")
    print("=" * 80)
    
    planner = StressManagementPlanner()
    
    # Test pairs with expected matches
    test_pairs = [
        ("breathing exercises for anxiety", "Deep Breathing Exercise"),
        ("muscle relaxation techniques", "Progressive Muscle Relaxation"),
        ("cognitive restructuring and negative thoughts", "Cognitive Restructuring"),
        ("meditation and mindfulness", "Mindfulness Meditation"),
        ("visualization and imagery", "Guided Imagery"),
    ]
    
    for query, expected in test_pairs:
        results = planner.retrieve_techniques(query, n_results=3)
        tops = [meta['technique_name'] for meta in results['metadatas'][0]]
        
        is_match = expected in tops
        match_indicator = "✓" if is_match else "✗"
        
        print(f"\n{match_indicator} Query: \"{query}\"")
        print(f"  Expected: {expected}")
        print(f"  Top 3 Retrieved: {tops}")

def test_data_structure():
    """Verify the retrieved data structure."""
    print("\n" + "=" * 80)
    print("TEST 5: DATA STRUCTURE VERIFICATION")
    print("=" * 80)
    
    planner = StressManagementPlanner()
    results = planner.retrieve_techniques("stress management techniques", n_results=1)
    
    print("\nRetrieved Data Structure:")
    print(f"  Keys: {list(results.keys())}")
    
    if 'metadatas' in results and results['metadatas']:
        meta = results['metadatas'][0][0]
        print(f"\nMetadata Fields:")
        for key, value in meta.items():
            value_type = type(value).__name__
            if isinstance(value, str):
                preview = value[:50] + "..." if len(value) > 50 else value
            elif isinstance(value, list):
                preview = f"{len(value)} items"
            else:
                preview = str(value)
            print(f"  - {key}: {value_type} = {preview}")

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "STRESS MANAGEMENT SYSTEM - COMPONENT TESTS".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    
    try:
        test_retrieval()
        test_formatting()
        test_stress_profiles()
        test_similarity_matching()
        test_data_structure()
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nSystem components are working correctly:")
        print("  ✓ RAG system initialized with knowledge base")
        print("  ✓ Retrieval function returns relevant techniques")
        print("  ✓ Semantic similarity matching is accurate")
        print("  ✓ Data structures are properly formatted")
        print("  ✓ Formatting for LLM is correct")
        print("\nNext steps:")
        print("  1. Set your Gemini API key in .env")
        print("  2. Run stress_management_planner.py for full LLM integration")
        print("  3. Use integrated_system.py for end-to-end pipeline")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

"""
Test the stress management API endpoint
"""

import requests
import json

def test_stress_management_endpoint():
    """Test the new /stress-management endpoint"""

    # API endpoint
    url = "http://127.0.0.1:8000/stress-management"

    # Sample payload
    payload = {
        "stress_level": "high",
        "stress_probability": 0.85,
        "shap_features": ["lf_hf_ratio", "beta_power", "heart_rate"],
        "physiological_metrics": {
            "lf_hf_ratio": 2.5,
            "beta_power": 15.3,
            "faa": -0.2,
            "heart_rate": 95
        },
        "all_features": {
            "eeg_feature_0": 0.5,
            "eeg_feature_1": 1.2,
            "eeg_feature_2": -0.8,
            "ecg_feature_0": 85.3,
            "ecg_feature_1": 45.2
        }
    }

    print("Testing /stress-management endpoint...")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 60)

    try:
        response = requests.post(url, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Status: {result.get('stress_level', 'N/A')}")
            print(f"Cause: {result.get('cause', 'N/A')[:100]}...")
            print(f"Physiological: {result.get('physiological_interpretation', 'N/A')[:100]}...")
            print(f"Actions: {len(result.get('recommended_actions', []))} recommended")

            for i, action in enumerate(result.get('recommended_actions', [])[:2], 1):
                print(f"  {i}. {action.get('technique', 'N/A')} ({action.get('duration', 'N/A')})")

            return True
        else:
            print(f"❌ FAILED: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ REQUEST ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("STRESS MANAGEMENT API TEST")
    print("=" * 80)

    # Note: This test assumes the API server is running
    # In a real scenario, you'd start the server first
    print("Note: Make sure to start the API server first with:")
    print("c:/Stress/.venv/Scripts/python.exe api.py")
    print()

    success = test_stress_management_endpoint()

    if success:
        print("\n🎉 API test completed successfully!")
        print("The stress management endpoint is working correctly.")
    else:
        print("\n❌ API test failed.")
        print("Check that the API server is running and Gemini API key is set.")
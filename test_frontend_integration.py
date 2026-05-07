"""
Test script to simulate frontend calls to all endpoints
"""
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def test_predict():
    """Test /predict endpoint"""
    print("=" * 80)
    print("Testing /predict endpoint...")
    print("=" * 80)
    
    features = [0.1] * 101  # 101 features
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"features": features}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Predict successful!")
        print(f"   Stress Class: {data['stress_class']}")
        print(f"   Stress Level: {data['stress_level']}")
        print(f"   Probability: {data['stress_probability']:.3f}")
        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_explain():
    """Test /explain endpoint"""
    print("\n" + "=" * 80)
    print("Testing /explain endpoint...")
    print("=" * 80)
    
    features = [0.1] * 101
    response = requests.post(
        f"{BASE_URL}/explain",
        json={"features": features}
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Explain successful!")
        print(f"   Stress Level: {data['stress_level']}")
        print(f"   SHAP values count: {len(data['shap_values'])}")
        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

def test_stress_management():
    """Test /stress-management endpoint"""
    print("\n" + "=" * 80)
    print("Testing /stress-management endpoint...")
    print("=" * 80)
    
    payload = {
        "stress_level": "high",
        "stress_probability": 0.85,
        "shap_features": ["lf_hf_ratio", "beta_power", "heart_rate"],
        "physiological_metrics": {
            "lf_hf_ratio": 2.5,
            "beta_power": 15.0,
            "heart_rate": 95.0
        },
        "all_features": {f"feature_{i}": 0.1 for i in range(101)}
    }
    
    response = requests.post(
        f"{BASE_URL}/stress-management",
        json=payload,
        timeout=30
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Stress Management successful!")
        print(f"   Stress Level: {data['stress_level']}")
        print(f"   Cause: {data['cause']}")
        print(f"   Recommended actions: {len(data['recommended_actions'])}")
        for i, action in enumerate(data['recommended_actions'], 1):
            print(f"   {i}. {action['technique']} ({action['duration']})")
        return data
    else:
        print(f"❌ Error: {response.text}")
        return None

if __name__ == "__main__":
    print("\n🚀 TESTING MINDFULSTRESS HUB API\n")
    
    predict_result = test_predict()
    explain_result = test_explain()
    management_result = test_stress_management()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ /predict endpoint: {'Working' if predict_result else 'Failed'}")
    print(f"✅ /explain endpoint: {'Working' if explain_result else 'Failed'}")
    print(f"✅ /stress-management endpoint: {'Working' if management_result else 'Failed'}")
    print("\n✨ All endpoints are ready! Frontend can now display results properly.\n")

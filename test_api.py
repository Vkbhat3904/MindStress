import requests
import json
import random

def test_api():
    url_predict = "http://127.0.0.1:8000/predict"
    url_explain = "http://127.0.0.1:8000/explain"
    
    # Generate 101 random floats to simulate the combined EEG (97) and ECG (4) feature vector
    dummy_features = [random.uniform(-2.0, 2.0) for _ in range(101)]
    
    payload = {
        "features": dummy_features
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    print("====================================")
    print("Testing /predict Endpoint:")
    print("====================================")
    print(f"Sending POST request to {url_predict} with {len(dummy_features)} dummy features...")
    
    try:
        response = requests.post(url_predict, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response JSON:")
            print(json.dumps(response.json(), indent=4))
        else:
            print("Failed Response:", response.text)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect. Is the FastAPI app running on port 8000?")
        return
        
    print("\n====================================")
    print("Testing /explain Endpoint:")
    print("====================================")
    print(f"Sending POST request to {url_explain} ...")
    
    try:
        response = requests.post(url_explain, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            results = response.json()
            print("Response JSON (Abbreviated):")
            print(f"- Stress Probability: {results.get('stress_probability')}")
            print(f"- Stress Level: {results.get('stress_level')}")
            
            print("\n- Attention Weights:")
            print(json.dumps(results.get('attention_weights'), indent=4))
            
            shap_vals = results.get('shap_values', {})
            print(f"\n- SHAP Values (First 5 of {len(shap_vals)} shown):")
            top_5_shap = {k: shap_vals[k] for k in list(shap_vals)[:5]}
            print(json.dumps(top_5_shap, indent=4))
            
        else:
            print("Failed Response:", response.text)
    except Exception as e:
        print(f"Error testing /explain: {e}")

if __name__ == "__main__":
    test_api()

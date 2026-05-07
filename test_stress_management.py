import requests
import json

# Test the stress management endpoint
url = 'http://127.0.0.1:8000/stress-management'
data = {
    'stress_level': 'high',
    'stress_probability': 0.85,
    'shap_features': ['lf_hf_ratio', 'beta_power', 'heart_rate'],
    'physiological_metrics': {
        'lf_hf_ratio': 2.5,
        'heart_rate': 95.0,
        'beta_power': 15.0
    },
    'all_features': {f'feature_{i}': 0.1 for i in range(101)}  # Dictionary format
}

print('Testing /stress-management endpoint...')
try:
    response = requests.post(url, json=data, timeout=30)
    print(f'Status Code: {response.status_code}')
    if response.status_code == 200:
        result = response.json()
        print('✅ Stress management plan generated successfully!')
        print(f'Stress Level: {result["stress_level"]}')
        print(f'Cause: {result["cause"]}')
        print(f'Number of recommended actions: {len(result["recommended_actions"])}')
    else:
        print(f'❌ Error: {response.text}')
except Exception as e:
    print(f'❌ Error: {e}')
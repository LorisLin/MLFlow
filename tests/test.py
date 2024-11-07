import requests

base_url = "http://localhost:5050"

def test_predict():
    print("Testing /predict endpoint...")
    data = {
        "features": [5.1, 3.5, 1.4, 0.2] 
    }

    response = requests.post(f"{base_url}/predict", json=data)

    assert response.status_code == 200, f"Error: {response.status_code}"
    response_data = response.json()
    assert "predictions" in response_data, "No 'predictions' key in response"
    
    print(f"Predictions: {response_data['predictions']}")
    print("/predict test passed!\n")

def test_update_model():
    print("Testing /update-model endpoint...")
    response = requests.post(f"{base_url}/update-model")
    assert response.status_code == 200, f"Error: {response.status_code}"
    response_data = response.json()
    assert "status" in response_data, "No 'status' key in response"
    print("/update-model test passed!\n")

if __name__ == "__main__":
    test_predict()
    test_update_model()
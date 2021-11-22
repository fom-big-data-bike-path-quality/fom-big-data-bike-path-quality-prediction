import json

import requests

from test_values import sample_asphalt, sample_paved_stones, sample_sett


class TestApp:
    def test_predict_cnn_asphalt(self):
        url = f"http://localhost:8000/predict/cnn"
        response = requests.post(url=url, json=sample_asphalt, headers={"Content-Type": "application/json"})
        assert response.status_code == 200
        assert json.loads(response.text)["surface_type"] == "asphalt"

    def test_predict_cnn_paved_stones(self):
        url = f"http://localhost:8000/predict/cnn"
        response = requests.post(url=url, json=sample_paved_stones, headers={"Content-Type": "application/json"})
        assert response.status_code == 200
        assert json.loads(response.text)["surface_type"] == "paved_stones"

    def test_predict_cnn_sett(self):
        url = f"http://localhost:8000/predict/cnn"
        response = requests.post(url=url, json=sample_sett, headers={"Content-Type": "application/json"})
        assert response.status_code == 200
        assert json.loads(response.text)["surface_type"] == "sett"

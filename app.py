import os

import torch
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Configure which model to use
        target = "2021-08-07-00:29:04"

        # Load model
        model = torch.load(os.path.join("./models/models", target, "model.pickle"))

        return jsonify({"hello": "world"})


if __name__ == '__main__':
    app.run()

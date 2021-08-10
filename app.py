import os
import shutil
from datetime import datetime

import torch
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Load model
        model_version = "2021-08-07-00:29:04"
        model = torch.load(os.path.join("./models/models", model_version, "model.pickle"))

        # Make workspace directory
        workspace_directory = os.path.join("workspace", datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        os.makedirs(workspace_directory, exist_ok=True)

        with open(os.path.join(workspace_directory, "bike_activity_sample.json"), "w") as json_file:
            json_file.write("%s" % request.get_json(force=True))

        # TODO

        # Delete workspace directory
        shutil.rmtree(workspace_directory)

        return jsonify({"hello": "world"})


if __name__ == '__main__':
    app.run()

import csv
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

        payload = request.get_json(force=True)

        convert_bike_activity_sample_to_json_file(workspace_directory, "bike_activity_sample.json", payload)
        convert_bike_activity_sample_to_csv_file(workspace_directory, "bike_activity_sample.csv", payload)

        # TODO

        # Delete workspace directory
        shutil.rmtree(workspace_directory)

        return jsonify({"hello": "world"})


def convert_bike_activity_sample_to_json_file(results_path, results_file_name, data):
    with open(os.path.join(results_path, results_file_name), "w") as json_file:
        json_file.write("%s" % data)


def convert_bike_activity_sample_to_csv_file(results_path, results_file_name, data):
    bike_activity_sample_with_measurements = data

    bike_activity_uid = None
    bike_activity_surface_type = None
    bike_activity_smoothness_type = None
    bike_activity_phone_position = None
    bike_activity_bike_type = None

    with open(results_path + "/" + results_file_name, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([
            # Descriptive values
            'bike_activity_uid',
            'bike_activity_sample_uid',
            'bike_activity_measurement',
            'bike_activity_measurement_timestamp',
            'bike_activity_measurement_lon',
            'bike_activity_measurement_lat',
            # Input values
            'bike_activity_measurement_speed',
            'bike_activity_measurement_accelerometer_x',
            'bike_activity_measurement_accelerometer_y',
            'bike_activity_measurement_accelerometer_z',
            'bike_activity_phone_position',
            'bike_activity_bike_type',
            # Output values
            'bike_activity_surface_type',
            'bike_activity_smoothness_type',
        ])

        bike_activity_sample = bike_activity_sample_with_measurements["bikeActivitySample"]
        bike_activity_measurements = bike_activity_sample_with_measurements["bikeActivityMeasurements"]

        bike_activity_sample_uid = bike_activity_sample["uid"]
        bike_activity_sample_surface_type = bike_activity_sample["surfaceType"] if "surfaceType" in bike_activity_sample else None

        for bike_activity_measurement in bike_activity_measurements:
            bike_activity_measurement_uid = bike_activity_measurement["uid"]
            bike_activity_measurement_timestamp = bike_activity_measurement["timestamp"]
            bike_activity_measurement_lon = bike_activity_measurement["lon"]
            bike_activity_measurement_lat = bike_activity_measurement["lat"]
            bike_activity_measurement_speed = bike_activity_measurement["speed"]
            bike_activity_measurement_accelerometer_x = bike_activity_measurement["accelerometerX"]
            bike_activity_measurement_accelerometer_y = bike_activity_measurement["accelerometerY"]
            bike_activity_measurement_accelerometer_z = bike_activity_measurement["accelerometerZ"]

            csv_writer.writerow([
                bike_activity_uid,
                bike_activity_sample_uid,
                bike_activity_measurement_uid,
                bike_activity_measurement_timestamp,
                bike_activity_measurement_lon,
                bike_activity_measurement_lat,
                # Input values
                bike_activity_measurement_speed,
                bike_activity_measurement_accelerometer_x,
                bike_activity_measurement_accelerometer_y,
                bike_activity_measurement_accelerometer_z,
                bike_activity_phone_position,
                bike_activity_bike_type,
                # Output values
                bike_activity_sample_surface_type if bike_activity_sample_surface_type is not None else bike_activity_surface_type,
                bike_activity_smoothness_type,
            ])


if __name__ == '__main__':
    app.run()

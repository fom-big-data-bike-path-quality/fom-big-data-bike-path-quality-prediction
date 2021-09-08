import csv
import os
import shutil
import sys
from datetime import datetime

import torch
from flask import Flask, jsonify, request

# Make library available in path
library_paths = [
    os.path.join(os.getcwd(), 'lib'),
    os.path.join(os.getcwd(), 'lib/data_pre_processing'),
    os.path.join(os.getcwd(), 'lib/data_preparation'),
    os.path.join(os.getcwd(), 'lib/log'),
    os.path.join(os.getcwd(), 'lib/base_model'),
    os.path.join(os.getcwd(), 'lib/base_model/layers'),
]

for p in library_paths:
    if not (p in sys.path):
        sys.path.insert(0, p)

# Import library classes
from logger_facade import LoggerFacade
from data_loader import DataLoader
from data_filterer import DataFilterer
from data_transformer import DataTransformer
from data_normalizer import DataNormalizer
from cnn_base_model_helper import CnnBaseModelHelper
from classifier import Classifier

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        model_version = "latest"
        model = Classifier(
            input_channels=1,  # TODO Derive this value from data
            # input_channels=train_array.shape[1],
            num_classes=18
        ).to(device)
        model.load_state_dict(torch.load(os.path.join("./models/models", model_version, "model.pickle")))
        model.eval()

        # Make workspace directory
        workspace_path = os.path.join("workspace", datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        os.makedirs(workspace_path, exist_ok=True)

        # Initialize logger
        logger = LoggerFacade(workspace_path, console=True, file=True)
        logger.log_line("Start Predication")

        payload = request.get_json(force=True)

        convert_bike_activity_sample_to_json_file(workspace_path, "bike_activity_sample.json", payload)
        convert_bike_activity_sample_to_csv_file(workspace_path, "bike_activity_sample.csv", payload)

        dataframes = DataLoader().run(
            logger=logger,
            data_path=workspace_path
        )

        try:
            dataframes = DataFilterer().run(logger=logger, dataframes=dataframes)
            dataframes = DataTransformer().run(logger=logger, dataframes=dataframes)
            dataframes = DataNormalizer().run(logger=logger, dataframes=dataframes)

            tensor = CnnBaseModelHelper().run(
                logger=logger,
                predict_dataframes=dataframes,
                log_path=workspace_path
            )

            outputs = model.forward(tensor)
            _, prediction = outputs.max(1)

            # Delete workspace directory
            shutil.rmtree(workspace_path)

            return jsonify({"surface_type": DataTransformer().runReverse(prediction)})

        except Exception as inst:
            return jsonify({"error": inst.args[0]})


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

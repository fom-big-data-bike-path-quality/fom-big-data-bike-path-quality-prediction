import csv
import os
import shutil
import sys

import torch
from flask import Flask, jsonify, request

file_path = os.path.realpath(__file__)
script_path = os.path.dirname(file_path)

# Make library available in path
library_paths = [
    os.path.join(script_path, 'analytics', 'lib'),
    os.path.join(script_path, 'analytics', 'lib', 'log'),
    os.path.join(script_path, 'analytics', 'lib', 'data_transformation'),
    os.path.join(script_path, 'analytics', 'lib', 'data_pre_processing'),
    os.path.join(script_path, 'analytics', 'lib', 'data_preparation'),
    os.path.join(script_path, 'analytics', 'lib', 'models'),
    os.path.join(script_path, 'analytics', 'lib', 'models', 'base_model_knn_dtw'),
    os.path.join(script_path, 'analytics', 'lib', 'models', 'base_model_cnn'),
    os.path.join(script_path, 'analytics', 'lib', 'models', 'base_model_cnn', 'layers'),
    os.path.join(script_path, 'analytics', 'lib', 'models', 'base_model_lstm'),
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
from model_preparator import ModelPreparator
from label_encoder import LabelEncoder
from cnn_classifier import CnnClassifier

app = Flask(__name__)

# Number of classes
num_classes = LabelEncoder().num_classes()

slice_width = 500
measurement_speed_limit = 5.0


@app.route('/predict/cnn', methods=['POST'])
def predict_cnn():
    if request.method == 'POST':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine number of linear channels based on slice width
        linear_channels = ModelPreparator().get_linear_channels(slice_width)

        # Define classifier
        classifier = CnnClassifier(
            input_channels=1,
            num_classes=num_classes,
            linear_channels=linear_channels
        ).to(device)

        # Load model
        model_version = "latest"
        classifier.load_state_dict(torch.load(
            os.path.join(script_path, "results", "results", "cnn", model_version, "04-modelling", "model.pickle"),
            map_location=torch.device(device)
        ))
        classifier.eval()

        # Make workspace directory
        workspace_path = os.path.join(script_path, "workspace", "latest")
        os.makedirs(workspace_path, exist_ok=True)

        # Initialize logger
        logger = LoggerFacade(workspace_path, console=True, file=True)
        logger.log_line("Start Prediction")

        payload = request.get_json(force=True)
        payload_to_json_file(workspace_path, "bike_activity_sample.json", payload)
        payload_to_csv_file(workspace_path, "bike_activity_sample.csv", payload)

        dataframes = DataLoader().run(
            logger=logger,
            data_path=workspace_path
        )

        try:
            dataframes = DataFilterer().run(logger=logger, dataframes=dataframes, slice_width=slice_width,
                                            measurement_speed_limit=measurement_speed_limit,
                                            keep_unflagged_lab_conditions=True)
            dataframes = DataTransformer().run(logger=logger, dataframes=dataframes, skip_label_encode_surface_type=True)
            dataframes = DataNormalizer().run(logger=logger, dataframes=dataframes)

            tensor = ModelPreparator().create_tensor(dataframes=dataframes, device=device)

            outputs = classifier.forward(tensor)
            _, prediction = outputs.max(1)

            # Delete workspace directory
            shutil.rmtree(workspace_path)

            return jsonify({"surface_type": DataTransformer().run_reverse(prediction)})

        except Exception as inst:
            return jsonify({"error": inst.args[0]})


def payload_to_json_file(results_path, results_file_name, data):
    with open(os.path.join(results_path, results_file_name), "w") as json_file:
        json_file.write(("%s" % data).replace("\'", "\""))


def payload_to_csv_file(results_path, results_file_name, data):
    bike_activity_sample_with_measurements = data

    bike_activity_uid = 0
    bike_activity_surface_type = "unknown"
    bike_activity_smoothness_type = "unknown"
    bike_activity_phone_position = "unknown"
    bike_activity_bike_type = "unknown"

    with open(results_path + "/" + results_file_name, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([
            # Descriptive values
            "bike_activity_uid",
            "bike_activity_sample_uid",
            "bike_activity_measurement",
            "bike_activity_measurement_timestamp",
            "bike_activity_measurement_lon",
            "bike_activity_measurement_lat",
            # Input values
            "bike_activity_measurement_speed",
            "bike_activity_measurement_accelerometer_x",
            "bike_activity_measurement_accelerometer_y",
            "bike_activity_measurement_accelerometer_z",
            "bike_activity_phone_position",
            "bike_activity_bike_type",
            # Output values
            "bike_activity_surface_type",
            "bike_activity_smoothness_type",
        ])

        bike_activity_sample = bike_activity_sample_with_measurements["bikeActivitySample"]
        bike_activity_measurements = bike_activity_sample_with_measurements["bikeActivityMeasurements"]

        bike_activity_sample_uid = bike_activity_sample["uid"]
        bike_activity_sample_surface_type = bike_activity_sample[
            "surfaceType"] if "surfaceType" in bike_activity_sample else None

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

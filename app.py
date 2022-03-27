import csv
import json
import os
import shutil
import sys
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi_versioning import VersionedFastAPI, version
from pydantic import BaseModel

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
from lstm_classifier import LstmClassifier
from test_values import sample_asphalt

app = FastAPI(name="Bike Path Quality")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "https://bike-path-quality.web.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Number of classes
num_classes = LabelEncoder().num_classes()

slice_width = 500
measurement_speed_limit = 5.0


class BikeActivitySample(BaseModel):
    bikeActivityUid: str
    lat: float
    lon: float
    speed: float
    timestamp: int
    uid: str


class BikeActivityMeasurement(BaseModel):
    accelerometerX: int
    accelerometerY: int
    accelerometerZ: int
    bikeActivitySampleUid: str
    lat: float
    lon: float
    speed: float
    timestamp: int
    uid: str


class BikeActivitySampleWithMeasurements(BaseModel):
    bikeActivitySample: BikeActivitySample
    bikeActivityMeasurements: List[BikeActivityMeasurement] = []


class ResultWrapper(BaseModel):
    surface_type = ""

    def __init__(self, surface_type: str):
        super().__init__()
        self.surface_type = surface_type


class ErrorWrapper(BaseModel):
    error = ""

    def __init__(self, error: str):
        super().__init__()
        self.error = error


@app.post('/predict/cnn', status_code=200)
@version(1, 0)
def predict_cnn(bike_activity_sample_with_measurements: BikeActivitySampleWithMeasurements = sample_asphalt,
                response: Response = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Determine kernel size based on slice width
    kernel_size = ModelPreparator().get_kernel_size(slice_width)

    # Determine number of linear channels based on slice width
    linear_channels = ModelPreparator().get_linear_channels(slice_width)

    # Define classifier
    classifier = CnnClassifier(
        input_channels=1,
        kernel_size=kernel_size,
        num_classes=num_classes,
        linear_channels=linear_channels
    ).to(device)

    # Load model
    model_version = "2022-02-24-12:07:23"
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

    # Serialize sample
    payload_to_json_file(workspace_path, "bike_activity_sample.json", bike_activity_sample_with_measurements)
    payload_to_csv_file(workspace_path, "bike_activity_sample.csv", bike_activity_sample_with_measurements)

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

        return ResultWrapper(surface_type=DataTransformer().run_reverse(prediction))

    except Exception as inst:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ErrorWrapper(error=inst.args[0])


@app.post('/predict/lstm')
@version(1, 0)
def predict_lstm(bike_activity_sample_with_measurements: BikeActivitySampleWithMeasurements = sample_asphalt,
                 response: Response = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set default values
    dropout = 0.5
    lstm_hidden_dimension = 128
    lstm_layer_dimension = 3

    # Define classifier
    classifier = LstmClassifier(device=device, input_size=slice_width, hidden_dimension=lstm_hidden_dimension,
                                layer_dimension=lstm_layer_dimension, num_classes=num_classes,
                                dropout=dropout).to(device)

    # Load model
    model_version = "2022-02-25-08:43:39"
    classifier.load_state_dict(torch.load(
        os.path.join(script_path, "results", "results", "lstm", model_version, "04-modelling", "model.pickle"),
        map_location=torch.device(device)
    ))
    classifier.eval()

    # Make workspace directory
    workspace_path = os.path.join(script_path, "workspace", "latest")
    os.makedirs(workspace_path, exist_ok=True)

    # Initialize logger
    logger = LoggerFacade(workspace_path, console=True, file=True)
    logger.log_line("Start Prediction")

    # Serialize sample
    payload_to_json_file(workspace_path, "bike_activity_sample.json", bike_activity_sample_with_measurements)
    payload_to_csv_file(workspace_path, "bike_activity_sample.csv", bike_activity_sample_with_measurements)

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

        return ResultWrapper(surface_type=DataTransformer().run_reverse(prediction))

    except Exception as inst:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return ErrorWrapper(error=inst.args[0])


def payload_to_json_file(results_path, results_file_name,
                         bike_activity_sample_with_measurements: BikeActivitySampleWithMeasurements):
    with open(os.path.join(results_path, results_file_name), "w") as json_file:
        json_content = json.dumps(bike_activity_sample_with_measurements, default=lambda x: x.__dict__)
        json_file.write("%s" % json_content)


def payload_to_csv_file(results_path, results_file_name,
                        bike_activity_sample_with_measurements: BikeActivitySampleWithMeasurements):
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

        for bike_activity_measurement in bike_activity_sample_with_measurements.bikeActivityMeasurements:
            csv_writer.writerow([
                bike_activity_uid,
                bike_activity_sample_with_measurements.bikeActivitySample.uid,
                bike_activity_measurement.uid,
                bike_activity_measurement.timestamp,
                bike_activity_measurement.lon,
                bike_activity_measurement.lat,
                # Input values
                bike_activity_measurement.speed,
                bike_activity_measurement.accelerometerX,
                bike_activity_measurement.accelerometerY,
                bike_activity_measurement.accelerometerZ,
                bike_activity_phone_position,
                bike_activity_bike_type,
                # Output values
                bike_activity_surface_type,
                bike_activity_smoothness_type
            ])


app = VersionedFastAPI(app, default_api_version=(1, 0))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

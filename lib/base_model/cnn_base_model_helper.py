import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_array(dataframes):
    """
    Converts an array of data frame into a 3D numpy array

    axis-0 = epoch
    axis-1 = features in a measurement
    axis-2 = measurements in an epoch

    """
    array = []

    for name, dataframe in dataframes.items():
        array.append(dataframe.to_numpy())

    return np.dstack(array).transpose(2, 1, 0)


def create_dataset(array):
    return TensorDataset(
        # 3D array with
        # axis-0 = epoch
        # axis-1 = features in a measurement (INPUT)
        # axis-2 = measurements in an epoch
        torch.tensor(data=array[:, -1:].astype("float64")).float(),
        # 1D array with
        # axis-0 = TARGET of an epoch
        torch.tensor(data=array[:, 0, :][:, 0].astype("int64")).long()
    )


def create_loader(dataset, batch_size=128, shuffle=False, num_workers=0):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


#
# Main
#

class CnnBaseModelHelper:

    def run(self, logger, predict_dataframes, log_path):
        # Create arrays
        predict_array = create_array(predict_dataframes)

        # Create data sets
        predict_dataset = create_dataset(predict_array)

        # Create data loaders
        predict_data_loader = create_loader(predict_dataset, shuffle=False)

        for i, batch in enumerate(predict_data_loader):
            x_raw, y_batch = [t.to(device) for t in batch]
            return x_raw

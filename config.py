import os
import mne
import scipy.io as sio
import json
import numpy as np

class Config:
    # Save path
    CLEANED_DATA_PATH = os.getenv("CLEANED_DATA_PATH", "data/clean")
    DE_DATA_PATH = os.getenv("DE_DATA_PATH", "data/de")
    ICA_DATA_PATH = os.getenv("ICA_DATA_PATH", "data/ica")

    # Preprocessing parameters
    SAMPLING_RATE = int(os.getenv("SAMPLING_RATE", 200))

    # Standard 62 scalp electrodes (10-20 system, excluding references/EOG/ECG)
    # Order matches SEED-IV convention
    STANDARD_62_ELECTRODES = [
        "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
        "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
        "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
        "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
        "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
        "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
        "CB1", "O1", "OZ", "O2", "CB2",
    ]

    @staticmethod
    def create_directories():
        os.makedirs(Config.CLEANED_DATA_PATH, exist_ok=True)
        os.makedirs(Config.DE_DATA_PATH, exist_ok=True)
        os.makedirs(Config.ICA_DATA_PATH, exist_ok=True)



class DataLoader:
    raw_dataset_info = [
        {
            "name": "example_data",
            "path": "",
            }
    ]


def load_raw_data(file_path):
    # Load .mat file
    mat = sio.loadmat(file_path)
    
    # Extract data and labels
    data = mat['data']  # Adjust key based on actual structure
    labels = mat['labels']  # Adjust key based on actual structure
    
    return data, labels

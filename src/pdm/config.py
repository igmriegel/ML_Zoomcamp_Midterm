"""Project-wide constants and configuration."""

from pathlib import Path

SEED = 42
MODEL_VERSION = "0.1.0"

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / "data" / "ai4i2020.csv"
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.bin"
METRICS_PATH = MODELS_DIR / "metrics.json"

TARGET_COL = "machine_failure"
MODE_COLS = ["twf", "hdf", "pwf", "osf"]
LEAKAGE_COLS = ["udi", "product_id", "twf", "hdf", "pwf", "osf", "rnf"]

RAW_FEATURE_COLS = [
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

DERIVED_FEATURE_COLS = ["temp_diff_k", "power_w", "strain", "strain_ratio"]

TYPE_STRAIN_LIMITS: dict[str, float] = {
    "L": 11000.0,
    "M": 12000.0,
    "H": 13000.0,
}

CATEGORICAL_COLS = ["type"]
NUMERIC_COLS = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    *DERIVED_FEATURE_COLS,
]

COST_FN = 10.0
COST_FP = 1.0

CSV_COLUMN_MAP: dict[str, str] = {
    "UID": "udi",
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature": "air_temperature_k",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature": "process_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed": "rotational_speed_rpm",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque": "torque_nm",
    "Torque [Nm]": "torque_nm",
    "Tool wear": "tool_wear_min",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

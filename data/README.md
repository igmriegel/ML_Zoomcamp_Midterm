# AI4I 2020 Predictive Maintenance Dataset

- **Source:** [UCI Machine Learning Repository (ID 601)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
- **License:** CC BY 4.0 (Matzka, S., 2020)
- **Rows / Columns:** 10,000 rows, 14 columns (`ai4i2020.csv`)

## Data Dictionary

| Raw Column | Normalized Name | Description |
|---|---|---|
| `UID` / `UDI` | `udi` | Unique row identifier (1 to 10,000) |
| `Product ID` | `product_id` | Quality variant (`L`/`M`/`H`) + serial number |
| `Type` | `type` | Product quality variant (`L`: 50%, `M`: 30%, `H`: 20%) |
| `Air temperature` | `air_temperature_k` | Ambient temperature [K] |
| `Process temperature` | `process_temperature_k` | Process temperature [K] |
| `Rotational speed` | `rotational_speed_rpm` | Spindle rotational speed [rpm] |
| `Torque` | `torque_nm` | Torque [Nm] |
| `Tool wear` | `tool_wear_min` | Cumulative tool wear [min] |
| `Machine failure` | `machine_failure` | Binary target (`1` = failure, `0` = normal) |
| `TWF` | `twf` | Tool wear failure mode |
| `HDF` | `hdf` | Heat dissipation failure mode |
| `PWF` | `pwf` | Power failure mode |
| `OSF` | `osf` | Overstrain failure mode |
| `RNF` | `rnf` | Random failure mode (excluded from multi-label target) |

## Known Label Inconsistencies
- 9 rows have `machine_failure = 1` with all five failure mode flags equal to `0`.
- 18 rows have `rnf = 1` with `machine_failure = 0`.
- Following the project plan, the original `machine_failure` label is preserved as-is, and these rows are documented in the EDA.

## Re-downloading
```bash
uv run python scripts/download_data.py
```

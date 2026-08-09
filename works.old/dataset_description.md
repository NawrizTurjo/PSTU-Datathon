# 📁 Dataset Description

## 📂 Files

- **`train.csv`**: The training set containing historical sensor data, maintenance logs, and the target variable.
- **`test.csv`**: The test set containing the same features as the training set, but without the target variable. You must predict the failure probability along with binary (`0`, `1`) predictions for these stations.

---

## 📊 Columns

The dataset contains a mix of categorical, numerical, and time-series aggregated features.

### 🏛️ Base Attributes
- **`base_number_of_dependent_farmers`**: Number of farmers relying on this station.
- **`base_station_installation_age_years`**: Age of the station in years.
- **`base_distance_from_coastal_river_km`**: Distance from the main river estuary.

### 📡 Real-time Sensor Readings
- **`sensor_current_battery_voltage_volts`**, **`sensor_ambient_temperature_celsius`**, **`sensor_water_salinity_ppm`**, etc.: Current readings from IoT sensors deployed at the station.

### ⚙️ Operational Counts & Financial Metrics
- **`count_dry_run_events`**, **`count_voltage_surge_events`**: Historical count of specific anomalies.
- **`cost_commercial_maintenance_bdt`**, **`cost_govt_grant_bdt`**: Financial logs associated with the station.

### 🎯 Target Variable
- **`Your_Target_Column`**: *(Train set only)* Binary indicator (`1` = Critical Failure within 7 days, `0` = Normal Operation).

---

## 🔮 What am I predicting?

You are predicting the **probability** (a float value between `0.0` and `1.0`) that a given station will experience a critical failure in the next 7 days, along with binary `0`/`1` predictions.

---

## 📝 Submission Format (STRICTLY ENFORCED)

Your submission file must exactly match the format below. It must contain a header and exactly three columns in this specific order:

1. **`id`**: The row index, starting from `0` and incrementing by `1`.
2. **`Target_Binary`**: Your predicted binary classification (`0` for Normal, `1` for Critical Failure).
3. **`Target_Probability`**: The predicted probability of failure (a float value strictly between `0.0` and `1.0`). This is mandatory for the ROC-AUC calculation.

### 💡 Submission Example (`submission.csv`)

```csv
id,Target_Binary,Target_Probability
0,0,0.0123
1,1,0.8745
2,0,0.0056
```

> [!WARNING]
> ### ⚠️ STRICT FORMAT VALIDATION WARNING
>
> The evaluation script will automatically reject any submission that:
> - Does not have the exact column names: `['id', 'Target_Binary', 'Target_Probability']`.
> - Contains missing (`NaN`) or infinite values in the `Target_Probability` column.
> - Contains probability values outside the `0.0` to `1.0` range.
> - Has a different number of rows than the `test.csv` file.
>
> **If your file violates any of these rules, you will receive an "Evaluation Error" and your score will not be calculated.**

# Dataset instructions

Download the PEMS-BAY traffic speed dataset from the DCRNN benchmark distribution
or use the copy already obtained for coursework. Place the converted CSV at:

`data/PEMS-BAY.csv`

The project expects either:
- wide format: timestamp + one numeric column per sensor, or
- long format: timestamp + sensor_id + speed.

For HDF5 conversion, inspect the file first. A common conversion pattern is:

```python
import pandas as pd
df = pd.read_hdf("pems-bay.h5")
print(df.shape, df.head())
df.to_csv("data/PEMS-BAY.csv", index_label="timestamp")
```

The source dataset contains measured traffic speeds. Artificial missingness and
drift in this assignment are controlled experimental perturbations and must not
be confused with the original data's naturally occurring missing values.

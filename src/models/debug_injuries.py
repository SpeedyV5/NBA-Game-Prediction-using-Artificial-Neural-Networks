import pandas as pd
from pathlib import Path

p = Path("data_raw/injury_reports_raw/injuries_latest.csv")
inj = pd.read_csv(p, low_memory=False)
print("shape:", inj.shape)
print("cols:", list(inj.columns))
print(inj.head(20))

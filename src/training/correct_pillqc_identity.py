from pathlib import Path
import pandas as pd
from datetime import datetime

path = Path("datasets/external/pillQC_manifest.csv")

if not path.exists():
    raise SystemExit("ERROR: PillQC manifest not found.")

df = pd.read_csv(path, dtype=str).fillna("")

backup = path.with_name("pillQC_manifest_before_identity_correction.csv")

if backup.exists():
    raise SystemExit("Backup already exists. No files changed.")

df.to_csv(backup, index=False, encoding="utf-8-sig")

tablet = df["task"] == "tablet"

# Do not claim physical sample IDs that have not been individually verified.
df.loc[tablet, "physical_sample_id"] = ""

# Record the visual assessment and its limitation.
df.loc[tablet, "notes"] = (
    df.loc[tablet, "notes"].astype(str)
    + " User visual assessment: images appear to show different physical "
      "tablets of the same design; individual physical identities are "
      "not independently documented."
)

df.to_csv(path, index=False, encoding="utf-8-sig")

print("========== MANIFEST CORRECTED ==========")
print("Backup:", backup)
print("Updated manifest:", path)
print("Tablet records:", int(tablet.sum()))
print("Physical sample IDs assigned:", 
      int((df.loc[tablet, "physical_sample_id"] != "").sum()))
print("\nCategory counts:")
print(df.groupby(["task", "label", "defect_type"]).size().to_string())
print("\nOriginal PharmaSight AI manifest was not changed.")

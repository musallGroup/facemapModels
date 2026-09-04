from pathlib import Path
import pandas as pd
import deeplabcut

PROJECT = Path(__file__).resolve().parent
CONFIG = PROJECT / "config.yaml"
SCORER = "LabelForge"

def convert_label_csvs_to_h5():
    labeled_data = PROJECT / "labeled-data"
    csv_files = sorted(labeled_data.glob(f"*/CollectedData_{SCORER}.csv"))
    if not csv_files:
        raise RuntimeError(f"No CollectedData_{SCORER}.csv files found.")

    print(f"Found {len(csv_files)} labeled-data folder(s).")
    for csv_path in csv_files:
        print(f"\nReading: {csv_path}")
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=[0, 1, 2])
        h5_path = csv_path.with_suffix(".h5")
        df.to_hdf(h5_path, key="df_with_missing", mode="w")

        check = pd.read_hdf(h5_path, key="df_with_missing")
        if list(df.index) != list(check.index):
            raise RuntimeError(f"H5 row index mismatch: {h5_path}")
        if list(df.columns) != list(check.columns):
            raise RuntimeError(f"H5 column mismatch: {h5_path}")

        print(f"✓ H5 created and re-opened: {h5_path.name}")

def main():
    print("=== LabelForge -> DeepLabCut real smoke test ===")
    print(f"Project: {PROJECT}")
    print(f"Config:  {CONFIG}\n")

    if not CONFIG.is_file():
        raise RuntimeError(f"config.yaml not found: {CONFIG}")

    print(f"DeepLabCut version: {deeplabcut.__version__}")

    print("\nSTEP 1/2 - Create DLC H5 label files")
    convert_label_csvs_to_h5()

    print("\nSTEP 2/2 - Ask DeepLabCut to create the training dataset")
    print("No network training will be started.\n")
    deeplabcut.create_training_dataset(str(CONFIG))

    print("\n==============================================")
    print("LABELFORGE -> DEEPLABCUT READY TO TRAIN")
    print("==============================================")
    print("DeepLabCut accepted the LabelForge project and")
    print("successfully created its training dataset.")
    print("No model training was started.")

if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import kaggle
import zipfile
import os


def prepare_data():
    kaggle.api.competition_download_files("spaceship-titanic")
    # unzip the data
    with zipfile.ZipFile("spaceship-titanic.zip", "r") as zip_ref:
        zip_ref.extractall()
    # remove the zip file
    os.remove("spaceship-titanic.zip")


def split_data(public: Path, private: Path):
    df = pd.read_csv("train.csv")
    # Use a fixed random_state for reproducibility
    new_train, new_test = train_test_split(df, test_size=0.1, random_state=0)

    os.makedirs(public, exist_ok=True)
    os.makedirs(private, exist_ok=True)

    example_submission = new_test[["PassengerId", "Transported"]].copy()
    example_submission["Transported"] = False
    example_submission.to_csv(public / "sample_submission.csv", index=False)

    new_train.to_csv(public / "train.csv", index=False)
    print("training sample shape:", new_train.shape)
    new_test.to_csv(private / "test.csv", index=False)
    print("test sample shape:", new_test.shape)
    print(f"Validation data saved to {public / 'test.csv'}")
    new_test.drop("Transported", axis="columns").to_csv(public / "test.csv", index=False)

    # remove the previous files
    os.remove("train.csv")
    os.remove("sample_submission.csv")


def setup_data():
    # download the data
    prepare_data()

    # Get the directory where the script is located
    script_dir = Path(__file__).resolve().parent
    public_path = script_dir / "public"
    private_path = script_dir / "private"

    # split the data
    split_data(public_path, private_path)


if __name__ == "__main__":
    setup_data()

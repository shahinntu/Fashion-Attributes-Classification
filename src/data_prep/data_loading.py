import os

import pandas as pd
from datasets import Dataset, DatasetDict

from src import load_txt_to_list


def create_hf_train_dataset(data_dir):
    dataset_dict = {}

    for mode in ["train", "val"]:
        image_subpath_list = load_txt_to_list(
            os.path.join(data_dir, f"split/{mode}.txt")
        )
        attr_list = load_txt_to_list(os.path.join(data_dir, f"split/{mode}_attr.txt"))

        assert len(image_subpath_list) == len(attr_list), f"Length mismatch for {mode}"

        data = []

        for image_subpath, attr in zip(image_subpath_list, attr_list):
            image_path = os.path.join(data_dir, image_subpath.strip())
            label = list(map(int, attr.split()))

            if os.path.exists(image_path):
                data.append({"image_path": image_path, "label": label})

        data_df = pd.DataFrame(data)
        hf_dataset = Dataset.from_pandas(data_df)

        dataset_dict[mode] = hf_dataset

    return DatasetDict(dataset_dict)


def create_hf_test_dataset(data_dir):
    dataset_dict = {}

    for mode in ["test"]:
        image_subpath_list = load_txt_to_list(
            os.path.join(data_dir, f"split/{mode}.txt")
        )

        data = []

        for image_subpath in image_subpath_list:
            image_path = os.path.join(data_dir, image_subpath.strip())

            if os.path.exists(image_path):
                data.append({"image_path": image_path})

        data_df = pd.DataFrame(data)
        hf_dataset = Dataset.from_pandas(data_df)

        dataset_dict[mode] = hf_dataset

    return DatasetDict(dataset_dict)

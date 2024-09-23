import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class FashionNetDataset(Dataset):
    def __init__(self, hf_dataset, transforms):
        self._hf_dataset = hf_dataset
        self._transforms = transforms

    def __len__(self):
        return len(self._hf_dataset)

    def __getitem__(self, index):
        data = self._hf_dataset[index]
        image = Image.open(data["image_path"])
        label = (
            np.array(data["label"])
            if "label" in self._hf_dataset.column_names
            else None
        )

        if label is not None:
            return self._transforms(image), label

        return self._transforms(image)

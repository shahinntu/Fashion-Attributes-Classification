import os

from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from src import FashionNetDataset
from src import Query2LabelNet
from src import load_checkpoint, Params


class Predictor:
    def __init__(self, model, config, device):
        self._model = model
        self._config = config
        self._device = device

        self._model = self._model.to(self._device)
        self._model.eval()

    @classmethod
    def from_pretrained(cls, model_log_dir, restore_version, device):
        config_path = os.path.join(
            model_log_dir, restore_version, "hyper_params/params.json"
        )
        model_state_path = os.path.join(
            model_log_dir, restore_version, "state/model_best.pth"
        )

        config = Params(config_path)
        model = Query2LabelNet(config)

        load_checkpoint(model_state_path, model)

        return cls(model, config, device)

    @torch.no_grad()
    def predict(self, hf_dataset):
        dataloader = self._get_dataloader(hf_dataset)

        predictions = []

        with tqdm(total=len(dataloader), desc="Predicting") as t:
            for images in dataloader:
                images = images.to(self._device)
                model_pred = self._model(images)
                batch_prediction = torch.stack(
                    [label_pred.argmax(-1) for label_pred in model_pred], dim=1
                )
                predictions.append(batch_prediction)
                t.update()

        predictions = torch.cat(predictions, dim=0)

        return predictions

    def _get_dataloader(self, hf_dataset):
        pred_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION,
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION,
                    )
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    self._config.DATA_AUGMENTATION.NORM_MEAN,
                    self._config.DATA_AUGMENTATION.NORM_STD,
                ),
            ]
        )

        pred_dataset = FashionNetDataset(hf_dataset, pred_transforms)
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TEST,
            shuffle=False,
        )

        return pred_dataloader

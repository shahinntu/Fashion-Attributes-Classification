from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler

from src import (
    create_hf_train_dataset,
    FashionNetDataset,
    Mixup,
    Query2LabelNet,
    MultiLabelCrossEntropyLoss,
    AvgClassAccuracy,
    Trainer,
    Params,
    get_label_class_weights,
)


class MLPipeline:
    def __init__(self, args):
        self._config = Params(args.config_path)
        self._train_dataloader, self._val_dataloader = self._prepare_data(args)
        self._trainer = self._get_trainer(args)

    def run(self):
        self._trainer.train(self._train_dataloader, self._val_dataloader)

    def _prepare_data(self, args):
        hf_dataset_dict = create_hf_train_dataset(args.data_dir)
        hf_train_dataset = hf_dataset_dict["train"]
        hf_val_dataset = hf_dataset_dict["val"]

        train_transforms = transforms.Compose(
            [
                transforms.Resize(
                    (
                        self._config.DATA_AUGMENTATION.RESIZE_RESOLUTION,
                        self._config.DATA_AUGMENTATION.RESIZE_RESOLUTION,
                    )
                ),
                (
                    transforms.RandomCrop(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                    if self._config.DATA_AUGMENTATION.RANDOM_CROP
                    else transforms.CenterCrop(
                        self._config.DATA_AUGMENTATION.TARGET_RESOLUTION
                    )
                ),
                (
                    transforms.RandomHorizontalFlip()
                    if self._config.DATA_AUGMENTATION.RANDOM_HORIZONTAL_FLIP
                    else transforms.Lambda(lambda x: x)
                ),
                (
                    transforms.RandomRotation(30)
                    if self._config.DATA_AUGMENTATION.RANDOM_ROTATION
                    else transforms.Lambda(lambda x: x)
                ),
                (
                    transforms.ColorJitter()
                    if self._config.DATA_AUGMENTATION.COLOR_JITTER
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    self._config.DATA_AUGMENTATION.NORM_MEAN,
                    self._config.DATA_AUGMENTATION.NORM_STD,
                ),
            ]
        )
        train_dataset = FashionNetDataset(hf_train_dataset, train_transforms)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self._config.TRAINING.BATCH_SIZE.TRAIN,
            shuffle=True,
        )

        val_transforms = transforms.Compose(
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
        val_dataset = FashionNetDataset(hf_val_dataset, val_transforms)
        val_dataloader = DataLoader(
            val_dataset, batch_size=self._config.TRAINING.BATCH_SIZE.TEST, shuffle=False
        )

        return train_dataloader, val_dataloader

    def _get_trainer(self, args):
        hf_dataset_dict = create_hf_train_dataset(args.data_dir)
        hf_train_dataset = hf_dataset_dict["train"]

        mixup_fn = None
        mixup_active = (
            self._config.DATA_AUGMENTATION.MIXUP > 0
            or self._config.DATA_AUGMENTATION.CUTMIX > 0.0
            or self._config.DATA_AUGMENTATION.CUTMIX_MINMAX is not None
        )
        if mixup_active:
            mixup_fn = Mixup(
                self._config.MODEL.NUM_LABELS_LIST,
                mixup_alpha=self._config.DATA_AUGMENTATION.MIXUP,
                cutmix_alpha=self._config.DATA_AUGMENTATION.CUTMIX,
                cutmix_minmax=self._config.DATA_AUGMENTATION.CUTMIX_MINMAX,
                prob=self._config.DATA_AUGMENTATION.MIXUP_PROB,
                switch_prob=self._config.DATA_AUGMENTATION.MIXUP_SWITCH_PROB,
                mode=self._config.DATA_AUGMENTATION.MIXUP_MODE,
                label_smoothing=self._config.MODEL.LABEL_SMOOTHING,
            )

        label_class_weights = get_label_class_weights(hf_train_dataset["label"])

        model = Query2LabelNet(self._config)
        criterion = MultiLabelCrossEntropyLoss(
            label_class_weights, is_soft_target=mixup_active
        )

        optimizer = AdamW(
            model.parameters(),
            lr=self._config.TRAINING.ADAM_OPTIMIZER.LEARNING_RATE,
            betas=(
                self._config.TRAINING.ADAM_OPTIMIZER.BETA1,
                self._config.TRAINING.ADAM_OPTIMIZER.BETA2,
            ),
            weight_decay=self._config.TRAINING.ADAM_OPTIMIZER.WEIGHT_DECAY,
            eps=self._config.TRAINING.ADAM_OPTIMIZER.EPSILON,
        )
        lr_scheduler = get_scheduler(
            self._config.TRAINING.LR_SCHEDULER.TYPE,
            optimizer=optimizer,
            num_warmup_steps=round(
                self._config.TRAINING.LR_SCHEDULER.WARMUP_STEPS
                * self._config.ACCELERATOR.GRADIENT_ACCUMULATION_STEPS
            ),
            num_training_steps=round(
                self._config.TRAINING.EPOCHS
                * len(self._train_dataloader)
                / self._config.ACCELERATOR.GRADIENT_ACCUMULATION_STEPS
            ),
        )

        metrics = {
            "avg_class_accuracy": AvgClassAccuracy(self._config.MODEL.NUM_LABELS_LIST)
        }
        objective = "avg_class_accuracy"

        trainer = Trainer(
            model,
            criterion,
            optimizer,
            lr_scheduler,
            self._config,
            args.model_log_dir,
            metrics=metrics,
            objective=objective,
            mixup_fn=mixup_fn,
        )

        return trainer

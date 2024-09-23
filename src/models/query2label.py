from torch import nn
from transformers import AutoModel

from src import PositionEmbeddingSine
from src import Transformer


class Query2LabelNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._backbone = AutoModel.from_pretrained(config.MODEL.BACKBONE_MODEL_NAME)
        self._transformer = Transformer(
            config.MODEL.TRANSFORMER.HIDDEN_DIM,
            config.MODEL.TRANSFORMER.N_HEADS,
            config.MODEL.TRANSFORMER.ENC_LAYERS,
            config.MODEL.TRANSFORMER.DEC_LAYERS,
            config.MODEL.TRANSFORMER.FEED_FWD_DIM,
            dropout=config.MODEL.TRANSFORMER.DROPOUT,
        )
        self._backbone_fmap_size = int(
            config.DATA_AUGMENTATION.TARGET_RESOLUTION
            / (
                self._backbone.config.patch_size
                * 2 ** (len(self._backbone.config.depths) - 1)
            )
        )
        self._position_encoder = PositionEmbeddingSine(
            self._transformer.d_model / 2,
            normalize=True,
            maxH=self._backbone_fmap_size,
            maxW=self._backbone_fmap_size,
        )
        self._input_proj = nn.Conv2d(
            self._backbone.config.hidden_size, self._transformer.d_model, kernel_size=1
        )
        self._query_embed = nn.Embedding(
            len(config.MODEL.NUM_LABELS_LIST), self._transformer.d_model
        )

        self._classifiers = nn.ModuleList(
            [
                nn.Linear(self._transformer.d_model, num_labels)
                for num_labels in config.MODEL.NUM_LABELS_LIST
            ]
        )

    def forward(self, x):
        backbone_features = self._backbone(x).last_hidden_state
        batch_size, _, channel_dim = backbone_features.shape
        backbone_features = backbone_features.transpose(1, 2).reshape(
            batch_size, channel_dim, self._backbone_fmap_size, self._backbone_fmap_size
        )

        pos_embedding = self._position_encoder(backbone_features)
        query_input = self._query_embed.weight

        hidden_states = self._transformer(
            self._input_proj(backbone_features), query_input, pos_embedding
        )[0][-1]

        logits_list = [
            classifier(hidden_states[:, i, :])
            for i, classifier in enumerate(self._classifiers)
        ]
        return logits_list

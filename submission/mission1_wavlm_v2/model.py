"""Mission 1 WavLM gender classifier, matching the executed training notebook."""

import torch
from torch import nn
from transformers import WavLMConfig, WavLMModel


class GenderModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        if isinstance(config, dict):
            config = WavLMConfig.from_dict(config)
        self.backbone = WavLMModel(config)
        size = config.hidden_size
        self.layer_weights = nn.Parameter(torch.zeros(config.num_hidden_layers + 1))
        self.attention = nn.Sequential(
            nn.Linear(size, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(2 * size),
            nn.Dropout(0.2),
            nn.Linear(2 * size, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

    def forward(self, input_values, attention_mask):
        states = self.backbone(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).hidden_states
        x = sum(w * h for w, h in zip(self.layer_weights.softmax(0), states))
        mask = self.backbone._get_feature_vector_attention_mask(
            x.shape[1], attention_mask
        )
        score = self.attention(x).squeeze(-1).float().masked_fill(~mask, float("-inf"))
        weights = score.softmax(-1).unsqueeze(-1)
        x = x.float()
        mean = (weights * x).sum(1)
        var = (weights * (x - mean[:, None]).square()).sum(1)
        return self.classifier(
            torch.cat((mean, var.clamp_min(1e-5).sqrt()), -1)
        )


def load_checkpoint(path, device):
    """Restore the self-contained checkpoint without downloading pretrained files."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("preprocessing_version") != 1:
        raise ValueError("Unsupported checkpoint preprocessing version")
    if payload.get("label_map") != {"F": 0, "M": 1}:
        raise ValueError("Unsupported checkpoint label mapping")
    model = GenderModel(payload["backbone_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval(), payload

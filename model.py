"""Member 3: degradation-aware lightweight Swin image restoration model.

This module intentionally depends only on PyTorch and the Python standard
library.  It contains the complete model definition so that it can be copied
into a standalone Google Colab without requiring timm or torchvision.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODEL_NAME = "Degradation-aware Swin"
INPUT_SHAPE = (1, 128, 128)
OUTPUT_SHAPE = (1, 256, 256)


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    """Partition a channels-last image into non-overlapping windows."""
    b, h, w, c = x.shape
    if h % window_size or w % window_size:
        raise ValueError(
            f"Spatial size {(h, w)} must be divisible by window size "
            f"{window_size}."
        )
    x = x.view(
        b,
        h // window_size,
        window_size,
        w // window_size,
        window_size,
        c,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, c)
    )


def _window_reverse(
    windows: Tensor, window_size: int, height: int, width: int
) -> Tensor:
    """Reverse :func:`_window_partition` into a channels-last image."""
    windows_per_image = (height // window_size) * (width // window_size)
    if windows.shape[0] % windows_per_image:
        raise ValueError("Window batch cannot be reversed to the requested size.")
    b = windows.shape[0] // windows_per_image
    x = windows.view(
        b,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(b, height, width, -1)
    )


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, probability: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("Drop-path probability must be in [0, 1).")
        self.probability = float(probability)

    def forward(self, x: Tensor) -> Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        return x * random_tensor.floor() / keep_probability


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return self.dropout2(x)


class WindowAttention(nn.Module):
    """Window multi-head self-attention with learned relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("Feature dimension must be divisible by head count.")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        relative_positions = (2 * window_size - 1) ** 2
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_positions, num_heads)
        )

        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(window_size),
                torch.arange(window_size),
                indexing="ij",
            )
        )
        coordinates_flat = coordinates.flatten(1)
        relative_coordinates = (
            coordinates_flat[:, :, None] - coordinates_flat[:, None, :]
        )
        relative_coordinates = relative_coordinates.permute(1, 2, 0).contiguous()
        relative_coordinates[:, :, 0] += window_size - 1
        relative_coordinates[:, :, 1] += window_size - 1
        relative_coordinates[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coordinates.sum(-1)
        self.register_buffer(
            "relative_position_index", relative_position_index, persistent=True
        )

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        b_windows, token_count, channels = x.shape
        expected_tokens = self.window_size * self.window_size
        if token_count != expected_tokens or channels != self.dim:
            raise ValueError(
                f"Expected window tokens [B,{expected_tokens},{self.dim}], "
                f"received {tuple(x.shape)}."
            )

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            b_windows, token_count, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = query * self.scale
        attention = query @ key.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_bias = relative_bias.view(
            token_count, token_count, self.num_heads
        ).permute(2, 0, 1)
        attention = attention + relative_bias.to(dtype=attention.dtype).unsqueeze(0)

        if attention_mask is not None:
            number_of_windows = attention_mask.shape[0]
            if b_windows % number_of_windows:
                raise ValueError("Attention windows and shifted-window mask disagree.")
            attention = attention.view(
                b_windows // number_of_windows,
                number_of_windows,
                self.num_heads,
                token_count,
                token_count,
            )
            attention = attention + attention_mask.to(
                device=attention.device, dtype=attention.dtype
            ).unsqueeze(0).unsqueeze(2)
            attention = attention.view(
                -1, self.num_heads, token_count, token_count
            )

        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        x = (attention @ value).transpose(1, 2).reshape(
            b_windows, token_count, channels
        )
        x = self.projection(x)
        return self.projection_dropout(x)


class FiLMSwinBlock(nn.Module):
    """A Swin block with a separate identity-safe FiLM map per block."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int,
        shift_size: int,
        condition_dim: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if shift_size not in (0, window_size // 2):
            raise ValueError("Swin shifts must be 0 or half the window size.")
        height, width = input_resolution
        if height % window_size or width % window_size:
            raise ValueError("Input resolution must divide evenly into windows.")

        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.film = nn.Linear(condition_dim, 2 * dim)
        # gamma is used as (1 + delta_gamma), and a zero projection makes the
        # complete FiLM operation exactly the identity at initialization.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(
            dim=dim,
            window_size=window_size,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout=dropout)

        mask = self._make_attention_mask() if shift_size else None
        self.register_buffer("attention_mask", mask, persistent=False)

    def _make_attention_mask(self) -> Tensor:
        height, width = self.input_resolution
        mask = torch.zeros((1, height, width, 1))
        window_size = self.window_size
        shift_size = self.shift_size
        height_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        width_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        region = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                mask[:, height_slice, width_slice, :] = region
                region += 1

        mask_windows = _window_partition(mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attention_mask.masked_fill(
            attention_mask != 0, float(-100.0)
        ).masked_fill(attention_mask == 0, float(0.0))

    def forward(self, x: Tensor, degradation_vector: Tensor) -> Tensor:
        b, height, width, channels = x.shape
        if (height, width) != self.input_resolution or channels != self.dim:
            raise ValueError(
                f"Expected Swin feature [B,{self.input_resolution[0]},"
                f"{self.input_resolution[1]},{self.dim}], got {tuple(x.shape)}."
            )
        if degradation_vector.shape != (b, self.dim):
            raise ValueError(
                f"Expected degradation vector {(b, self.dim)}, got "
                f"{tuple(degradation_vector.shape)}."
            )

        shortcut = x
        delta_gamma, beta = self.film(degradation_vector).chunk(2, dim=-1)
        conditioned = x * (1.0 + delta_gamma[:, None, None, :])
        conditioned = conditioned + beta[:, None, None, :]
        conditioned = self.norm1(conditioned)

        if self.shift_size:
            conditioned = torch.roll(
                conditioned,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

        windows = _window_partition(conditioned, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, channels)
        attended_windows = self.attention(windows, self.attention_mask)
        attended_windows = attended_windows.view(
            -1, self.window_size, self.window_size, channels
        )
        attended = _window_reverse(
            attended_windows, self.window_size, height, width
        )

        if self.shift_size:
            attended = torch.roll(
                attended,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )

        x = shortcut + self.drop_path(attended)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class ResidualCNNBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.activation(self.conv1(x)))


class DegradationEncoder(nn.Module):
    """Two residual CNN blocks followed by GAP and a 48-D MLP."""

    def __init__(self, dim: int = 48, second_linear: bool = True) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualCNNBlock(dim),
            ResidualCNNBlock(dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        layers: list[nn.Module] = [nn.Linear(dim, dim), nn.GELU()]
        if second_linear:
            layers.append(nn.Linear(dim, dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.mlp(x)


class DegradationAwareSwin(nn.Module):
    """Member 3 restoration network with no degradation-order head."""

    def __init__(
        self,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        degradation_second_linear: bool = True,
    ) -> None:
        super().__init__()
        dim = 48
        depth = 6
        input_resolution = (128, 128)

        self.stem = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.content_encoder = nn.Sequential(
            ResidualCNNBlock(dim),
            ResidualCNNBlock(dim),
        )
        self.degradation_encoder = DegradationEncoder(
            dim=dim, second_linear=degradation_second_linear
        )

        drop_path_values = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.swin_blocks = nn.ModuleList(
            [
                FiLMSwinBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=4,
                    window_size=8,
                    shift_size=0 if index % 2 == 0 else 4,
                    condition_dim=dim,
                    mlp_ratio=2.0,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                    drop_path=drop_path_values[index],
                )
                for index in range(depth)
            ]
        )

        self.pre_shuffle = nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.hr_blocks = nn.Sequential(
            ResidualCNNBlock(dim),
            ResidualCNNBlock(dim),
            ResidualCNNBlock(dim),
            ResidualCNNBlock(dim),
        )
        self.reconstruction = nn.Conv2d(dim, 1, kernel_size=3, padding=1)

    @staticmethod
    def _validate_input(x: Tensor) -> None:
        if x.ndim != 4 or tuple(x.shape[1:]) != INPUT_SHAPE:
            raise ValueError(
                f"Expected input [B,1,128,128], received {tuple(x.shape)}."
            )
        if not x.is_floating_point():
            raise TypeError("Model input must be a floating-point tensor.")

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        bicubic = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False,
        )

        stem = self.stem(x)
        content = self.content_encoder(stem)
        degradation_vector = self.degradation_encoder(stem)

        features = content.permute(0, 2, 3, 1).contiguous()
        for block in self.swin_blocks:
            features = block(features, degradation_vector)
        features = features.permute(0, 3, 1, 2).contiguous()

        # Residual fusion explicitly anchors the transformer result to the stem.
        features = features + stem
        features = self.pixel_shuffle(self.pre_shuffle(features))
        features = self.hr_blocks(features)
        residual = self.reconstruction(features)
        output = bicubic + residual
        if tuple(output.shape[1:]) != OUTPUT_SHAPE:
            raise RuntimeError(f"Unexpected model output shape {tuple(output.shape)}.")
        return output


def _extract_model_config(config: Any) -> Mapping[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping (for example, a loaded YAML dict).")
    nested = config.get("model")
    return nested if isinstance(nested, Mapping) else config


def _validate_exact_architecture(config: Mapping[str, Any]) -> None:
    expected_values = {
        "in_channels": 1,
        "out_channels": 1,
        "dim": 48,
        "embed_dim": 48,
        "depth": 6,
        "num_heads": 4,
        "window_size": 8,
        "mlp_ratio": 2.0,
        "upscale": 2,
    }
    for key, expected in expected_values.items():
        if key in config and config[key] != expected:
            raise ValueError(
                f"Member 3 architecture requires {key}={expected!r}; "
                f"received {config[key]!r}."
            )


def build_model(config: Mapping[str, Any] | None = None) -> DegradationAwareSwin:
    """Build the exact six-block, 48-channel Member 3 architecture."""
    model_config = _extract_model_config(config)
    _validate_exact_architecture(model_config)
    return DegradationAwareSwin(
        dropout=float(model_config.get("dropout", 0.0)),
        attention_dropout=float(model_config.get("attention_dropout", 0.0)),
        drop_path_rate=float(model_config.get("drop_path_rate", 0.0)),
        degradation_second_linear=bool(
            model_config.get("degradation_second_linear", True)
        ),
    )


def _checkpoint_state(checkpoint: Any) -> tuple[Mapping[str, Any], Mapping[str, Tensor]]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping.")
    config = checkpoint.get("config", checkpoint.get("model_config", {}))
    if not isinstance(config, Mapping):
        config = {}

    state: Any = None
    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            state = candidate
            break
    if state is None and checkpoint and all(
        isinstance(value, Tensor) for value in checkpoint.values()
    ):
        state = checkpoint
    if not isinstance(state, Mapping):
        raise KeyError(
            "Checkpoint has no model_state_dict/state_dict/model tensor mapping."
        )

    cleaned = dict(state)
    for prefix in ("module.", "_orig_mod.", "model."):
        if cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {key[len(prefix) :]: value for key, value in cleaned.items()}
    return config, cleaned


def load_model(checkpoint_path: str, device: str | torch.device) -> DegradationAwareSwin:
    """Load an actual Member 3 checkpoint and return an eval-mode model."""
    target_device = torch.device(device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # Compatibility with older Colab PyTorch releases.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config, state = _checkpoint_state(checkpoint)
    model = build_model(config)
    model.load_state_dict(state, strict=True)
    model.to(target_device)
    model.eval()
    return model


@torch.inference_mode()
def predict(
    model: nn.Module,
    lr_tensor: Tensor,
    device: str | torch.device,
) -> Tensor:
    """Run label-free restoration and return ``[B,1,256,256]``."""
    if not isinstance(lr_tensor, Tensor):
        raise TypeError("lr_tensor must be a torch.Tensor.")
    if lr_tensor.ndim != 4 or tuple(lr_tensor.shape[1:]) != INPUT_SHAPE:
        raise ValueError(
            f"predict expects [B,1,128,128], received {tuple(lr_tensor.shape)}."
        )
    if not torch.isfinite(lr_tensor).all():
        raise FloatingPointError("Model input contains NaN or Inf values.")
    target_device = torch.device(device)
    model.to(target_device)
    model.eval()
    output = model(
        lr_tensor.to(device=target_device, dtype=torch.float32, non_blocking=True)
    )
    if isinstance(output, tuple):
        output = output[0]
    expected = (lr_tensor.shape[0],) + OUTPUT_SHAPE
    if tuple(output.shape) != expected:
        raise RuntimeError(
            f"predict expected output {expected}, received {tuple(output.shape)}."
        )
    if not torch.isfinite(output).all():
        raise FloatingPointError("Model output contains NaN or Inf values.")
    return output.detach()


__all__ = [
    "MODEL_NAME",
    "DegradationAwareSwin",
    "DegradationEncoder",
    "build_model",
    "load_model",
    "predict",
]

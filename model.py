"""Member 4: order-aware degradation-conditioned lightweight Swin.

The auxiliary order classifier is available only through ``return_aux=True``.
Normal inference remains completely label-free and returns only the restored
image.  This standalone module requires PyTorch but no external model library.
"""

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F


MODEL_NAME = "Final Order-aware Swin"
INPUT_SHAPE = (1, 128, 128)
OUTPUT_SHAPE = (1, 256, 256)
ORDER_NAMES = ("GSD", "GDS", "SGD", "SDG", "DGS", "DSG")


def _window_partition(x: Tensor, window_size: int) -> Tensor:
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
    """Window attention with custom relative-position bias."""

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
        self.register_buffer(
            "relative_position_index",
            relative_coordinates.sum(-1),
            persistent=True,
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
                f"Expected [B,{expected_tokens},{self.dim}], got {tuple(x.shape)}."
            )

        qkv = self.qkv(x).reshape(
            b_windows, token_count, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)

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

        attention = self.attention_dropout(F.softmax(attention, dim=-1))
        x = (attention @ value).transpose(1, 2).reshape(
            b_windows, token_count, channels
        )
        return self.projection_dropout(self.projection(x))


class FiLMSwinBlock(nn.Module):
    """Shifted/non-shifted Swin block with its own zero-init FiLM map."""

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
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(
            dim,
            window_size,
            num_heads,
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
        window_size = self.window_size
        shift_size = self.shift_size
        image_mask = torch.zeros((1, height, width, 1))
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
                image_mask[:, height_slice, width_slice, :] = region
                region += 1
        windows = _window_partition(image_mask, window_size).view(
            -1, window_size * window_size
        )
        mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, float(-100.0)).masked_fill(
            mask == 0, float(0.0)
        )

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
        x = x * (1.0 + delta_gamma[:, None, None, :])
        x = x + beta[:, None, None, :]
        x = self.norm1(x)
        if self.shift_size:
            x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )

        windows = _window_partition(x, self.window_size).view(
            -1, self.window_size * self.window_size, channels
        )
        attended_windows = self.attention(windows, self.attention_mask).view(
            -1, self.window_size, self.window_size, channels
        )
        x = _window_reverse(attended_windows, self.window_size, height, width)
        if self.shift_size:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        x = shortcut + self.drop_path(x)
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
        return self.mlp(self.pool(self.blocks(x)).flatten(1))


class FinalOrderAwareSwin(nn.Module):
    """Final model with a masked-training-compatible six-class order head."""

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
        self.stem = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.content_encoder = nn.Sequential(
            ResidualCNNBlock(dim),
            ResidualCNNBlock(dim),
        )
        self.degradation_encoder = DegradationEncoder(
            dim, second_linear=degradation_second_linear
        )
        self.order_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, len(ORDER_NAMES)),
        )

        drop_path_values = torch.linspace(0.0, drop_path_rate, depth).tolist()
        self.swin_blocks = nn.ModuleList(
            [
                FiLMSwinBlock(
                    dim=dim,
                    input_resolution=(128, 128),
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

    def forward(
        self, x: Tensor, return_aux: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
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
        logits = self.order_head(degradation_vector) if return_aux else None

        features = content.permute(0, 2, 3, 1).contiguous()
        for block in self.swin_blocks:
            features = block(features, degradation_vector)
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features + stem
        features = self.pixel_shuffle(self.pre_shuffle(features))
        features = self.hr_blocks(features)
        output = bicubic + self.reconstruction(features)
        if tuple(output.shape[1:]) != OUTPUT_SHAPE:
            raise RuntimeError(f"Unexpected model output shape {tuple(output.shape)}.")
        if return_aux:
            if logits is None:  # Keeps static analyzers honest.
                raise RuntimeError("Auxiliary logits were not created.")
            return output, logits
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
        "num_order_classes": 6,
        "upscale": 2,
    }
    for key, expected in expected_values.items():
        if key in config and config[key] != expected:
            raise ValueError(
                f"Member 4 architecture requires {key}={expected!r}; "
                f"received {config[key]!r}."
            )


def build_model(config: Mapping[str, Any] | None = None) -> FinalOrderAwareSwin:
    """Build the exact six-block, 48-channel final architecture."""
    model_config = _extract_model_config(config)
    _validate_exact_architecture(model_config)
    return FinalOrderAwareSwin(
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


def load_model(checkpoint_path: str, device: str | torch.device) -> FinalOrderAwareSwin:
    """Load a Member 4 checkpoint, strictly, and put it in evaluation mode."""
    target_device = torch.device(device)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:
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
    """Run final image inference without an order label or auxiliary output."""
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
    "ORDER_NAMES",
    "FinalOrderAwareSwin",
    "DegradationEncoder",
    "build_model",
    "load_model",
    "predict",
]

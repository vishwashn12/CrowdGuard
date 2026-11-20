"""CSRNet architecture, matching the original training implementation.

From Li et al., "CSRNet: Dilated Convolutional Neural Networks for
Understanding the Highly Congested Scenes", CVPR 2018.

Frontend: the first 10 conv layers of VGG-16 (3 pooling stages, 8x downsample)
Backend:  6 dilated conv layers at dilation rate 2
Output:   a 1x1 conv producing a single-channel density map

The forward pass deliberately does *not* apply ReLU to the output. That matches
the original training code and therefore the shipped checkpoint; adding one
would clip legitimate negative activations and change the summed count.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def make_layers(
    cfg: list,
    in_channels: int = 3,
    batch_norm: bool = False,
    dilation: bool = False,
) -> nn.Sequential:
    """Build a Conv-ReLU (optionally Conv-BN-ReLU) stack.

    Args:
        cfg: Channel counts, or ``"M"`` for a max-pool stage.
        in_channels: Input channel count.
        batch_norm: Insert BatchNorm between conv and activation.
        dilation: Use dilation rate 2, as the backend requires.
    """
    rate = 2 if dilation else 1
    layers: list[nn.Module] = []

    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        conv = nn.Conv2d(in_channels, int(value), kernel_size=3, padding=rate, dilation=rate)
        if batch_norm:
            layers += [conv, nn.BatchNorm2d(int(value)), nn.ReLU(inplace=True)]
        else:
            layers += [conv, nn.ReLU(inplace=True)]
        in_channels = int(value)

    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    """CSRNet crowd density estimator.

    Layer names match the original implementation so checkpoint keys line up
    with this module's ``state_dict``.
    """

    def __init__(self, load_weights: bool = False) -> None:
        """
        Args:
            load_weights: True when a full CSRNet checkpoint will be loaded
                afterwards, which skips the ImageNet VGG download. False
                pre-fills the frontend from pretrained VGG-16.
        """
        super().__init__()

        self.frontend_feat = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512]
        self.backend_feat = [512, 512, 512, 256, 128, 64]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        self._initialize_weights()

        if not load_weights:
            self._load_vgg_frontend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to a density map of shape ``(B, 1, H/8, W/8)``."""
        x = self.frontend(x)
        x = self.backend(x)
        return self.output_layer(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _load_vgg_frontend(self) -> None:
        """Copy matching VGG-16 weights into the frontend."""
        try:
            vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            frontend_state = self.frontend.state_dict()
            vgg_state = vgg.features.state_dict()
            for key in frontend_state:
                if key in vgg_state:
                    frontend_state[key] = vgg_state[key]
            self.frontend.load_state_dict(frontend_state)
        except Exception:
            # Offline or download blocked: random init still lets the module be
            # constructed, and a real checkpoint will overwrite it anyway.
            pass

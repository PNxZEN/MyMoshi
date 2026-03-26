import typing as tp
import numpy as np
import torch
import torch.nn as nn

class MimiDilatedResidual(nn.Module):
    """
    A single Residual block in the SEANet architecture.
    Uses dilated convolutions to expand the receptive field, capturing wide 
    temporal contexts without increasing computational cost.
    """
    def __init__(self, dim: int, dilation: int, compress: int = 2):
        super().__init__()
        # The residual branch features a bottleneck to reduce parameters (from Demucs)
        hidden = dim // compress
        
        # We explicitly calculate causal padding: pad the left side so no future leaks.
        # Causality Math: padding = (kernel_size - 1) * dilation. 
        # Kernel size here is 3. Padding = (3 - 1) * dilation = 2 * dilation.
        self.pad = nn.ConstantPad1d((2 * dilation, 0), 0.0)
        
        # 1. Non-linear activation (ELU is standard for Encodec/Mimi)
        self.block = nn.Sequential(
            nn.ELU(alpha=1.0),
            # 2. Dilated Convolution
            self.pad,
            nn.Conv1d(dim, hidden, kernel_size=3, dilation=dilation),
            nn.ELU(alpha=1.0),
            # 3. 1x1 Convolution to restore channels
            nn.Conv1d(hidden, dim, kernel_size=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x of shape (Batch, Channels, Time)
        """
        return x + self.block(x)


class MimiEncoder(nn.Module):
    """
    SEANet Convolutional Encoder (Mimi)
    Converts 24kHz raw audio down to 12.5Hz framerate (downsampling factor of 1920).
    
    The downsampling ratios are [8, 6, 5, 4, 2] -> 8*6*5*4*2 = 1920.
    Because the module is an encoder, we reverse the list when constructing layers
    (downsampling by 2 first, then 4, 5, 6, 8) to reach the 1920 factor smoothly.
    """
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 512,  # Latent dimension (Mimi paper uses 512)
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 6, 5, 4, 2], # Product = 1920
    ):
        super().__init__()
        self.ratios = list(reversed(ratios)) # E.g., [2, 4, 5, 6, 8]
        self.hop_length = int(np.prod(self.ratios)) # 1920
        
        # Initial convolution mapping 1ch audio to n_filters
        # Causal padding for kernel 7: left pad by 6
        self.initial_pad = nn.ConstantPad1d((6, 0), 0.0)
        model = [
            self.initial_pad,
            nn.Conv1d(channels, n_filters, kernel_size=7)
        ]
        
        mult = 1
        for ratio in self.ratios:
            # 1. Residual Layers (dilation increasing exponentially: 1, 2, 4...)
            for j in range(n_residual_layers):
                dilation = 2 ** j
                model.append(MimiDilatedResidual(mult * n_filters, dilation=dilation))
            
            # 2. Downsampling Convolution
            # kernel_size = ratio * 2 ensures overlap, preventing aliasing.
            # Stride = ratio. 
            pad_val = (ratio * 2) - ratio  # basic non-causal math for symmetry offset
            model += [
                nn.ELU(alpha=1.0),
                nn.ConstantPad1d((pad_val, 0), 0.0), # Causal pad
                nn.Conv1d(
                    mult * n_filters, 
                    mult * n_filters * 2, 
                    kernel_size=ratio * 2, 
                    stride=ratio
                )
            ]
            mult *= 2
            
        # Final projection to the latent dimension (e.g. 512) before quantization
        model += [
            nn.ELU(alpha=1.0),
            nn.ConstantPad1d((6, 0), 0.0),
            nn.Conv1d(mult * n_filters, dimension, kernel_size=7)
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x: [Batch, Channels, Time] (e.g. [B, 1, 24000])
        Returns: Latent encoding: [Batch, Dimension, Time / 1920] (e.g. [B, 512, 12.5Hz])
        """
        return self.model(x)


class MimiDecoder(nn.Module):
    """
    SEANet Convolutional Decoder (Mimi)
    Converts 12.5Hz latent codes (post-quantization) back to 24kHz raw audio.
    """
    def __init__(
        self,
        channels: int = 1,
        dimension: int = 512, 
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 6, 5, 4, 2],
    ):
        super().__init__()
        self.ratios = ratios # [8, 6, 5, 4, 2] 
        self.hop_length = int(np.prod(self.ratios))
        
        mult = int(2 ** len(self.ratios)) # Max multiplier
        
        # Initial mapping from latent continuous code back to filters
        model = [
            nn.ConstantPad1d((6, 0), 0.0),
            nn.Conv1d(dimension, mult * n_filters, kernel_size=7)
        ]

        for ratio in self.ratios:
            # 1. Upsampling Transposed Convolution
            # Transpose output math: (L_in - 1)*stride - 2*padding + dilation*(kernel_size - 1) + output_padding + 1
            # We skip explicit causal trimming for education to just show the PyTorch mechanism.
            model += [
                nn.ELU(alpha=1.0),
                nn.ConvTranspose1d(
                    mult * n_filters, 
                    mult * n_filters // 2, 
                    kernel_size=ratio * 2, 
                    stride=ratio,
                    padding=ratio // 2
                )
            ]
            
            # 2. Residual Layers
            for j in range(n_residual_layers):
                dilation = 2 ** j
                model.append(MimiDilatedResidual(mult * n_filters // 2, dilation=dilation))
                
            mult //= 2
            
        # Final projection to audio channels
        model += [
            nn.ELU(alpha=1.0),
            nn.ConstantPad1d((6, 0), 0.0),
            nn.Conv1d(n_filters, channels, kernel_size=7)
        ]
        
        self.model = nn.Sequential(*model)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args: z: [Batch, Dimension, Time / 1920] (e.g. [B, 512, 12.5Hz])
        Returns: Reconstructed Audio: [Batch, Channels, Time] (e.g. [B, 1, 24000])
        """
        return self.model(z)


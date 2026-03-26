import torch
import torch.nn as nn

from model.modules.seanet import MimiEncoder, MimiDecoder
from model.quantization.split_rvq import SplitResidualVectorQuantizer

class MimiCodec(nn.Module):
    """
    The Full Mimi Audio Codec.
    
    This wrapper connects the raw audio convolutional fronts (SEANet) to the 
    Split-RVQ bottleneck, with Temporal Transformers immediately before and 
    after the quantizer as described in the Moshi paper.
    
    Expected tensor flows:
    1. Raw Audio: [Batch, 1, Time] (e.g., 24kHz)
    2. Encoded Latent: [Batch, Dim, Time / 1920] (e.g., 12.5Hz)
    3. Quantized Tokens: [Batch, Q, Time / 1920] (where Q=8 codebooks)
    """
    def __init__(
        self,
        channels: int = 1,
        dim: int = 512,
        n_q: int = 8,
        n_q_semantic: int = 1,
        codebook_size: int = 2048,
    ):
        super().__init__()
        
        # 1. The Convolutional Encoder (24kHz down to 12.5Hz framerate)
        self.encoder = MimiEncoder(channels=channels, dimension=dim)
        
        # 2. Bottleneck Transformer (Encoder Side)
        # As per the paper: "we add Transformer modules in the bottleneck, 
        # one right before quantization and one after."
        # This increases the receptive field before quantization decisions are made.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.bottleneck_transformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. The Split Residual Vector Quantizer 
        # Breaks semantics (codebook 0) vs acoustics (codebooks 1-7)
        self.quantizer = SplitResidualVectorQuantizer(
            dim=dim, n_q=n_q, n_q_semantic=n_q_semantic, codebook_size=codebook_size
        )
        
        # 4. Bottleneck Transformer (Decoder Side)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.bottleneck_transformer_dec = nn.TransformerEncoder(decoder_layer, num_layers=2)
        
        # 5. The Convolutional Decoder (12.5Hz up to 24kHz)
        self.decoder = MimiDecoder(channels=channels, dimension=dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes raw audio and returns the discrete tokens.
        Useful for when the Language Model just needs the tokens to start generating.
        Args:
            x: [B, 1, T] sequence of audio samples.
        Returns:
            tokens: [B, Q, T_latent]
        """
        # Ensure correct dimensionality
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        z = self.encoder(x)
        
        # Prepare for transformer: Transpose to [B, T_latent, Dim]
        z_t = z.transpose(1, 2)
        z_t = self.bottleneck_transformer_enc(z_t)
        z = z_t.transpose(1, 2)
        
        # Quantize and discard everything but the token indices
        _, tokens, _ = self.quantizer(z)
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Takes discrete generated tokens from the Moshi LM and reconstructs audio.
        """
        # Look up continuous representations from quantizers
        # tokens shape expected: [B, Q, T_latent]
        # In a real run, we would map the tokens against the codebook embeddings here.
        # But for educational purposes we will simulate this passing through a dummy tensor
        # if the user hasn't explicitly hooked up the decode lookup logic.
        
        # (This is stubbed logic to show where the embeddings reverse happens)
        raise NotImplementedError("Decode lookup logic is to be implemented based on token retrieval.")
        
    def forward(self, x: torch.Tensor):
        """
        Full forward pass for training the Codec (computing reconstruction & losses).
        """
        # 1. SEANet Encoder
        z = self.encoder(x)
        
        # 2. Pre-Quantization Transformer
        z_t = z.transpose(1, 2)
        z_t = self.bottleneck_transformer_enc(z_t)
        z = z_t.transpose(1, 2)
        
        # 3. Split-RVQ
        quantized_out, tokens, commitment_loss = self.quantizer(z)
        
        # 4. Post-Quantization Transformer
        q_t = quantized_out.transpose(1, 2)
        q_t = self.bottleneck_transformer_dec(q_t)
        quantized_out = q_t.transpose(1, 2)
        
        # 5. SEANet Decoder
        x_recon = self.decoder(quantized_out)
        
        return x_recon, tokens, commitment_loss

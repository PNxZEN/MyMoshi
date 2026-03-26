import torch
import time
import torch.nn as nn

"""
profile_inference.py
--------------------
This script simulates profiling the Mimi Codec inference pass. 
It explicitly outlines the bottleneck downsampling shape transformations and
prints the latency and memory dimensions for education purposes.

Why Profile?
Raw audio at 24kHz equates to 24,000 floats per second. Processing this sequentially
is far too expensive in memory and attention calculation space for a Transformer (`O(N^2)`). 
The Mimi codec solves this by aggressively downsampling audio into a latent shape.

Let's verify these tensor behaviors.
"""

class DummyMimiEncoder(nn.Module):
    """
    Mock representation of the SEANet encoder.
    The true downsample ratios are: 8, 6, 5, 4, 2
    Total downsample = 8 * 6 * 5 * 4 * 2 = 1920
    """
    def __init__(self, in_channels=1, hidden_dim=512):
        super().__init__()
        # In reality, this is a sequence of 1D Convolutions with specific strides.
        # We simulate the exact mathematical compression here to validate tensor shapes.
        self.downsample_factor = 1920
        self.hidden_dim = hidden_dim
        
        # A mock linear projection just to change channels
        self.proj = nn.Conv1d(in_channels, hidden_dim, kernel_size=1)
        
    def forward(self, x):
        # x is continuous audio: [Batch, 1 Channel, Sequence_Length]
        B, C, L = x.shape
        # Change channels from 1 -> 512
        x = self.proj(x)
        # Simulate the downsampling through Strided Convs using average pooling
        x = nn.functional.avg_pool1d(x, kernel_size=self.downsample_factor)
        return x


def profile_mimi_codec():
    print("Profiling Educational Mimi Codec")
    print("================================\n")
    
    # Environment Setup
    batch_size = 1
    sample_rate = 24000 # 24kHz
    duration_seconds = 2.5 # Simulate 2.5 seconds of audio
    seq_len = int(sample_rate * duration_seconds)
    channels = 1
    
    print(f"Goal: Compress {duration_seconds} seconds of audio via SEANet.")
    
    # 1. Create Raw Audio Tensor
    # Shape: [Batch, Channels, Time Samples]
    raw_audio = torch.randn(batch_size, channels, seq_len)
    print(f"[Input] Raw Audio Shape: {list(raw_audio.shape)} -> ({raw_audio.shape[-1]} continuous floats)")
    
    # 2. Init Mock Encoder
    encoder = DummyMimiEncoder(in_channels=channels, hidden_dim=512)
    encoder.eval()
    
    # 3. Profile Latency & Shapes
    print("\nStarting forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        latent_representation = encoder(raw_audio)
        
    inference_time = (time.time() - start_time) * 1000 # ms
    
    # 4. Result Analysis
    # The new sequence length should be exactly: original_length // 1920
    expected_latent_len = seq_len // 1920
    frames_per_sec = expected_latent_len / duration_seconds
    
    print(f"[Output] Latent Audio Shape: {list(latent_representation.shape)}")
    print(f"\nAnalysis:")
    print(f" -> Hidden Channel Size: {latent_representation.shape[1]}")
    print(f" -> Latent Time Steps: {latent_representation.shape[2]}")
    print(f" -> Frequency Frame Rate: {frames_per_sec} Hz")
    
    print(f"\nLatency Profiling Results:")
    print(f" - End-to-end continuous to latent pass took {inference_time:.2f} ms")
    print(f" - Successfully squashed 24,000 Hz into discrete {frames_per_sec} Hz Transformer blocks!")

if __name__ == "__main__":
    profile_mimi_codec()
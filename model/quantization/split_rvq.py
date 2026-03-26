import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Core Vector Quantizer.
    Takes a continuous vector and snaps it to the nearest vector in a learned codebook.
    """
    def __init__(self, dim: int, codebook_size: int = 2048):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        
        # The learned discrete tokens (embeddings)
        self.embedding = nn.Embedding(codebook_size, dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, x: torch.Tensor):
        """
        Args: x of shape (Batch, Channels/Dim, Time)
        """
        # Reshape to (Batch * Time, Channels) to compute distances
        b, c, t = x.shape
        x_flat = x.transpose(1, 2).contiguous().view(-1, c)
        
        # Calculate L2 distances between x and all items in the codebook
        # d^2(x, y) = ||x||^2 - 2(x^T)y + ||y||^2
        distances = (
            torch.sum(x_flat**2, dim=1, keepdim=True)
            - 2 * torch.matmul(x_flat, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )
        
        # Get the index of the closest embedding
        indices = torch.argmin(distances, dim=1)
        
        # Retrieve the closest continuous vector
        quantized_flat = self.embedding(indices)
        quantized = quantized_flat.view(b, t, c).transpose(1, 2)
        indices = indices.view(b, 1, t)
        
        # Straight-Through Estimator (STE) trick
        # This allows gradients to flow backwards through the non-differentiable argmin operation.
        # During the forward pass, it acts as the quantized output. 
        # During backward pass, the gradient of the loss with respect to quantized is passed directly to x.
        quantized_out = x + (quantized - x).detach()
        
        # Commitment loss: penalize distance between raw vector and codebook vector
        commitment_loss = F.mse_loss(quantized.detach(), x)
        
        return quantized_out, indices, commitment_loss

class SplitResidualVectorQuantizer(nn.Module):
    """
    The novel Split-RVQ introduced in the Mimi / Moshi paper.
    
    Instead of a single stream of quantization, it splits into:
    1. Semantic Quantizers: Distilled from WavLM, capturing actual linguistic content without speaker identity.
    2. Acoustic Quantizers: Captures prosody, timber, and environmental noise via adversarial reconstruction.
    """
    def __init__(
        self,
        dim: int = 512,
        n_q: int = 8,
        n_q_semantic: int = 1,
        codebook_size: int = 2048,
    ):
        super().__init__()
        self.n_q = n_q
        self.n_q_semantic = n_q_semantic
        self.n_q_acoustic = n_q - n_q_semantic
        
        # Specifically split the lists mentally and structurally for the user
        self.semantic_quantizers = nn.ModuleList([
            VectorQuantizer(dim, codebook_size) for _ in range(self.n_q_semantic)
        ])
        
        self.acoustic_quantizers = nn.ModuleList([
            VectorQuantizer(dim, codebook_size) for _ in range(self.n_q_acoustic)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args: 
            x: continuous representations from the encoder, shape [B, D, T]
        Returns:
            quantized_out: the sum of standard continuous representations [B, D, T]
            all_indices: stack of discrete tokens for both semantic and acoustic [B, Q, T]
        """
        residual = x
        quantized_out = 0.0
        
        all_indices = []
        total_commitment_loss = 0.0
        
        # 1. First pass through SEMANTIC codebooks
        for quantizer in self.semantic_quantizers:
            q_out, indices, loss = quantizer(residual)
            quantized_out += q_out
            residual = residual - q_out # Subtract to pass the "leftover" error up to the next level
            all_indices.append(indices)
            total_commitment_loss += loss
            
        # 2. Sequential passes through ACOUSTIC codebooks
        for quantizer in self.acoustic_quantizers:
            q_out, indices, loss = quantizer(residual)
            quantized_out += q_out
            residual = residual - q_out # Standard RVQ cascading logic
            all_indices.append(indices)
            total_commitment_loss += loss
            
        all_indices = torch.cat(all_indices, dim=1) # Shape: [B, n_q, T]
        
        return quantized_out, all_indices, total_commitment_loss

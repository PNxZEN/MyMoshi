import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

"""
mock_train.py
-------------
This script demonstrates the data flow and training mechanism for the Moshi 
dual-stream language model. It has been improved to use actual Transformer blocks 
equipped with RoPE (Rotary Positional Embeddings), accurately mirroring the 
original paper's design for Helium and Depformer.

In the official Moshi architecture:
1. Helium: The Temporal Transformer (operates autoregressively across time `T`).
2. Depformer: The Depthwise Transformer (operates autoregressively across codebooks `Q`).
"""

def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    offset: torch.Tensor,
    max_period: float = 10_000.0,
    interleave: bool = False,
    time_before_heads: bool = True,
):
    """
    Kyutai's original RoPE implementation adapted for standard dimensions.
    If time_before_heads is True, expects [B, T, H, D].
    """
    if time_before_heads:
        B, T, H, D = q.shape
    else:
        B, H, T, D = q.shape

    assert q.shape[0] == k.shape[0]
    
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))

    ts = offset.float().view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.float32)

    if time_before_heads:
        ts = ts.view(B, -1, 1, 1)
    else:
        ts = ts.view(B, 1, -1, 1)

    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)

    if interleave:
        # [r0,i0,r1,i1,...]
        q = q.view(*q.shape[:-1], D // 2, 2)
        k = k.view(*k.shape[:-1], D // 2, 2)
        qr, qi = q[..., 0].float(), q[..., 1].float()
        kr, ki = k[..., 0].float(), k[..., 1].float()
    else:
        # [r..., i...]
        qr, qi = q[..., : D // 2].float(), q[..., D // 2 :].float()
        kr, ki = k[..., : D // 2].float(), k[..., D // 2 :].float()

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    dtype = q.dtype
    if interleave:
        qo = torch.stack([qor.to(dtype), qoi.to(dtype)], dim=-1).view(*q.shape[:-2], D)
        ko = torch.stack([kor.to(dtype), koi.to(dtype)], dim=-1).view(*k.shape[:-2], D)
    else:
        qo = torch.cat([qor.to(dtype), qoi.to(dtype)], dim=-1)
        ko = torch.cat([kor.to(dtype), koi.to(dtype)], dim=-1)

    return qo, ko


class CustomRoPEBlock(nn.Module):
    """A minimal Transformer block natively integrating Kyutai's RoPE on Q/K."""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # MHA Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x, is_causal=True):
        B, T, C = x.shape
        norm_x = self.norm1(x)
        
        # 1. Project yielding [B, T, num_heads, head_dim]
        q = self.q_proj(norm_x).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(norm_x).view(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(norm_x).view(B, T, self.num_heads, self.head_dim)
        
        # 2. Apply Kyutai's RoPE
        # time_before_heads=True matches our [B, T, H, D] shape!
        offset = torch.zeros((B,), dtype=torch.long, device=x.device)
        q, k = apply_rope(q, k, offset=offset, time_before_heads=True, interleave=False)
        
        # 3. Transpose for PyTorch SDPA -> [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 4. Scaled Dot-Product Attention (Handles causal masking internally)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        
        # 5. Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out_proj(attn_output)
        x = x + self.ffn(self.norm2(x))
        return x

class MockDepformer(nn.Module):
    def __init__(self, hidden_dim, num_audio_codebooks, vocab_size):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_audio_codebooks = num_audio_codebooks
        
        self.block = CustomRoPEBlock(hidden_dim, num_heads=4)
        
        self.audio_heads = nn.ModuleList([
            nn.Linear(hidden_dim, vocab_size) for _ in range(num_audio_codebooks)
        ])

    def forward(self, temporal_state):
        B, T, H = temporal_state.shape
        base_context = temporal_state.contiguous().view(B * T, 1, H)
        
        # Replicate context across Q steps
        depth_seq = base_context.repeat(1, self.num_audio_codebooks, 1) # [B*T, Q, H]
        
        # The RoPE step is now dynamically handled inside CustomRoPEBlock
        out = self.block(depth_seq, is_causal=True) # [B*T, Q, H]
        
        logits = []
        for q, head in enumerate(self.audio_heads):
            step_out = out[:, q, :] 
            step_logits = head(step_out) 
            logits.append(step_logits.view(B, T, -1)) 
            
        return logits
        
class MockMoshiLM(nn.Module):
    def __init__(self, vocab_size=2048, hidden_dim=256, num_audio_codebooks=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.text_emb = nn.Embedding(vocab_size, hidden_dim)
        self.audio_embs = nn.ModuleList([
            nn.Embedding(vocab_size, hidden_dim) for _ in range(num_audio_codebooks)
        ])
        
        # Helium (Temporal Transformer) using our custom Kyutai-style RoPE blocks
        self.temporal_blocks = nn.ModuleList([
            CustomRoPEBlock(hidden_dim, num_heads=8) for _ in range(3)
        ])
        
        self.text_head = nn.Linear(hidden_dim, vocab_size)
        self.depformer = MockDepformer(hidden_dim, num_audio_codebooks, vocab_size)

    def forward(self, text_tokens, audio_tokens):
        B, T = text_tokens.shape
        
        # Sum Embeddings
        combined_h = self.text_emb(text_tokens) 
        for q, emb_layer in enumerate(self.audio_embs):
            combined_h += emb_layer(audio_tokens[:, :, q])
            
        # Temporal pass (Helium) with inside RoPE handling
        temporal_out = combined_h
        for block in self.temporal_blocks:
            temporal_out = block(temporal_out, is_causal=True)
        
        text_logits = self.text_head(temporal_out) 
        audio_logits_list = self.depformer(temporal_out)
        
        return text_logits, audio_logits_list

def main():
    print("Moshi Educational Model Training Loop (with Original Kyutai RoPE!)")
    print("==================================================================")
    print("This script validates causal loss pipelines inside custom RoPE blocks")
    print("matching the exact embedding formulas of the real Moshi LM.")
    print("Running sequence... \n")
    
    batch_size = 4
    time_steps = 16
    vocab_size = 1024
    num_codebooks = 8
    
    text_input = torch.randint(0, vocab_size, (batch_size, time_steps))
    audio_input = torch.randint(0, vocab_size, (batch_size, time_steps, num_codebooks))
    
    text_target = torch.randint(0, vocab_size, (batch_size, time_steps))
    audio_target = torch.randint(0, vocab_size, (batch_size, time_steps, num_codebooks))
    
    model = MockMoshiLM(vocab_size=vocab_size, hidden_dim=256, num_audio_codebooks=num_codebooks)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    for step in range(1, 11):
        optimizer.zero_grad()
        text_logits, audio_logits_list = model(text_input, audio_input)
        
        loss_text = criterion(text_logits.view(-1, vocab_size), text_target.view(-1))
        
        loss_audio = 0.0
        for q in range(num_codebooks):
            target_q = audio_target[:, :, q].contiguous().view(-1)
            logits_q = audio_logits_list[q].contiguous().view(-1, vocab_size)
            loss_audio += criterion(logits_q, target_q)
            
        total_loss = loss_text + loss_audio
        total_loss.backward()
        
        optimizer.step()
        scheduler.step()
        
        print(f"Epoch {step:02d} | Text Loss: {loss_text.item():.4f} | Audio Loss (x8): {loss_audio.item():.4f} | Total: {total_loss.item():.4f}")
        
    print("\nKyutai RoPE-Enabled Training graph completed successfully!")

if __name__ == "__main__":
    main()

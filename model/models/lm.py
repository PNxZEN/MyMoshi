import torch
import torch.nn as nn

class MoshiLM(nn.Module):
    """
    Mathematical reconstruction of the Moshi Language Model.
    
    This replaces the complex deployment pipelines in the original repo with 
    a highly readable PyTorch module demonstrating how Moshi simultaneously 
    interleaves User text tokens and Agent audio/text tokens.
    
    The architecture relies on two intertwined models:
    1. Temporal_Transformer: A standard large autoregressive transformer (Helium)
       that resolves sequences over time (T). 
    2. Depformer (Depthwise Transformer): A small transformer operating purely 
       over the codebook depth (Q) at a single time step!
    """
    def __init__(
        self,
        dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 12,
        num_audio_codebooks: int = 8,
        audio_vocab_size: int = 2048,
        text_vocab_size: int = 32000,
    ):
        super().__init__()
        
        self.num_audio_codebooks = num_audio_codebooks
        self.audio_vocab_size = audio_vocab_size
        self.text_vocab_size = text_vocab_size
        
        # --- 1. Embeddings ---
        # Moshi expects K streams per timestep: [1 Text Token, 8 Audio Tokens]
        # It creates embeddings for all of them and sums them.
        self.text_emb = nn.Embedding(text_vocab_size, dim)
        self.audio_embs = nn.ModuleList([
            nn.Embedding(audio_vocab_size, dim) for _ in range(num_audio_codebooks)
        ])
        
        # --- 2. Temporal Transformer (Simulated "Helium" Base) ---
        # This processes the sum of embeddings across the Time dimension
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=num_heads, 
            batch_first=True, 
            norm_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(temporal_layer, num_layers=num_layers)
        
        # Output Head for Text: Predicts the Inner Monologue (Text Token) for timestep t+1
        self.text_head = nn.Linear(dim, text_vocab_size)
        
        # --- 3. Depformer (Depthwise Transformer) ---
        # Once the text token is predicted, the Depformer autoregressively 
        # predicts the 8 Audio Codebooks one by one vertically.
        depformer_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=4,
            batch_first=True,
            norm_first=True
        )
        self.depformer = nn.TransformerEncoder(depformer_layer, num_layers=2)
        
        # Individual output heads for each acoustic codebook level
        self.audio_heads = nn.ModuleList([
            nn.Linear(dim, audio_vocab_size) for _ in range(num_audio_codebooks)
        ])

    def forward_train(self, text_tokens: torch.Tensor, audio_tokens: torch.Tensor):
        """
        A simplified training pass demonstrating Moshi's dual streams.
        Args:
            text_tokens: [B, T]
            audio_tokens: [B, Q, T] where Q = 8
        Returns:
            text_logits: [B, T, Text_Vocab]
            audio_logits: [B, Q, T, Audio_Vocab]
        """
        B, T = text_tokens.shape
        Q = self.num_audio_codebooks
        
        # 1. Sum up embeddings for the Temporal Transformer
        # This is the "interleaved multimodal" input logic.
        combined_embedded = self.text_emb(text_tokens)  # [B, T, Dim]
        for q in range(Q):
            combined_embedded += self.audio_embs[q](audio_tokens[:, q, :])
            
        # 2. Add masking for autoregressive Temporal modeling (Causal Mask)
        # Note: A real implementation tracks KV caches, but this shows the math.
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T).to(combined_embedded.device)
        
        temporal_out = self.temporal_transformer(combined_embedded, mask=causal_mask) # [B, T, Dim]
        
        # 3. Predict the Text tokens via Inner Monologue
        text_logits = self.text_head(temporal_out) # [B, T, Text_Vocab]
        
        # 4. Depformer Expansion
        # The Temporal output acts as the initial state for the Depformer at EACH timestep.
        # We expand it to the codebook dimension to compute acoustic logits.
        # Depformer input shape: [B * T, Q, Dim]
        temporal_flat = temporal_out.view(B * T, 1, -1)
        
        # We simulate the depth-wise sequence by creating Q steps 
        # (In training, we use teacher-forcing via real audio tokens)
        flattened_audio_tokens = audio_tokens.transpose(1, 2).reshape(B * T, Q) # [B*T, Q]
        
        depformer_inputs = [temporal_flat] # Start with Temporal state (representing Text + Semantic intention)
        
        # Autoregressively embed the true tokens to feed into subsequent Depformer steps
        for q in range(Q - 1):
            emb = self.audio_embs[q](flattened_audio_tokens[:, q]).unsqueeze(1)
            depformer_inputs.append(emb)
            
        depformer_inputs = torch.cat(depformer_inputs, dim=1) # [B*T, Q, Dim]
        
        # Mask the depthwise dimension so predicting Codebook $i$ only sees $< i$
        depth_mask = nn.Transformer.generate_square_subsequent_mask(Q).to(combined_embedded.device)
        
        depformer_out = self.depformer(depformer_inputs, mask=depth_mask) # [B*T, Q, Dim]
        
        # 5. Route to specific audio heads
        # Each layer of the RVQ gets its own linear projection head
        audio_logits = []
        for q in range(Q):
            logits_q = self.audio_heads[q](depformer_out[:, q, :]) # [B*T, Audio_Vocab]
            audio_logits.append(logits_q.view(B, T, -1).unsqueeze(1)) # [B, 1, T, Audio_Vocab]
            
        audio_logits = torch.cat(audio_logits, dim=1) # [B, Q, T, Audio_Vocab]
        
        return text_logits, audio_logits

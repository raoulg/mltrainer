import math
import os

import psutil
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def print_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    # rss is the Resident Set Size, the non-swapped physical memory a process has used.
    # It's a good measure of the actual memory the process is consuming.
    memory_mb = process.memory_info().rss / (1024 * 1024)
    print(f"Current process memory usage: {memory_mb:.2f} MB")


class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding module. This remains unchanged from your implementation.
    It adds positional information to the input embeddings.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_seq_len, d_model)
        # Shape: (1, max_seq_len, d_model) to allow broadcasting to the batch dimension

        # Apply sine to even indices and cosine to odd indices
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # register_buffer makes 'pe' a part of the module's state, but not a parameter
        # This means it will be moved to the correct device (e.g., GPU) with .to(device)
        # but won't be updated by the optimizer.
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Injects positional information into the input tensor.
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        # Add the positional encoding. We only use the encodings up to the sequence length of the input.
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MemoryEfficientTransformerBlock(nn.Module):
    """
    A memory-efficient Transformer Block using torch.nn.functional.scaled_dot_product_attention.

    This block replaces the standard nn.MultiheadAttention with a more memory-friendly
    approach. Instead of a single heavy layer, we define our own projection layers
    for query (Q), key (K), and value (V) and then call the optimized attention function.
    """

    def __init__(self, hidden_size, num_heads, dropout):
        super(MemoryEfficientTransformerBlock, self).__init__()

        # Ensure that the hidden size is divisible by the number of heads
        assert hidden_size % num_heads == 0, (
            "hidden_size must be divisible by num_heads"
        )

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        # A single linear layer to project the input to Q, K, and V all at once.
        # This is more efficient than having three separate linear layers.
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)

        # A final linear layer to project the concatenated heads back to the original hidden size.
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # Standard Feed-Forward network
        self.ff = nn.Sequential(
            nn.Linear(
                hidden_size, 4 * hidden_size
            ),  # Often, the FFN intermediate size is 4x hidden_size
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout),
        )

        # Layer normalization layers
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        """
        Forward pass for the memory-efficient transformer block.
        This implements Pre-layerNorm (as opposed to Post-layerNorm)
        because it is found to give better training stability in many cases.
        """
        # Pre-LayerNorm and skip connection for the attention part
        identity = x
        x_norm = self.layer_norm1(x)

        # 1. Project to Q, K, V
        # Input x_norm shape: (batch_size, seq_len, hidden_size)
        # qkv shape: (batch_size, seq_len, 3 * hidden_size)
        qkv = self.qkv_proj(x_norm)

        # 2. Reshape and permute for multi-head attention
        # New shape: (batch_size, seq_len, num_heads, 3 * head_dim)
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_heads, 3 * self.head_dim)
        # Permute to: (batch_size, num_heads, seq_len, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)

        # 3. Split into Q, K, V
        # Each will have shape: (batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv.chunk(3, dim=-1)

        # 4. Apply scaled dot-product attention
        # This is the core of the memory optimization. It avoids creating the large NxN attention matrix.
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # Set to True for autoregressive tasks (e.g., decoding)
        )

        # 5. Concatenate heads and project back
        # Reshape back to (batch_size, seq_len, hidden_size)
        attn_output = (
            attn_output.permute(0, 2, 1, 3).contiguous().view(-1, self.hidden_size)
        )
        attn_output = attn_output.view(x.size(0), x.size(1), self.hidden_size)

        # Project the concatenated heads
        attention_out = self.out_proj(attn_output)

        # First skip connection (Add & Norm)
        x = identity + attention_out

        # Pre-LayerNorm and skip connection for the FFN part
        identity = x
        x_norm = self.layer_norm2(x)
        ff_out = self.ff(x_norm)

        # Second skip connection (Add & Norm)
        x = identity + ff_out

        return x


class Transformer(nn.Module):
    """
    The main Transformer model. This remains largely the same, but now it uses the
    MemoryEfficientTransformerBlock instead of the original one.
    """

    def __init__(
        self,
        config: dict,
    ) -> None:
        super().__init__()
        # Initial convolution layer to create embeddings from the raw signal
        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=config["hidden"],
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.pos_encoder = PositionalEncoding(config["hidden"], config["dropout"])

        # Create multiple memory-efficient transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                MemoryEfficientTransformerBlock(
                    config["hidden"], config["num_heads"], config["dropout"]
                )
                for _ in range(config["num_blocks"])
            ]
        )

        # Final output layer
        self.out = nn.Linear(config["hidden"], config["output"])

    def forward(self, x: Tensor) -> Tensor:
        # Expected input shape for this model: (batch, seq_len, channels=1)
        # conv1d expects: (batch, channels, seq_len)
        # We transpose to match the conv1d expected input format.
        x = self.conv1d(x.transpose(1, 2))

        # After conv, shape is (batch, hidden_channels, new_seq_len)
        # Transpose back for the transformer blocks: (batch, new_seq_len, hidden_channels)
        x = x.transpose(1, 2)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Apply multiple transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Global Average Pooling: average across the sequence dimension
        x = x.mean(dim=1)

        # Final linear layer for classification/regression
        x = self.out(x)
        return x

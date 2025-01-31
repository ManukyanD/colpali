from numpy import dtype
import torch
import torch.nn as nn
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input


class LatentOutputAttn(nn.Module):

    def __init__(
        self, num_vectors=512, hidden_size=1536, intermediate_size=8960, num_heads=12
    ):
        super().__init__()
        self.num_vectors = num_vectors
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.latent_kv = nn.Parameter(
            torch.randn(
                (self.num_vectors, self.hidden_size),
                requires_grad=True,
            ),
            requires_grad=True,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=self.intermediate_size,
                out_features=self.hidden_size,
            ),
        )

    def forward(self, hidden_states, attention_mask):
        bsz = hidden_states.size()[0]
        (
            queries,
            indices_q,
            cu_seqlens_q,
            max_seqlen_in_batch_q,
            used_seqlens_in_batch_q,
        ) = unpad_input(hidden_states=hidden_states, attention_mask=attention_mask)
        queries = queries.view(queries.shape[0], self.num_heads, -1)

        cu_seqlens_k = torch.cumsum(
            torch.tensor([0] + [self.num_vectors] * bsz), dim=0, dtype=torch.int32
        ).to(device=self.latent_kv.device)

        keys = (
            self.latent_kv.unsqueeze(0)
            .repeat((bsz, 1, 1))
            .view(bsz * self.num_vectors, self.num_heads, -1)
        )
        values = (
            self.latent_kv.unsqueeze(0)
            .repeat((bsz, 1, 1))
            .view(bsz * self.num_vectors, self.num_heads, -1)
        )

        attn_output = flash_attn_varlen_func(
            q=queries,
            k=keys,
            v=values,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=self.num_vectors,
            causal=False,
        )
        hidden_states = pad_input(
            hidden_states=attn_output,
            indices=indices_q,
            batch=bsz,
            seqlen=max_seqlen_in_batch_q,
        )

        hidden_states = hidden_states.view(bsz, -1, self.hidden_size).contiguous()
        hidden_states = self.output_projection(hidden_states)
        return hidden_states

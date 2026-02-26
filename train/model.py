"""train/model.py — VLA model: DINOv2 + CLIP text + cross-attention + action head.

Architecture
------------
    Input: frame (3,224,224) + language command string
        │                           │
    DINOv2-small (frozen, eval)  CLIPTextModel (frozen, eval)
    (B, 257, 384)                 (B, L, 512)
        │                           │
        │                  Text projection Linear(512→384)
        │                           │ (B, L, 384)
        └──── Cross-attention (n layers, pre-norm) ────┘
              visual tokens as Q, text tokens as K/V
              fused = fused + CrossAttn(LayerNorm(fused), txt, txt)
              (B, 257, 384)
                      │
               Attention pooling over 256 patch tokens
               pool_scores → softmax → weighted sum → (B, 384)
                      │
              MLP action head 384→256→64→2
                      │
              2.0 * tanh(x)  ← bounded to (−2, 2) rad/s
                      │
              (pan_vel, tilt_vel)

Frozen encoders account for ~85M params (DINOv2-small + CLIP text).
Trainable params: txt_proj + fusion + fusion_norms + pool_scores + action_head ≈ 1.5-2M.

Design notes
------------
Attention pooling: the 256 patch tokens from DINOv2 are pooled via a small
learned linear (pool_scores, 385 params) that produces per-patch scalar
weights, followed by softmax and a weighted sum. This lets the network attend
to face-relevant patches. Mean pooling is permutation-invariant (destroys
where information); CLS is trained for discriminative global features, not
localization. Attention pooling with a learned score is a cheap middle ground.

Scaled tanh output: 2.0 * tanh(x) bounds predictions to (−2, 2) rad/s from
step 0, matching the oracle label range and preventing large early-training
MSE losses.

Pre-norm: LayerNorm before each cross-attention layer rather than after, for
more stable training.

Dropout: applied after each hidden activation in the action head. With ~2M
trainable params and a small dataset, this is the primary regulariser.

Frozen encoder eval mode: self.dino and self.clip_text are always kept in
eval() mode via an overridden train() method. Freezing parameters alone is
not sufficient — .training still controls dropout/stochastic-depth inside
the encoder, which would inject noise into the frozen forward pass.

inference_mode vs no_grad: the encoder forward passes use torch.no_grad()
rather than torch.inference_mode(). inference_mode creates "inference tensors"
that cannot be saved for backward; passing them to a trainable nn.Linear
(which saves its input to compute weight gradients) raises a RuntimeError.
no_grad() does not have this restriction and is the correct choice here.

Output range / inference note
------------------------------
The oracle labels span roughly ±0.5 rad/s for typical scenarios (kp_pan=10,
error ≈ A·ω ≈ 0.15 rad/s), with a hard cap at ±2.0 rad/s (oracle max_velocity).
The ONNX inference node (vla_node.py) clips outputs to ±1.0 rad/s as an
additional safety rail; this is conservative but acceptable given the empirical
label distribution.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn


class VLAModel(nn.Module):
    """Vision-Language-Action model for pan-tilt face tracking.

    Each cross-attention fusion layer uses a pre-norm residual:
        fused = fused + CrossAttn(Q=LayerNorm(fused), K=txt, V=txt)
    Padding tokens in the text sequence are masked so visual tokens cannot
    attend to them (key_padding_mask derived from attention_mask).

    Frozen encoders (dino, clip_text) are permanently kept in eval() mode
    so their internal dropout does not inject noise during training.

    Args:
        n_fusion_layers: Number of cross-attention layers (default 2).
        n_heads:         Attention heads; must divide VIS_DIM=384 (default 6 → head_dim=64).
        pretrained:      If True, load pretrained DINOv2 + CLIPTextModel weights from
                         HuggingFace. Set False to use lightweight stubs (no download
                         required — intended for unit tests and offline CI).
        dropout:         Dropout probability applied after each hidden activation in the
                         action head (default 0.1). Set 0.0 to disable.
    """

    DINO_ID  = "facebook/dinov2-small"
    CLIP_ID  = "openai/clip-vit-base-patch32"
    VIS_DIM  = 384   # DINOv2-small hidden dim
    # 256 patch tokens + 1 CLS, for all DINOv2 variants at 224×224 input (patch_size=14).
    VIS_SEQ  = 257
    TXT_DIM  = 512   # CLIP text hidden dim

    def __init__(
        self,
        n_fusion_layers: int = 2,
        n_heads: int = 6,
        pretrained: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Encoders (frozen, permanently in eval mode) ───────────────────
        if pretrained:
            from transformers import AutoModel, CLIPTextModel
            # use_safetensors=True: required by transformers ≥5.2 (CVE-2025-32434
            # blocks torch.load unless torch ≥2.6; safetensors is unaffected).
            self.dino      = AutoModel.from_pretrained(self.DINO_ID,  use_safetensors=True)
            self.clip_text = CLIPTextModel.from_pretrained(self.CLIP_ID, use_safetensors=True)
            # Verify the loaded model produces the expected sequence length at
            # the inference image size (224×224), not the training image_size
            # in the config (DINOv2 is trained at 518×518 but handles any
            # size via position-embedding interpolation).
            infer_size = 224
            n_patches = (infer_size // self.dino.config.patch_size) ** 2
            expected_seq = n_patches + 1  # +1 for CLS
            assert expected_seq == self.VIS_SEQ, (
                f"DINOv2 patch_size={self.dino.config.patch_size} at {infer_size}px "
                f"gives {expected_seq} tokens but VIS_SEQ={self.VIS_SEQ}."
            )
        else:
            self.dino      = _VisualStub()
            self.clip_text = _TextStub()

        for p in self.dino.parameters():
            p.requires_grad = False
        for p in self.clip_text.parameters():
            p.requires_grad = False

        # Keep encoders in eval so their dropout/stochastic-depth is disabled.
        # train() is overridden below to enforce this even after model.train() calls.
        self.dino.eval()
        self.clip_text.eval()

        # ── Trainable components ──────────────────────────────────────────
        # Project text embeddings from CLIP dim → DINOv2 dim
        self.txt_proj = nn.Linear(self.TXT_DIM, self.VIS_DIM)

        # Cross-attention layers: visual tokens (Q) attend to projected text tokens (K, V)
        self.fusion = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.VIS_DIM,
                num_heads=n_heads,
                batch_first=True,
            )
            for _ in range(n_fusion_layers)
        ])

        # Pre-norm LayerNorm applied before each cross-attention step
        self.fusion_norms = nn.ModuleList([
            nn.LayerNorm(self.VIS_DIM)
            for _ in range(n_fusion_layers)
        ])

        # Attention pooling over the 256 patch tokens.
        # Produces a per-patch scalar score; softmax + weighted sum gives a
        # location-sensitive pooled representation without destroying spatial structure.
        self.pool_scores = nn.Linear(self.VIS_DIM, 1)

        # MLP action head: pooled patch representation → (pan_vel, tilt_vel)
        self.action_head = nn.Sequential(
            nn.Linear(self.VIS_DIM, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64),           nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def train(self, mode: bool = True) -> VLAModel:
        """Override to keep frozen encoders permanently in eval mode."""
        super().train(mode)
        self.dino.eval()
        self.clip_text.eval()
        return self

    def forward(
        self,
        frames: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass.

        Args:
            frames:         (B, 3, 224, 224) float32 — ImageNet-normalised frames
            input_ids:      (B, L) long — CLIP tokenizer output
            attention_mask: (B, L) long — 1 for real tokens, 0 for padding

        Returns:
            actions: (B, 2) float32 — predicted (pan_vel, tilt_vel) in rad/s,
                     bounded to (−2, 2) via 2.0 * tanh(x)
        """
        # no_grad disables gradient tracking for the frozen encoder forward passes.
        # inference_mode cannot be used here: nn.Linear saves its input tensor for
        # the weight gradient computation, and inference tensors cannot be saved
        # for backward (RuntimeError). no_grad avoids this without a copy.
        with torch.no_grad():
            vis        = self.dino(pixel_values=frames).last_hidden_state       # (B, 257, 384)
            txt_hidden = self.clip_text(
                input_ids, attention_mask=attention_mask
            ).last_hidden_state                                                  # (B, L,  512)

        txt_proj = self.txt_proj(txt_hidden)                                    # (B, L,  384)

        # True in positions that should be *ignored* (padding tokens)
        key_padding_mask = (attention_mask == 0)                                # (B, L)  bool

        fused = vis
        for attn_layer, norm in zip(self.fusion, self.fusion_norms):
            normed   = norm(fused)                                              # pre-norm
            attn_out, _ = attn_layer(
                normed, txt_proj, txt_proj,
                key_padding_mask=key_padding_mask,
            )                                                                    # (B, 257, 384)
            fused = fused + attn_out                                            # residual, no post-norm

        # Attention pooling over the 256 patch tokens (exclude CLS at index 0).
        # pool_scores learns which patch directions signal face presence/location.
        patches = fused[:, 1:, :]                                               # (B, 256, 384)
        weights = torch.softmax(self.pool_scores(patches), dim=1)               # (B, 256,   1)
        spatial = (weights * patches).sum(dim=1)                                # (B,       384)

        return 2.0 * torch.tanh(self.action_head(spatial))                      # (B, 2)


# ---------------------------------------------------------------------------
# Lightweight stubs — pretrained=False, no HuggingFace downloads required
# ---------------------------------------------------------------------------

class _VisualStub(nn.Module):
    """Stub DINOv2-small: correct output shape, has one frozen parameter
    so gradient tests can verify freezing behaviour without a 85 MB download."""

    def __init__(self) -> None:
        super().__init__()
        # Tiny sentinel parameter — frozen by VLAModel after construction
        self._sentinel = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, pixel_values: torch.Tensor, **kw):
        B = pixel_values.shape[0]
        hs = torch.zeros(
            B, VLAModel.VIS_SEQ, VLAModel.VIS_DIM,
            device=pixel_values.device,
        )
        return SimpleNamespace(last_hidden_state=hs)


class _TextStub(nn.Module):
    """Stub CLIPTextModel: correct output shape, has one frozen parameter
    so gradient tests can verify freezing behaviour without a 350 MB download."""

    def __init__(self) -> None:
        super().__init__()
        self._sentinel = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        **kw,
    ):
        B, L = input_ids.shape
        device = input_ids.device
        # Each position i activates a distinct dimension, so different positions
        # produce distinguishable K/V vectors in cross-attention.  This lets
        # test_padding_mask_applied verify that key_padding_mask is wired up
        # (masking out positions changes the weighted sum).
        hs = torch.zeros(B, L, VLAModel.TXT_DIM, device=device)
        for i in range(min(L, VLAModel.TXT_DIM)):
            hs[:, i, i] = 1.0
        return SimpleNamespace(last_hidden_state=hs)

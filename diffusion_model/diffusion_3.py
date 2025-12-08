"""
Significantly expanded architecture with:
- 4-6x more parameters
- Multi-resolution attention
- Deeper encoders
- More residual blocks per level
- Cross-attention conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
import numpy as np
import json

# =======================
# 1. ENHANCED ENCODERS
# =======================

class IdentityEncoder(nn.Module):
    """
    Deep identity encoder with 768 dimensions
    Significantly more capacity for identity preservation
    """
    def __init__(self, output_dim=768):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1: 256x256 -> 128x128
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),

            # Block 2: 128x128 -> 64x64
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),

            # Block 3: 64x64 -> 32x32
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),

            # Block 4: 32x32 -> 16x16
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.GroupNorm(8, 512),
            nn.SiLU(),

            # # Block 5: 16x16 -> 8x8
            nn.Conv2d(512, 768, 3, stride=2, padding=1),
            nn.GroupNorm(8, 768),
            nn.SiLU(),
            nn.Conv2d(768, 768, 3, padding=1),
            nn.GroupNorm(8, 768),
            nn.SiLU(),

            # Global pool and project
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        self.proj = nn.Sequential(
            nn.Linear(768, output_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        """
        Args:
            x: input image (batch, 3, H, W)
        Returns:
            identity_embed: (batch, output_dim)
        """
        features = self.encoder(x)
        return self.proj(features)


class AgeEncoder(nn.Module):
    """Enhanced age encoder with positional encoding"""
    def __init__(self, max_age=100, embed_dim=256):
        super().__init__()
        self.age_embedding = nn.Embedding(max_age + 1, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )

    def forward(self, age):
        """
        Args:
            age: tensor of shape (batch,) with integer ages
        Returns:
            age_embed: (batch, embed_dim)
        """
        embed = self.age_embedding(age)
        return self.proj(embed)
# =======================
# 2. ATTENTION MODULES
# =======================

class SpatialAttention(nn.Module):
    """Enhanced self-attention for spatial features"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(self, x):
        """
        Args:
            x: (batch, channels, H, W)
        Returns:
            attention output: (batch, channels, H, W)
        """
        B, C, H, W = x.shape

        # Normalize
        h = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h)
        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # (3, B, heads, HW, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)

        # Apply attention
        out = attn @ v  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)

        # Project and residual
        out = self.proj(out)
        return x + out


class CrossAttention(nn.Module):
    """Cross-attention for conditioning on embeddings"""
    def __init__(self, channels, cond_dim, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        assert channels % num_heads == 0
        
        self.norm_x = nn.GroupNorm(8, channels)
        self.norm_cond = nn.LayerNorm(cond_dim)
        
        self.to_q = nn.Conv2d(channels, channels, 1)
        self.to_k = nn.Linear(cond_dim, channels)
        self.to_v = nn.Linear(cond_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5
        
    def forward(self, x, cond):
        """
        Args:
            x: spatial features (batch, channels, H, W)
            cond: conditioning (batch, cond_dim)
        Returns:
            cross-attention output (batch, channels, H, W)
        """
        B, C, H, W = x.shape
        
        # Normalize
        x_norm = self.norm_x(x)
        cond_norm = self.norm_cond(cond)
        
        # Query from spatial features
        q = self.to_q(x_norm)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        q = q.permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        
        # Key and Value from conditioning
        k = self.to_k(cond_norm)  # (B, C)
        v = self.to_v(cond_norm)  # (B, C)
        k = k.reshape(B, self.num_heads, 1, C // self.num_heads)
        v = v.reshape(B, self.num_heads, 1, C // self.num_heads)
        
        # Cross-attention
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = attn @ v  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # Project and residual
        out = self.proj(out)
        return x + out

# =======================
# 3. ENHANCED RESIDUAL BLOCKS
# =======================

class ResidualBlock(nn.Module):
    """Enhanced residual block with cross-attention conditioning"""
    def __init__(self, in_ch, out_ch, time_dim=512, cond_dim=1024, dropout=0.1, use_cross_attn=False):
        super().__init__()
        
        self.use_cross_attn = use_cross_attn

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)

        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch * 2)
        )

        # Conditioning embedding projection
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_ch * 2)
        )
        
        # Optional cross-attention
        if use_cross_attn:
            self.cross_attn = CrossAttention(out_ch, cond_dim, num_heads=8)

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_embed, cond_embed):
        """
        Args:
            x: (batch, in_ch, H, W)
            t_embed: time embedding (batch, time_dim)
            cond_embed: concatenated age+identity embedding (batch, cond_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)

        # Time conditioning with scale and shift
        time_emb = self.time_mlp(t_embed)
        time_scale, time_shift = time_emb.chunk(2, dim=1)
        h = h * (1 + time_scale[:, :, None, None]) + time_shift[:, :, None, None]

        # Conditioning with scale and shift
        cond_emb = self.cond_mlp(cond_embed)
        cond_scale, cond_shift = cond_emb.chunk(2, dim=1)
        h = h * (1 + cond_scale[:, :, None, None]) + cond_shift[:, :, None, None]

        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)
        
        # Optional cross-attention
        if self.use_cross_attn:
            h = self.cross_attn(h, cond_embed)

        return h + self.shortcut(x)

# =======================
# 4. TIME EMBEDDING
# =======================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time embedding"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb

# =======================
# 5. LARGE U-NET
# =======================

class ConditionalUNet(nn.Module):
    """
    Large-scale U-Net with:
    - 192 base channels (2x increase)
    - 3 residual blocks per level
    - Multi-resolution attention (64x64, 32x32, 16x16)
    - Cross-attention at key resolutions
    - Deeper bottleneck
    """
    def __init__(
        self,
        img_channels=3,
        base_ch=80,
        time_dim=512,
        age_dim=256,
        identity_dim=240,
        dropout=0.1,
        use_checkpoint=True
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        cond_dim = age_dim + identity_dim  

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(time_dim * 4, time_dim)
        )

        # Enhanced conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, cond_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cond_dim * 2, cond_dim),
            nn.LayerNorm(cond_dim)
        )

        # ==================
        # ENCODER (4 levels, 3 blocks each)
        # ==================
        
        # Level 1: 256x256, channels=192
        self.enc1_1 = ResidualBlock(img_channels, base_ch, time_dim, cond_dim, dropout)
        self.enc1_2 = ResidualBlock(base_ch, base_ch, time_dim, cond_dim, dropout)
        self.enc1_3 = ResidualBlock(base_ch, base_ch, time_dim, cond_dim, dropout)
        self.down1 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)

        # Level 2: 128x128, channels=384
        self.enc2_1 = ResidualBlock(base_ch, base_ch * 2, time_dim, cond_dim, dropout)
        self.enc2_2 = ResidualBlock(base_ch * 2, base_ch * 2, time_dim, cond_dim, dropout)
        self.enc2_3 = ResidualBlock(base_ch * 2, base_ch * 2, time_dim, cond_dim, dropout)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 3, stride=2, padding=1)

        # Level 3: 64x64, channels=576 (with attention)
        self.enc3_1 = ResidualBlock(base_ch * 2, base_ch * 3, time_dim, cond_dim, dropout, use_cross_attn=True)
        self.enc3_2 = ResidualBlock(base_ch * 3, base_ch * 3, time_dim, cond_dim, dropout)
        self.enc3_3 = ResidualBlock(base_ch * 3, base_ch * 3, time_dim, cond_dim, dropout)
        self.enc3_attn = SpatialAttention(base_ch * 3, num_heads=8)
        self.down3 = nn.Conv2d(base_ch * 3, base_ch * 3, 3, stride=2, padding=1)

        # # Level 4: 32x32, channels=768 (with attention)
        self.enc4_1 = ResidualBlock(base_ch * 3, base_ch * 4, time_dim, cond_dim, dropout, use_cross_attn=True)
        self.enc4_2 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim, dropout)
        self.enc4_3 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim, dropout)
        self.enc4_attn = SpatialAttention(base_ch * 4, num_heads=8)
        self.down4 = nn.Conv2d(base_ch * 4, base_ch * 4, 3, stride=2, padding=1)

        # ==================
        # BOTTLENECK: 16x16, channels=768 (6 blocks with attention)
        # ==================
        self.bottleneck_blocks = nn.ModuleList([
            ResidualBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim, dropout, use_cross_attn=(i % 2 == 0))
            for i in range(6)
        ])
        self.bottleneck_attns = nn.ModuleList([
            SpatialAttention(base_ch * 4, num_heads=8)
            for _ in range(3)
        ])

        # ==================
        # DECODER (3 levels, 3 blocks each)
        # ==================
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Level 4: 32x32
        self.dec4_1 = ResidualBlock(base_ch * 8, base_ch * 4, time_dim, cond_dim, dropout, use_cross_attn=True)
        self.dec4_2 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim, dropout)
        self.dec4_3 = ResidualBlock(base_ch * 4, base_ch * 4, time_dim, cond_dim, dropout)
        self.dec4_attn = SpatialAttention(base_ch * 4, num_heads=8)

        # Level 3: 64x64
        self.dec3_1 = ResidualBlock(base_ch * 7, base_ch * 3, time_dim, cond_dim, dropout, use_cross_attn=True)
        self.dec3_2 = ResidualBlock(base_ch * 3, base_ch * 3, time_dim, cond_dim, dropout)
        self.dec3_3 = ResidualBlock(base_ch * 3, base_ch * 3, time_dim, cond_dim, dropout)
        self.dec3_attn = SpatialAttention(base_ch * 3, num_heads=8)

        # Level 2: 128x128
        self.dec2_1 = ResidualBlock(base_ch * 5, base_ch * 2, time_dim, cond_dim, dropout)
        self.dec2_2 = ResidualBlock(base_ch * 2, base_ch * 2, time_dim, cond_dim, dropout)
        self.dec2_3 = ResidualBlock(base_ch * 2, base_ch * 2, time_dim, cond_dim, dropout)

        # Level 1: 256x256
        self.dec1_1 = ResidualBlock(base_ch * 3, base_ch, time_dim, cond_dim, dropout)
        self.dec1_2 = ResidualBlock(base_ch, base_ch, time_dim, cond_dim, dropout)
        self.dec1_3 = ResidualBlock(base_ch, base_ch, time_dim, cond_dim, dropout)

        # Output
        self.out = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, img_channels, 3, padding=1)
        )

    def forward(self, x, t, age_embed, identity_embed):
        """
        Args:
            x: noisy image (batch, 3, H, W)
            t: timestep (batch,)
            age_embed: (batch, age_dim)
            identity_embed: (batch, identity_dim)
        Returns:
            predicted noise (batch, 3, H, W)
        """
        # Prepare conditioning
        t_embed = self.time_mlp(t)
        cond_embed = torch.cat([age_embed, identity_embed], dim=1)
        cond_embed = self.cond_proj(cond_embed)

        # Helper for gradient checkpointing
        def create_forward_fn(module):
            def forward_fn(*inputs):
                return module(*inputs)
            return forward_fn

        # ==================
        # ENCODER
        # ==================
        # Level 1
        if self.use_checkpoint and self.training:
            e1 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc1_1), x, t_embed, cond_embed, use_reentrant=False
            )
            e1 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc1_2), e1, t_embed, cond_embed, use_reentrant=False
            )
            e1 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc1_3), e1, t_embed, cond_embed, use_reentrant=False
            )
        else:
            e1 = self.enc1_1(x, t_embed, cond_embed)
            e1 = self.enc1_2(e1, t_embed, cond_embed)
            e1 = self.enc1_3(e1, t_embed, cond_embed)

        # Level 2
        e2_in = self.down1(e1)
        if self.use_checkpoint and self.training:
            e2 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc2_1), e2_in, t_embed, cond_embed, use_reentrant=False
            )
            e2 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc2_2), e2, t_embed, cond_embed, use_reentrant=False
            )
            e2 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc2_3), e2, t_embed, cond_embed, use_reentrant=False
            )
        else:
            e2 = self.enc2_1(e2_in, t_embed, cond_embed)
            e2 = self.enc2_2(e2, t_embed, cond_embed)
            e2 = self.enc2_3(e2, t_embed, cond_embed)

        # Level 3 (with attention)
        e3_in = self.down2(e2)
        if self.use_checkpoint and self.training:
            e3 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc3_1), e3_in, t_embed, cond_embed, use_reentrant=False
            )
            e3 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc3_2), e3, t_embed, cond_embed, use_reentrant=False
            )
            e3 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc3_3), e3, t_embed, cond_embed, use_reentrant=False
            )
            e3 = torch.utils.checkpoint.checkpoint(self.enc3_attn, e3, use_reentrant=False)
        else:
            e3 = self.enc3_1(e3_in, t_embed, cond_embed)
            e3 = self.enc3_2(e3, t_embed, cond_embed)
            e3 = self.enc3_3(e3, t_embed, cond_embed)
            e3 = self.enc3_attn(e3)

        # Level 4 (with attention)
        e4_in = self.down3(e3)
        if self.use_checkpoint and self.training:
            e4 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc4_1), e4_in, t_embed, cond_embed, use_reentrant=False
            )
            e4 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc4_2), e4, t_embed, cond_embed, use_reentrant=False
            )
            e4 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.enc4_3), e4, t_embed, cond_embed, use_reentrant=False
            )
            e4 = torch.utils.checkpoint.checkpoint(self.enc4_attn, e4, use_reentrant=False)
        else:
            e4 = self.enc4_1(e4_in, t_embed, cond_embed)
            e4 = self.enc4_2(e4, t_embed, cond_embed)
            e4 = self.enc4_3(e4, t_embed, cond_embed)
            e4 = self.enc4_attn(e4)

        # ==================
        # BOTTLENECK (6 blocks with attention)
        # ==================
        b = self.down4(e4)
        for i, block in enumerate(self.bottleneck_blocks):
            if self.use_checkpoint and self.training:
                b = torch.utils.checkpoint.checkpoint(
                    create_forward_fn(block), b, t_embed, cond_embed, use_reentrant=False
                )
            else:
                b = block(b, t_embed, cond_embed)
            
            # Apply attention every 2 blocks
            if i % 2 == 1:
                attn_idx = i // 2
                if self.use_checkpoint and self.training:
                    b = torch.utils.checkpoint.checkpoint(
                        self.bottleneck_attns[attn_idx], b, use_reentrant=False
                    )
                else:
                    b = self.bottleneck_attns[attn_idx](b)
        # ==================
        # DECODER
        # ==================
        # Level 4
        d4_in = torch.cat([self.up(b), e4], dim=1)
        if self.use_checkpoint and self.training:
            d4 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec4_1), d4_in, t_embed, cond_embed, use_reentrant=False
            )
            d4 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec4_2), d4, t_embed, cond_embed, use_reentrant=False
            )
            d4 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec4_3), d4, t_embed, cond_embed, use_reentrant=False
            )
            d4 = torch.utils.checkpoint.checkpoint(self.dec4_attn, d4, use_reentrant=False)
        else:
            d4 = self.dec4_1(d4_in, t_embed, cond_embed)
            d4 = self.dec4_2(d4, t_embed, cond_embed)
            d4 = self.dec4_3(d4, t_embed, cond_embed)
            d4 = self.dec4_attn(d4)

        # Level 3
        
        d3_in = torch.cat([self.up(d4), e3], dim=1)
        if self.use_checkpoint and self.training:
            d3 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec3_1), d3_in, t_embed, cond_embed, use_reentrant=False
            )
            d3 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec3_2), d3, t_embed, cond_embed, use_reentrant=False
            )
            d3 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec3_3), d3, t_embed, cond_embed, use_reentrant=False
            )
            d3 = torch.utils.checkpoint.checkpoint(self.dec3_attn, d3, use_reentrant=False)
        else:
           
            d3 = self.dec3_1(d3_in, t_embed, cond_embed)
            d3 = self.dec3_2(d3, t_embed, cond_embed)
            d3 = self.dec3_3(d3, t_embed, cond_embed)
            d3 = self.dec3_attn(d3)

        # Level 2
       
        d2_in = torch.cat([self.up(d3), e2], dim=1)
        if self.use_checkpoint and self.training:
            d2 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec2_1), d2_in, t_embed, cond_embed, use_reentrant=False
            )
            d2 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec2_2), d2, t_embed, cond_embed, use_reentrant=False
            )
            d2 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec2_3), d2, t_embed, cond_embed, use_reentrant=False
            )
        else:
            d2 = self.dec2_1(d2_in, t_embed, cond_embed)
            d2 = self.dec2_2(d2, t_embed, cond_embed)
            d2 = self.dec2_3(d2, t_embed, cond_embed)

        # Level 1
        
        d1_in = torch.cat([self.up(d2), e1], dim=1)
        if self.use_checkpoint and self.training:
            d1 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec1_1), d1_in, t_embed, cond_embed, use_reentrant=False
            )
            d1 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec1_2), d1, t_embed, cond_embed, use_reentrant=False
            )
            d1 = torch.utils.checkpoint.checkpoint(
                create_forward_fn(self.dec1_3), d1, t_embed, cond_embed, use_reentrant=False
            )
        else:
            d1 = self.dec1_1(d1_in, t_embed, cond_embed)
            d1 = self.dec1_2(d1, t_embed, cond_embed)
            d1 = self.dec1_3(d1, t_embed, cond_embed)

        return self.out(d1)

# =======================
# 6. COMPLETE MODEL
# =======================

class AgeTransformationDiffusion(nn.Module):
    """Complete large-scale age transformation diffusion model"""
    def __init__(self, max_age=120, img_size=256):
        super().__init__()
        self.age_encoder = AgeEncoder(max_age=max_age, embed_dim=256)
        self.identity_encoder = IdentityEncoder(output_dim=768)
        self.unet = ConditionalUNet(
            img_channels=3,
            base_ch=192,
            time_dim=512,
            age_dim=256,
            identity_dim=768,
            dropout=0.1,
            use_checkpoint=True
        )

    def forward(self, x_noisy, t, current_age, target_age, x_clean):
        """
        Args:
            x_noisy: noisy image at timestep t (batch, 3, H, W)
            t: diffusion timestep (batch,)
            current_age: current age of person (batch,)
            target_age: target age for transformation (batch,)
            x_clean: clean reference image for identity (batch, 3, H, W)
        Returns:
            predicted_noise: (batch, 3, H, W)
        """
        identity_embed = self.identity_encoder(x_clean)
        target_age_embed = self.age_encoder(target_age)
        noise_pred = self.unet(x_noisy, t, target_age_embed, identity_embed)
       
        return noise_pred
# =======================
# 9. MODEL SUMMARY
# =======================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Create model and count parameters
    model = AgeTransformationDiffusion(max_age=100, img_size=256)
    
    total_params = count_parameters(model)
    identity_params = count_parameters(model.identity_encoder)
    age_params = count_parameters(model.age_encoder)
    unet_params = count_parameters(model.unet)
    
    print("=" * 60)
    print("LARGE-SCALE AGE TRANSFORMATION DIFFUSION MODEL")
    print("=" * 60)
    print(f"Total Parameters: {total_params:,}")
    print(f"  - Identity Encoder: {identity_params:,}")
    print(f"  - Age Encoder: {age_params:,}")
    print(f"  - U-Net: {unet_params:,}")
    print("=" * 60)
    print("\nKey Improvements:")
    print("  ✓ 192 base channels (2x increase)")
    print("  ✓ 3 residual blocks per level (3x increase)")
    print("  ✓ Multi-resolution attention (3 levels)")
    print("  ✓ Cross-attention conditioning")
    print("  ✓ Deeper identity encoder (5 levels)")
    print("  ✓ 6-block bottleneck with attention")
    print("  ✓ Enhanced embeddings (256/768 dims)")
    print("=" * 60)
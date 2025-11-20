"""
Vision Mamba - Selective State Space Model for Vision Control

Core architecture based on Mamba/S4 state space models.
No CNN, No Transformer, No Diffusion - Pure SSM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model - Mamba의 핵심

    입력에 따라 동적으로 선택적 정보 보존
    O(N) complexity - 트랜스포머의 O(N^2)보다 훨씬 빠름
    """

    def __init__(self, d_model, d_state=16, expand=2):
        """
        Args:
            d_model: 모델 차원
            d_state: state 차원 (메모리)
            expand: 내부 확장 비율
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # SSM parameters - 입력에 따라 동적으로 변함 (Selective!)
        self.delta_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)

        # A matrix - log space에서 학습
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Args:
            x: (B, L, D) - Batch, Length, Dimension

        Returns:
            y: (B, L, D)
        """
        B, L, D = x.shape

        # Input projection
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)

        # Activation
        x = F.silu(x)  # Swish activation

        # SSM computation
        y = self.selective_scan(x)

        # Gating with residual
        y = y * F.silu(res)

        # Output projection
        output = self.out_proj(y)

        return output

    def selective_scan(self, x):
        """
        Selective Scan - Mamba의 핵심 메커니즘

        입력에 따라 동적으로 파라미터를 조정하여
        중요한 정보는 기억하고 불필요한 정보는 망각
        """
        batch_size, L, D = x.shape

        # Delta (step size) - 각 타임스텝의 중요도
        delta = F.softplus(self.delta_proj(x))  # (B, L, D)

        # B, C matrices - 입력 의존적
        B_mat = self.B_proj(x)  # (B, L, d_state)
        C_mat = self.C_proj(x)  # (B, L, d_state)

        # A matrix
        A = -torch.exp(self.A_log.float())  # (D, d_state)

        # Discretization - continuous to discrete
        deltaA = torch.exp(delta.unsqueeze(-1) * A)  # (B, L, D, d_state)
        deltaB = delta.unsqueeze(-1) * B_mat.unsqueeze(2)  # (B, L, D, d_state)

        # Selective scan (parallel associative scan)
        # 이 부분이 실제로는 CUDA 커널로 최적화되어야 하지만,
        # 여기서는 간단한 recurrent 형태로 구현
        h = x.new_zeros(batch_size, D, self.d_state)
        hs = []

        x_expanded = x.unsqueeze(-1)  # (B, L, D, 1)

        for i in range(L):
            h = deltaA[:, i] * h + deltaB[:, i] * x_expanded[:, i]
            y_i = (h * C_mat[:, i].unsqueeze(1)).sum(dim=-1)  # (B, D)
            hs.append(y_i)

        y = torch.stack(hs, dim=1)  # (B, L, D)

        # Skip connection
        y = y + x * self.D

        return y


class MambaBlock(nn.Module):
    """
    Mamba Block - Normalization + SSM + MLP
    """

    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state, expand)

    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            (B, L, D)
        """
        # Pre-norm + SSM with residual
        return x + self.ssm(self.norm(x))


class PatchEmbedding(nn.Module):
    """
    이미지를 패치로 나누고 임베딩 (CNN 없이!)
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Linear projection (CNN 아님!)
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, embed_dim)
        """
        B = x.shape[0]

        # Patchify
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, 'b c h w -> b (h w) c')

        # Add positional encoding
        x = x + self.pos_embed

        return x


class VisionMamba(nn.Module):
    """
    Vision Mamba - 완전한 비전 인코더

    No CNN (only initial patchify), No Transformer, No Attention
    Pure Selective State Space Model
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=192,
        depth=12,
        d_state=16,
        expand=2
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )

        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(embed_dim, d_state, expand)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - input image

        Returns:
            (B, num_patches, embed_dim) - encoded features
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Apply Mamba blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        return x

    def get_global_feature(self, x):
        """
        전체 이미지의 global feature 추출
        """
        features = self.forward(x)  # (B, num_patches, embed_dim)

        # Global average pooling
        global_feat = features.mean(dim=1)  # (B, embed_dim)

        return global_feat


def create_vision_mamba_tiny():
    """경량 모델 - 데모용"""
    return VisionMamba(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=6,  # 얕게
        d_state=8,
        expand=2
    )


def create_vision_mamba_small():
    """중간 모델"""
    return VisionMamba(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        d_state=16,
        expand=2
    )


def create_vision_mamba_base():
    """기본 모델"""
    return VisionMamba(
        img_size=224,
        patch_size=16,
        embed_dim=512,
        depth=24,
        d_state=16,
        expand=2
    )


if __name__ == "__main__":
    # 테스트
    model = create_vision_mamba_tiny()

    # Dummy input
    x = torch.randn(1, 3, 224, 224)

    # Forward pass
    features = model(x)
    global_feat = model.get_global_feature(x)

    print(f"Input shape: {x.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Global feature shape: {global_feat.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

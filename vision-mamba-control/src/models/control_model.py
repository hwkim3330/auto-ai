"""
Vision Mamba Control Model - Camera-adaptive control with FiLM

Uses FiLM (Feature-wise Linear Modulation) to adapt to different camera conditions.
Outputs control signals: steering, throttle, brake
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .mamba import VisionMamba, create_vision_mamba_tiny


class FiLMLayer(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) Layer

    카메라 조건에 따라 feature를 동적으로 조정
    - 밝기, 대비, 색온도 등에 적응
    - y = gamma * x + beta (채널별로)
    """

    def __init__(self, num_features, condition_dim=16):
        """
        Args:
            num_features: feature 차원 (embed_dim)
            condition_dim: 조건 벡터 차원
        """
        super().__init__()

        # Condition encoder (카메라 상태 → 조건 벡터)
        self.condition_encoder = nn.Sequential(
            nn.Linear(3, 32),  # 입력: [brightness, contrast, saturation]
            nn.ReLU(),
            nn.Linear(32, condition_dim),
            nn.ReLU()
        )

        # FiLM 파라미터 생성
        self.gamma_generator = nn.Linear(condition_dim, num_features)
        self.beta_generator = nn.Linear(condition_dim, num_features)

        # 기본값으로 초기화 (gamma=1, beta=0)
        nn.init.ones_(self.gamma_generator.weight)
        nn.init.zeros_(self.gamma_generator.bias)
        nn.init.zeros_(self.beta_generator.weight)
        nn.init.zeros_(self.beta_generator.bias)

    def forward(self, x, camera_stats):
        """
        Args:
            x: (B, L, D) - feature tensor
            camera_stats: (B, 3) - [brightness, contrast, saturation]

        Returns:
            (B, L, D) - modulated features
        """
        # Condition 임베딩
        condition = self.condition_encoder(camera_stats)  # (B, condition_dim)

        # FiLM 파라미터 생성
        gamma = self.gamma_generator(condition).unsqueeze(1)  # (B, 1, D)
        beta = self.beta_generator(condition).unsqueeze(1)    # (B, 1, D)

        # Affine transformation
        return gamma * x + beta


class ActionHead(nn.Module):
    """
    Action Prediction Head

    Vision features → Control actions
    - Steering: [-1, 1] (왼쪽 ~ 오른쪽)
    - Throttle: [0, 1] (정지 ~ 최대 가속)
    - Brake: [0, 1] (브레이크 없음 ~ 최대 제동)
    """

    def __init__(self, embed_dim=192, hidden_dim=256):
        super().__init__()

        # Feature aggregation
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Action prediction network
        self.action_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # [steering, throttle, brake]
        )

    def forward(self, features):
        """
        Args:
            features: (B, num_patches, embed_dim)

        Returns:
            actions: (B, 3) - [steering, throttle, brake]
        """
        # Global pooling
        B, L, D = features.shape
        x = features.transpose(1, 2)  # (B, D, L)
        x = self.pool(x).squeeze(-1)  # (B, D)

        # Predict actions
        actions = self.action_net(x)  # (B, 3)

        # Apply activation functions
        steering = torch.tanh(actions[:, 0])      # [-1, 1]
        throttle = torch.sigmoid(actions[:, 1])   # [0, 1]
        brake = torch.sigmoid(actions[:, 2])      # [0, 1]

        return torch.stack([steering, throttle, brake], dim=1)


class VisionMambaControl(nn.Module):
    """
    Complete Vision Mamba Control System

    Camera → Vision Mamba → FiLM → Actions

    - 웹캠 입력 처리
    - 카메라 조건에 적응 (FiLM)
    - 제어 신호 출력 (steering, throttle, brake)
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=6,
        use_film=True
    ):
        super().__init__()

        self.use_film = use_film

        # Vision encoder (Mamba)
        self.vision_encoder = VisionMamba(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            d_state=8,
            expand=2
        )

        # FiLM layer (카메라 적응)
        if use_film:
            self.film = FiLMLayer(embed_dim, condition_dim=16)

        # Action head
        self.action_head = ActionHead(embed_dim, hidden_dim=256)

    def forward(
        self,
        image: torch.Tensor,
        camera_stats: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 3, H, W) - RGB image
            camera_stats: (B, 3) - [brightness, contrast, saturation]

        Returns:
            dict with:
                - actions: (B, 3) - [steering, throttle, brake]
                - features: (B, num_patches, embed_dim)
        """
        # Vision encoding
        features = self.vision_encoder(image)  # (B, num_patches, embed_dim)

        # FiLM conditioning (선택적)
        if self.use_film and camera_stats is not None:
            features = self.film(features, camera_stats)

        # Action prediction
        actions = self.action_head(features)  # (B, 3)

        return {
            'actions': actions,
            'features': features,
            'steering': actions[:, 0],
            'throttle': actions[:, 1],
            'brake': actions[:, 2]
        }

    def predict_from_webcam(
        self,
        frame,
        brightness: float = 0.5,
        contrast: float = 0.5,
        saturation: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        웹캠 프레임에서 직접 예측

        Args:
            frame: numpy array (H, W, 3) in BGR
            brightness, contrast, saturation: 카메라 통계 (0-1)

        Returns:
            (steering, throttle, brake) as floats
        """
        import cv2
        import numpy as np

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # Normalize to [0, 1]
        frame_norm = frame_resized.astype(np.float32) / 255.0

        # To tensor (B, C, H, W)
        image_tensor = torch.from_numpy(frame_norm).permute(2, 0, 1).unsqueeze(0).float()

        # Camera stats
        camera_stats_tensor = torch.tensor([[brightness, contrast, saturation]], dtype=torch.float32)

        # Predict
        with torch.no_grad():
            output = self.forward(image_tensor, camera_stats_tensor)

        steering = output['steering'].item()
        throttle = output['throttle'].item()
        brake = output['brake'].item()

        return steering, throttle, brake


def create_control_model_tiny(use_film=True):
    """경량 제어 모델 - 웹캠 데모용"""
    return VisionMambaControl(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        depth=6,
        use_film=use_film
    )


def create_control_model_base(use_film=True):
    """기본 제어 모델"""
    return VisionMambaControl(
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        use_film=use_film
    )


if __name__ == "__main__":
    # 테스트
    model = create_control_model_tiny()

    # Dummy input
    image = torch.randn(1, 3, 224, 224)
    camera_stats = torch.tensor([[0.5, 0.5, 0.5]])  # 중간 밝기/대비/채도

    # Forward pass
    output = model(image, camera_stats)

    print(f"Image shape: {image.shape}")
    print(f"Actions shape: {output['actions'].shape}")
    print(f"Steering: {output['steering'].item():.3f}")
    print(f"Throttle: {output['throttle'].item():.3f}")
    print(f"Brake: {output['brake'].item():.3f}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

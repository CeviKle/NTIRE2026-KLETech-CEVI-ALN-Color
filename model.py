import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# NAF Block
# =========================

class NAFBlock(nn.Module):

    def __init__(self, c):
        super().__init__()

        self.norm1 = nn.GroupNorm(1, c)

        self.pw1 = nn.Conv2d(c, c*2, 1)
        self.dw = nn.Conv2d(c*2, c*2, 3, padding=1, groups=c*2)
        self.pw2 = nn.Conv2d(c*2, c, 1)

        self.norm2 = nn.GroupNorm(1, c)

        self.ffn1 = nn.Conv2d(c, c*2, 1)
        self.ffn2 = nn.Conv2d(c*2, c, 1)

        self.act = nn.GELU()

    def forward(self, x):

        y = self.norm1(x)

        y = self.pw1(y)
        y = self.act(y)
        y = self.dw(y)
        y = self.pw2(y)

        x = x + y

        y = self.norm2(x)

        y = self.ffn1(y)
        y = self.act(y)
        y = self.ffn2(y)

        return x + y


# =========================
# Illumination Network
# =========================

class IlluminationNet(nn.Module):

    def __init__(self, feat=64, blocks=8):
        super().__init__()

        self.inp = nn.Conv2d(3, feat, 3, padding=1)

        self.low_down = nn.Conv2d(feat, feat, 5, stride=2, padding=2)

        self.body_low = nn.Sequential(
            *[NAFBlock(feat) for _ in range(blocks//2)]
        )

        self.body_high = nn.Sequential(
            *[NAFBlock(feat) for _ in range(blocks//2)]
        )

        self.fuse = nn.Conv2d(feat*2, feat, 1)

        self.out = nn.Conv2d(feat, 3, 3, padding=1)

    def forward(self, x):

        f = self.inp(x)

        low = self.low_down(f)
        low = self.body_low(low)

        low = F.interpolate(
            low,
            size=f.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        high = self.body_high(f)

        fused = torch.cat([low, high], dim=1)
        fused = self.fuse(fused)

        o = self.out(fused)

        return torch.clamp(x + o, 0.0, 1.0)


# =========================
# Refinement Network
# =========================

class RefinementNet(nn.Module):

    def __init__(self, feat=64, blocks=12):
        super().__init__()

        self.inp = nn.Conv2d(3, feat, 3, padding=1)

        self.body = nn.Sequential(
            *[NAFBlock(feat) for _ in range(blocks)]
        )

        self.out = nn.Conv2d(feat, 3, 3, padding=1)

    def forward(self, x):

        f = self.inp(x)

        res = self.body(f)
        f = f + res

        o = self.out(f)

        return torch.clamp(x + o, 0.0, 1.0)


# =========================
# Final Model
# =========================

class final_net(nn.Module):

    def __init__(self):
        super().__init__()

        self.remove_model = IlluminationNet()
        self.enhancement_model = RefinementNet()

    def forward(self, x):

        stage1 = self.remove_model(x)

        stage2 = self.enhancement_model(stage1)

        return stage1, stage2
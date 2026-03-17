import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# -------------------- LayerNorm --------------------
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        self.body = BiasFree_LayerNorm(dim) if LayerNorm_type=='BiasFree' else WithBias_LayerNorm(dim)
    def forward(self, x):
        h,w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

# -------------------- FeedForward --------------------
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, 1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, 3, 1, 1, groups=hidden_features*2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)

# -------------------- Attention (FSAS) --------------------
class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.to_hidden = nn.Conv2d(dim, dim*6, 1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim*6, dim*6, 3, 1, 1, groups=dim*6, bias=bias)
        self.project_out = nn.Conv2d(dim*2, dim, 1, bias=bias)
        self.norm = LayerNorm(dim*2, 'WithBias')
        self.base_patch_size = 2
        self.chunk_size = 16

    def forward(self, x):
        b,c,H,W = x.shape
        hidden = self.to_hidden(x)
        q,k,v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        patch_h, patch_w = min(self.base_patch_size,H), min(self.base_patch_size,W)
        q_patch = rearrange(q, 'b c (h ph) (w pw) -> b c h w ph pw', ph=patch_h, pw=patch_w)
        k_patch = rearrange(k, 'b c (h ph) (w pw) -> b c h w ph pw', ph=patch_h, pw=patch_w)

        out = torch.zeros_like(q_patch, dtype=torch.float32)
        for i in range(0, q_patch.shape[1], self.chunk_size):
            q_chunk = q_patch[:, i:i+self.chunk_size].float()
            k_chunk = k_patch[:, i:i+self.chunk_size].float()
            out[:, i:i+self.chunk_size] = torch.fft.irfft2(torch.fft.rfft2(q_chunk) * torch.fft.rfft2(k_chunk),
                                                           s=(patch_h, patch_w))

        out = rearrange(out, 'b c h w ph pw -> b c (h ph) (w pw)', ph=patch_h, pw=patch_w)
        out = self.norm(out)
        output = v * out
        return self.project_out(output)

# -------------------- Transformer Block --------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# -------------------- Patch Embed --------------------
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, 3, 1, 1)
    def forward(self, x): return self.proj(x)

# -------------------- Downsample/Upsample --------------------
class Downsample(nn.Module):
    def __init__(self, n_feat): super().__init__(); self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, 3, 1,1), nn.PixelUnshuffle(2))
    def forward(self,x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat): super().__init__(); self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, 3, 1,1), nn.PixelShuffle(2))
    def forward(self,x): return self.body(x)

# -------------------- Restormer --------------------
class Restormer(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=32,
                 num_blocks=[2,3,3,4], num_refinement_blocks=2,
                 heads=[1,2,4,8]):

        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim, heads[0]) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim*2, heads[1]) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(dim*2)
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim*4, heads[2]) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(dim*4)
        self.latent = nn.Sequential(*[TransformerBlock(dim*8, heads[3]) for _ in range(num_blocks[3])])
        self.up4_3 = Upsample(dim*8)
        self.reduce_chan_level3 = nn.Conv2d(dim*8, dim*4, 1)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim*4, heads[2]) for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(dim*4)
        self.reduce_chan_level2 = nn.Conv2d(dim*4, dim*2, 1)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim*2, heads[1]) for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(dim*2)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim*2, heads[0]) for _ in range(num_blocks[0])])
        self.refinement = nn.Sequential(*[TransformerBlock(dim*2, heads[0]) for _ in range(num_refinement_blocks)])
        self.output = nn.Conv2d(dim*2, out_channels, 3, 1,1)

    def forward(self, inp_img):
        enc1 = self.patch_embed(inp_img)
        out1 = self.encoder_level1(enc1)
        enc2 = self.down1_2(out1); out2 = self.encoder_level2(enc2)
        enc3 = self.down2_3(out2); out3 = self.encoder_level3(enc3)
        enc4 = self.down3_4(out3); latent = self.latent(enc4)
        dec3 = self.up4_3(latent); dec3 = torch.cat([dec3, out3],1); dec3 = self.reduce_chan_level3(dec3); out_dec3 = self.decoder_level3(dec3)
        dec2 = self.up3_2(out_dec3); dec2 = torch.cat([dec2, out2],1); dec2 = self.reduce_chan_level2(dec2); out_dec2 = self.decoder_level2(dec2)
        dec1 = self.up2_1(out_dec2); dec1 = torch.cat([dec1, out1],1); out_dec1 = self.decoder_level1(dec1)
        out_dec1 = self.refinement(out_dec1)
        return self.output(out_dec1) + inp_img

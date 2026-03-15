from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def normalize(in_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.conv = (
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            if with_conv
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.conv is not None:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.conv = (
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
            if with_conv
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return F.avg_pool2d(x, kernel_size=2, stride=2)
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float,
        temb_channels: int = 512,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = (
            nn.Linear(temb_channels, self.out_channels) if temb_channels > 0 else None
        )
        self.norm2 = normalize(self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)

        self.conv_shortcut = None
        self.nin_shortcut = None
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels,
                    self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x: torch.Tensor, temb: torch.Tensor | None) -> torch.Tensor:
        h = self.conv1(nonlinearity(self.norm1(x)))
        if self.temb_proj is not None and temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        elif self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.norm = normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h_size, w_size = q.shape
        q = q.reshape(b, c, h_size * w_size).permute(0, 2, 1)
        k = k.reshape(b, c, h_size * w_size)
        weights = torch.bmm(q, k) * (int(c) ** -0.5)
        weights = F.softmax(weights, dim=2)

        v = v.reshape(b, c, h_size * w_size)
        weights = weights.permute(0, 2, 1)
        h = torch.bmm(v, weights).reshape(b, c, h_size, w_size)
        return x + self.proj_out(h)


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: tuple[int, ...] | list[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res //= 2
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = [self.conv_in(x)]
        for level in range(self.num_resolutions):
            for block_idx in range(self.num_res_blocks):
                h = self.down[level].block[block_idx](hs[-1], None)
                if len(self.down[level].attn) > 0:
                    h = self.down[level].attn[block_idx](h)
                hs.append(h)
            if level != self.num_resolutions - 1:
                hs.append(self.down[level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        h = self.conv_out(nonlinearity(self.norm_out(h)))
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: tuple[int, ...] | list[int] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: tuple[int, ...] | list[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        give_pre_end: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.give_pre_end = give_pre_end

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, temb_channels=0, dropout=dropout)

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=0,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up)

        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h, None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)

        for level in reversed(range(self.num_resolutions)):
            for block_idx in range(self.num_res_blocks + 1):
                h = self.up[level].block[block_idx](h, None)
                if len(self.up[level].attn) > 0:
                    h = self.up[level].attn[block_idx](h)
            if level != 0:
                h = self.up[level].upsample(h)

        if self.give_pre_end:
            return h
        return self.conv_out(nonlinearity(self.norm_out(h)))


class VectorQuantizer2(nn.Module):
    def __init__(self, n_e: int, e_dim: int, beta: float) -> None:
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[None, None, torch.Tensor]]:
        z = rearrange(z, "b c h w -> b h w c").contiguous()
        z_flattened = z.view(-1, self.e_dim)
        distances = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(self.embedding.weight, "n d -> d n"))
        )
        min_encoding_indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, "b h w c -> b c h w").contiguous()
        return z_q, loss, (None, None, min_encoding_indices)

    def get_codebook_entry(self, indices: torch.Tensor, shape: tuple[int, int, int, int]) -> torch.Tensor:
        z_q = self.embedding(indices).view(shape)
        return z_q.permute(0, 3, 1, 2).contiguous()

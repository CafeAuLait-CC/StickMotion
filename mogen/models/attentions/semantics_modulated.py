import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.stylization_block import StylizationBlock
from ..builder import ATTENTIONS
from einops import rearrange


class SubAttention(nn.Module):
    def __init__(self, latent_dim, cond_latent_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.norm_x = nn.LayerNorm(latent_dim)
        self.norm_cond = nn.LayerNorm(cond_latent_dim)

        self.key_cond = nn.Linear(cond_latent_dim, latent_dim)
        self.value_cond = nn.Linear(cond_latent_dim, latent_dim)
        self.key_x = nn.Linear(latent_dim, latent_dim)
        self.value_x = nn.Linear(latent_dim, latent_dim)

        self.proj_y = nn.Linear(latent_dim, latent_dim)

    # from line_profiler import profile
    # @profile
    def forward(self, query, x, cond_emb, src_mask, cond_mask):
        """
        x: B, T, H, D
        """
        B, T, D = x.shape
        N = x.shape[1] + cond_emb.shape[1]
        H = self.num_heads
        # B, N, D

        key = torch.cat(
            (
                self.key_cond(self.norm_cond(cond_emb))
                + (1 - cond_mask) * -1000000,  # -inf: 10%
                self.key_x(self.norm_x(x)) + (1 - src_mask) * -1000000,
            ),
            dim=1,
        )  # bnhd [256, 371, 512]
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        # re_feat_value = cond_emb.reshape(B, -1, D)
        value = torch.cat(
            (
                self.value_cond(self.norm_cond(cond_emb)) * cond_mask,
                self.value_x(self.norm_x(x)) * src_mask,
            ),
            dim=1,
        ).view(B, N, H, -1)  # bnhl l=d
        # B, H, HD, HD
        attention = torch.einsum("bnhd,bnhl->bhdl", key, value)
        y = torch.einsum("bnhd,bhdl->bnhl", query, attention).reshape(B, T, D)
        y = self.proj_y(y)
        return y


class SelfAttention(nn.Module):
    def __init__(self, latent_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.norm_x = nn.LayerNorm(latent_dim)

        self.key_x = nn.Linear(latent_dim, latent_dim)
        self.value_x = nn.Linear(latent_dim, latent_dim)

        self.proj_y = nn.Linear(latent_dim, latent_dim)

    # from line_profiler import profile
    # @profile
    def forward(self, query, x, src_mask):
        """
        x: B, T, H, D
        """
        B, T, D = x.shape
        N = x.shape[1]
        H = self.num_heads
        # B, N, D

        key = (
            self.key_x(self.norm_x(x)) + (1 - src_mask) * -1000000
        )  # bnhd [256, 371, 512]
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        # B, N, H, HD
        # re_feat_value = cond_emb.reshape(B, -1, D)
        value = (self.value_x(self.norm_x(x)) * src_mask).view(B, N, H, -1)  # bnhl l=d
        # B, H, HD, HD
        attention = torch.einsum("bnhd,bnhl->bhdl", key, value)
        y = torch.einsum("bnhd,bhdl->bnhl", query, attention).reshape(B, T, D)
        y = self.proj_y(y)
        return y


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


@ATTENTIONS.register_module()
class SemanticsModulatedAttention(nn.Module):
    def __init__(
        self,
        latent_dim,
        text_latent_dim,
        joint_latent_dim,
        # stick_latent_dim,
        num_heads,
        dropout,
        time_embed_dim,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.text_encoder = SubAttention(latent_dim, text_latent_dim, num_heads)
        self.joint_encoder = SubAttention(latent_dim, joint_latent_dim, num_heads)
        # self.stick_encoder = SubAttention(latent_dim, stick_latent_dim, num_heads)
        self.y_encoder = SelfAttention(latent_dim, num_heads)
        # self.mid_proj = nn.Sequential(
        #     nn.Linear(latent_dim, latent_dim),
        #     nn.LeakyReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )

        self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

    # from line_profiler import profile
    # @profile
    def forward(
        self, x, text_emb, joint_emb, other_emb, src_mask, cond_type, joint_mask
    ):
        # def forward(self, x, text_emb, stick_emb, other_emb, src_mask, cond_type, stick_mask):
        """
        x: B, T, D
        xf: B, N, L # text features; re_dict: retrieval information
        cond_type [text, both, stick, none].sum() == batch_size
        """
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        ci = [sum(cond_type[:i]) for i in range(len(cond_type))]

        text_query = query[: ci[2]]
        text_x = x[: ci[2]]
        text_x_mask = src_mask[: ci[2]]
        text_emb = text_emb[: ci[2]]

        # stick_query = query[ci[1]:ci[3]]
        # stick_x = x[ci[1]:ci[3]]
        # stick_x_mask = src_mask[ci[1]:ci[3]]
        # stick_emb = stick_emb[ci[1]:ci[3]]
        # stick_mask = stick_mask[ci[1]:ci[3]]

        joint_query = query[ci[1] : ci[3]]
        joint_x = x[ci[1] : ci[3]]
        joint_x_mask = src_mask[ci[1] : ci[3]]
        joint_emb = joint_emb[ci[1] : ci[3]]
        joint_mask = joint_mask[ci[1] : ci[3]]

        text_y = self.text_encoder(text_query, text_x, text_emb, text_x_mask, 1)
        joint_y = self.joint_encoder(
            joint_query, joint_x, joint_emb, joint_x_mask, joint_mask
        )
        # stick_y = self.stick_encoder(stick_query, stick_x, stick_emb, stick_x_mask, stick_mask)
        query[: ci[2]] = query[: ci[2]] + text_y
        query[ci[1] : ci[3]] = query[ci[1] : ci[3]] + joint_y
        # query[ci[1]:ci[3]] = query[ci[1]:ci[3]] + stick_y
        query = self.y_encoder(query, x, src_mask)

        y = x + self.proj_out(query, other_emb)

        return y


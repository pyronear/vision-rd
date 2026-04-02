"""Tests for smokeynet_adapted.spatial_attention."""

import torch

from smokeynet_adapted.spatial_attention import SpatialAttentionViT


class TestSpatialAttentionViT:
    def test_output_shape(self):
        vit = SpatialAttentionViT(d_model=32, nhead=4, num_layers=2)
        tube_emb = torch.randn(3, 32)
        bbox_coords = torch.randn(3, 4)
        out = vit(tube_emb, bbox_coords)
        assert out.shape == (32,)

    def test_single_tube(self):
        vit = SpatialAttentionViT(d_model=16, nhead=4, num_layers=1)
        tube_emb = torch.randn(1, 16)
        out = vit(tube_emb)
        assert out.shape == (16,)

    def test_no_bbox_coords(self):
        vit = SpatialAttentionViT(d_model=16, nhead=4, num_layers=1)
        tube_emb = torch.randn(2, 16)
        out = vit(tube_emb, bbox_coords=None)
        assert out.shape == (16,)

    def test_empty_input(self):
        vit = SpatialAttentionViT(d_model=16, nhead=4, num_layers=1)
        tube_emb = torch.zeros(0, 16)
        out = vit(tube_emb)
        assert out.shape == (16,)
        assert torch.all(out == 0)

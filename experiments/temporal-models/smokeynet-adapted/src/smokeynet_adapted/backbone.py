"""YOLO backbone feature extraction with RoI Align.

Runs the full image through the YOLO backbone to obtain feature maps, then
uses ``torchvision.ops.roi_align`` to extract fixed-size features at each
detection bbox.  This avoids the scale mismatch that would result from
cropping small regions and resizing them.
"""

import torch
from torch import Tensor, nn
from torchvision.ops import roi_align


class YoloRoiExtractor(nn.Module):
    """Extract per-detection features via YOLO backbone + RoI Align.

    Pipeline per frame:
      1. Forward full image through YOLO backbone -> feature map
      2. Expand bboxes by ``context_factor``, convert to absolute coords
      3. RoI Align at each bbox -> ``(N, C, roi_size, roi_size)``
      4. Global average pool -> ``(N, C)``
      5. Linear projection -> ``(N, d_model)``

    Args:
        yolo_model: A loaded YOLO model from ``ultralytics``.
        d_model: Output feature dimension.
        roi_size: Spatial size of RoI Align output (default 7).
        context_factor: Bbox expansion factor before RoI Align (default 1.2).
        feature_layer_idx: Which backbone layer to extract features from.
            Defaults to -1 (last layer).
    """

    def __init__(
        self,
        yolo_model: nn.Module,
        d_model: int = 512,
        roi_size: int = 7,
        context_factor: float = 1.2,
        feature_layer_idx: int = -1,
    ) -> None:
        super().__init__()
        self.backbone = yolo_model.model.model[: self._backbone_end_idx(yolo_model)]
        self.roi_size = roi_size
        self.context_factor = context_factor
        self.feature_layer_idx = feature_layer_idx

        # Determine backbone output channels by a dummy forward pass
        backbone_channels = self._get_backbone_channels(yolo_model)
        self.projection = nn.Linear(backbone_channels, d_model)

    @staticmethod
    def _backbone_end_idx(yolo_model: nn.Module) -> int:
        """Find the index where the backbone ends in the YOLO sequential model.

        YOLO11 models have a ``model.model`` sequential with backbone layers
        followed by neck/head layers.  The backbone typically ends at index 10
        (before the Detect head).  We look for the save indices to find the
        backbone boundary.
        """
        model_list = yolo_model.model.model
        # The backbone is typically the first ~10 layers.
        # We use a heuristic: find the last Conv/C2f/C3k2/SPPF layer
        # before the first Concat or Detect layer.
        for i, layer in enumerate(model_list):
            cls_name = type(layer).__name__
            if cls_name in ("Concat", "Detect", "Segment"):
                return i
        return len(model_list)

    def _get_backbone_channels(self, yolo_model: nn.Module) -> int:
        """Determine backbone output channel count via a dummy forward."""
        device = next(self.backbone.parameters()).device
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        with torch.no_grad():
            out = self._run_backbone(dummy)
        return out.shape[1]

    def _run_backbone(self, x: Tensor) -> Tensor:
        """Forward through backbone layers, returning the last feature map."""
        for layer in self.backbone:
            x = layer(x)
        return x

    def extract_features(
        self,
        image: Tensor,
        bboxes_norm: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Extract RoI-aligned features for detections in a single frame.

        Args:
            image: ``(1, 3, H, W)`` normalised image tensor.
            bboxes_norm: ``(N, 4)`` normalised bboxes ``(cx, cy, w, h)``
                in [0, 1].
            image_size: ``(H, W)`` of the image for coord conversion.

        Returns:
            ``(N, d_model)`` feature tensor, or ``(0, d_model)`` if no bboxes.
        """
        device = next(self.backbone.parameters()).device
        d_model = self.projection.out_features
        if bboxes_norm.shape[0] == 0:
            return torch.zeros(0, d_model, device=device)

        # Move inputs to model device
        image = image.to(device)
        bboxes_norm = bboxes_norm.to(device)

        # 1. Backbone forward
        feat_map = self._run_backbone(image)

        # 2. Convert normalised (cx,cy,w,h) to absolute (x1,y1,x2,y2)
        h_img, w_img = image_size
        rois_abs = _norm_cxcywh_to_abs_xyxy(
            bboxes_norm, w_img, h_img, self.context_factor
        )

        # 3. Scale rois to feature map coordinates
        feat_h, feat_w = feat_map.shape[2], feat_map.shape[3]
        scale_x = feat_w / w_img
        scale_y = feat_h / h_img
        rois_scaled = rois_abs.clone()
        rois_scaled[:, 0] *= scale_x
        rois_scaled[:, 2] *= scale_x
        rois_scaled[:, 1] *= scale_y
        rois_scaled[:, 3] *= scale_y

        # Prepend batch index (all from batch 0)
        batch_idx = torch.zeros(rois_scaled.shape[0], 1, device=rois_scaled.device)
        rois_with_batch = torch.cat([batch_idx, rois_scaled], dim=1)

        # 4. RoI Align
        pooled = roi_align(
            feat_map,
            rois_with_batch,
            output_size=self.roi_size,
            spatial_scale=1.0,  # already scaled
            aligned=True,
        )  # (N, C, roi_size, roi_size)

        # 5. Global average pool -> (N, C)
        pooled = pooled.mean(dim=[2, 3])

        # 6. Project -> (N, d_model)
        return self.projection(pooled)

    def forward(
        self,
        image: Tensor,
        bboxes_norm: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Alias for :meth:`extract_features`."""
        return self.extract_features(image, bboxes_norm, image_size)


def _norm_cxcywh_to_abs_xyxy(
    bboxes: Tensor,
    img_w: int,
    img_h: int,
    context_factor: float = 1.0,
) -> Tensor:
    """Convert normalised (cx, cy, w, h) to absolute (x1, y1, x2, y2).

    Optionally expands each bbox by ``context_factor`` and clamps to image
    bounds.

    Args:
        bboxes: ``(N, 4)`` tensor of normalised bboxes.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        context_factor: Expansion factor (1.0 = no expansion).

    Returns:
        ``(N, 4)`` tensor of absolute corner coordinates.
    """
    cx = bboxes[:, 0] * img_w
    cy = bboxes[:, 1] * img_h
    w = bboxes[:, 2] * img_w * context_factor
    h = bboxes[:, 3] * img_h * context_factor

    x1 = (cx - w / 2).clamp(min=0)
    y1 = (cy - h / 2).clamp(min=0)
    x2 = (cx + w / 2).clamp(max=img_w)
    y2 = (cy + h / 2).clamp(max=img_h)

    return torch.stack([x1, y1, x2, y2], dim=1)

import numpy as np
import cv2
import torch

from common.interfaces import IdSpecialist, BBox


class PersonTorchreid(IdSpecialist):
    """
    Person re-identification specialist using torchreid OSNet x0.25.

    Why osnet_x0_25?
      - Lightweight (Pi 5 friendly) but strong for person ReID.
      - Widely used baseline with good generalization.
      - torchreid auto-downloads pretrained weights; no ONNX file needed.

    Output:
      - L2-normalized embedding vector (D,) as np.float32 suitable for cosine similarity.

    Config keys:
      - model_name: one of torchreid’s OSNet variants (default: "osnet_x0_25")
      - crop: square resize (default: 192)
      - min_bbox_h_frac: 0..1; gate tiny boxes (default: 0.08)
      - min_sharpness: Laplacian var threshold (default: 25.0)
    """

    name = "person.osnet_x025.torchreid"
    classes = ["person"]

    def __init__(self, cfg: dict):
        self.model_name = cfg.get("model_name", "osnet_x0_25")
        self.crop = int(cfg.get("crop", 192))
        self.min_h_frac = float(cfg.get("min_bbox_h_frac", 0.08))
        self.min_sharp = float(cfg.get("min_sharpness", 25.0))

        import torchreid

        self.device = torch.device("cpu")
        self.model = torchreid.models.build_model(name=self.model_name, num_classes=1000)
        # download and load pretrained weights
        url = torchreid.utils.download_pretrained_weights(self.model_name)
        torchreid.utils.load_pretrained_weights(self.model, url)
        self.model.eval().to(self.device)
        # remove classifier head if present to expose global features
        if hasattr(self.model, "classifier"):
            self.model.classifier = torch.nn.Identity()

        # simple normalization (ImageNet-ish)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def wants(self, det_conf: float, bbox: BBox, quality) -> bool:
        return (
            quality.get("bbox_h_frac", 0.0) >= self.min_h_frac
            and quality.get("sharpness", 0.0) >= self.min_sharp
        )

    def _preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        h, w = bgr.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        sq = bgr[y0 : y0 + side, x0 : x0 + side]
        img = cv2.resize(sq, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        x = np.transpose(rgb, (2, 0, 1))[None, ...]  # NCHW
        return torch.from_numpy(x).to(self.device)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(crop_bgr)
        with torch.no_grad():
            feat = self.model(x)  # shape: [1, D]
        e = feat.cpu().numpy().reshape(-1).astype(np.float32)
        e /= np.linalg.norm(e) + 1e-9
        return e

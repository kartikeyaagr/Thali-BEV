import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import timm


class KhanaClassifier:
    """Wraps a trained ConvNeXt .pt checkpoint for per-crop inference."""

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        model_name: str = "convnext_base",
        img_size: int = 224,
        device: str | None = None,
    ):
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.class_names = class_names
        self.model = self._load(Path(model_path), model_name, len(class_names))
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _load(self, path: Path, model_name: str, num_classes: int):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Case 1: torch.save(model, path) — full model object
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint

        # Case 2: {"state_dict": ..., ...} — training checkpoint
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            # Strip "module." prefix from DataParallel if present
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model = timm.create_model(model_name, num_classes=num_classes)
            model.load_state_dict(state_dict)
            return model

        # Case 3: training checkpoint with "model"/"ema" top-level keys
        # Prefer "ema" (averaged weights, higher accuracy) when available.
        if isinstance(checkpoint, dict) and ("ema" in checkpoint or "model" in checkpoint):
            raw = checkpoint.get("ema") or checkpoint.get("model")
            state_dict = {k.replace("module.", ""): v for k, v in raw.items()}
            model = timm.create_model(model_name, num_classes=num_classes)
            model.load_state_dict(state_dict)
            return model

        # Case 4: raw state_dict
        if isinstance(checkpoint, dict):
            state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            model = timm.create_model(model_name, num_classes=num_classes)
            model.load_state_dict(state_dict)
            return model

        raise ValueError(f"Unrecognised checkpoint format in {path}")

    def predict(self, crop: Image.Image) -> tuple[str, float]:
        """Return (class_name, confidence) for a single PIL crop."""
        tensor = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)
        return self.class_names[idx.item()], conf.item()

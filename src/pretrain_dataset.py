import json
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SingleFrameDataset(Dataset):
    """
    Dataset riêng cho pre-train CNN trên ảnh đơn.

    Mỗi sample gồm:
        image:  [3, 90, 160]
        target: [2] = (speed, course) đã normalize
    """

    def __init__(
        self,
        csv_file,
        images_root,
        telemetry_root,
        transform=None,
        timestamp_mode="mid",
    ):
        self.data = pd.read_csv(csv_file)
        self.images_root = images_root
        self.telemetry_root = telemetry_root
        self.transform = transform
        self.timestamp_mode = timestamp_mode

    def __len__(self):
        return len(self.data)

    def _pick_timestamp(self, start, end):
        if self.timestamp_mode == "start":
            return start
        if self.timestamp_mode == "end":
            return end
        # default: midpoint
        return start + (end - start) * 0.5

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_id = row["video_id"]
        start = float(row["start"])
        end = float(row["end"])

        t = self._pick_timestamp(start, end)

        # BDD-X frame naming: frame_{int(t*5)+1}.jpg
        frame_idx = int(t * 5) + 1
        img_path = os.path.join(self.images_root, video_id, f"frame_{frame_idx}.jpg")

        if os.path.exists(img_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (160, 90), (0, 0, 0))
        else:
            image = Image.new("RGB", (160, 90), (0, 0, 0))

        if self.transform is not None:
            image = self.transform(image)

        # Target speed/course tại cùng timestamp
        target = torch.zeros(2, dtype=torch.float32)
        json_path = os.path.join(self.telemetry_root, f"{video_id}.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                locs = data.get("locations", [])

                if len(locs) > 0:
                    safe_idx = min(int(t), len(locs) - 1)
                    raw_speed = float(locs[safe_idx].get("speed", 0.0))
                    raw_course = float(locs[safe_idx].get("course", 0.0))

                    # Normalize giống pipeline chính
                    target[0] = raw_speed / 30.0
                    target[1] = raw_course / 360.0
            except Exception:
                # fallback giữ target = 0
                pass

        return image, target

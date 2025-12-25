from glob import glob
from typing import Tuple, List

import nibabel as nib
import numpy as np
import cv2
from torch.utils.data import Dataset


class BraTSMidAxialBinary(Dataset):
    """
    BraTS Binary Segmentation Dataset
    - Mid axial slice
    - FLAIR modality
    - Binary label: Whole Tumor vs Background
    """

    def __init__(
        self,
        base_path: str = '../../data/testbrats/',
        out_shape: Tuple[int, int] = (128, 128),
        normalize: bool = True,
    ):
        self.out_shape = out_shape
        self.normalize = normalize

        self.case_dirs: List[str] = sorted(
            glob(f"{base_path}/BraTS-GLI-*")
        )
        print("loading data from :",base_path)
        assert len(self.case_dirs) > 0, "No BraTS cases found!"

    def __len__(self):
        return len(self.case_dirs)

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        case_id = case_dir.split("/")[-1]

        flair_path = f"{case_dir}/{case_id}-t2f.nii.gz"
        seg_path   = f"{case_dir}/{case_id}-seg.nii.gz"

        # Load volumes
        flair_vol = nib.load(flair_path).get_fdata()
        seg_vol   = nib.load(seg_path).get_fdata()

        # Mid axial slice
        z = flair_vol.shape[2] // 2
        image = flair_vol[:, :, z]
        label = seg_vol[:, :, z]

        # Binary Whole Tumor mask
        label = (label > 0).astype(np.uint8)

        # Normalize image
        if self.normalize:
            nonzero = image > 0
            if nonzero.any():
                image[nonzero] = (
                    image[nonzero] - image[nonzero].mean()
                ) / (image[nonzero].std() + 1e-8)

        # Resize
        image = cv2.resize(
            image,
            self.out_shape,
            interpolation=cv2.INTER_CUBIC
        )
        label = cv2.resize(
            label,
            self.out_shape,
            interpolation=cv2.INTER_NEAREST
        )

        # Final formatting
        image = image[None, :, :]          # (1,H,W)
        label = label[None, :, :]          # (1,H,W) for Dice/BCE

        return image.astype(np.float32), label.astype(np.uint8)

    def num_image_channel(self) -> int:
        return 1

    def num_classes(self) -> int:
        return 1

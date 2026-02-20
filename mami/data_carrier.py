# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

"""
DataCarrier: PyTorch Dataset for Paired RGB and Multispectral Images

This module provides a flexible dataset loader for working with paired RGB and
multispectral (MS) imagery from various sources. It supports multiple dataset
formats and naming conventions.

Key Features:
- Load paired RGB and 4-band multispectral images
- Automatic normalization for uint8 and uint16 images
- Optional resizing to 256x256
- Support for both directory and single-file loading
- Extensible loader system for different data sources

Typical Usage:
    from pathlib import Path
    from data_carrier import DataCarrier, load_sri_lanka

    # Load entire dataset
    dataset = DataCarrier(
        root_path=Path("data/MS_Sri_Lanka/"),
        load_data=load_sri_lanka,
        resize=True
    )

    # Load single image
    dataset = DataCarrier(
        root_path=Path("data/MS_Sri_Lanka/DJI_20230814123320_0001_D.JPG"),
        load_data=load_sri_lanka,
        resize=True
    )

    # Get a sample
    sample = dataset[0]
    rgb = sample["rgb"]   # torch.FloatTensor [3, H, W]
    ms = sample["ms"]     # torch.FloatTensor [4, H, W]
"""

import os
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Callable
from torch.utils.data import Dataset

# region Loaders
"""
Loader creation guidelines:

Each loader function should:
1. Accept a Path to a directory containing the dataset
2. Return a tuple of (rgb_paths: list[Path], ms_paths: list[Path])
3. Ensure ms_paths contains exactly 4 paths per RGB image (one per band)
4. Order MS bands consistently (e.g., [G, R, RE, NIR] for each RGB image)
5. Raise ValueError if the number of MS bands isn't divisible by 4

The ms_paths list structure:
[rgb_0_band_0, rgb_0_band_1, rgb_0_band_2, rgb_0_band_3,
 rgb_1_band_0, rgb_1_band_1, rgb_1_band_2, rgb_1_band_3, ...]
 """


def load_sri_lanka(root_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Loader for Sri Lanka dataset with DJI drone imagery.

    File naming convention:
    - RGB: DJI_<timestamp>_<id>_D.JPG
    - MS bands: DJI_<timestamp>_<id>_MS_<band>.TIF
      where <band> is one of: G (Green), R (Red), RE (Red Edge), NIR (Near Infrared)

    Args:
        root_path: Directory containing the Sri Lanka dataset

    Returns:
        tuple: (rgb_paths, ms_paths) where ms_paths contains 4 bands per RGB image

    Example:
        RGB: data/MS_Sri_Lanka/DJI_20230814123320_0001_D.JPG
        MS:  data/MS_Sri_Lanka/DJI_20230814123320_0001_MS_G.TIF
             data/MS_Sri_Lanka/DJI_20230814123320_0001_MS_R.TIF
             data/MS_Sri_Lanka/DJI_20230814123320_0001_MS_RE.TIF
             data/MS_Sri_Lanka/DJI_20230814123320_0001_MS_NIR.TIF
    """
    # Sri Lanka RGB full pictures have filenames like: <id>_D.JPG
    rgb_path_list = sorted([f for f in root_path.rglob("*_D.JPG") if f.is_file()])

    # Sri Lanka MS bands have filenames like: <id>_MS_<band>.TIF
    # Define band naming
    band_order = ["G", "R", "RE", "NIR"]
    ms_path_list: list[Path] = []
    for rgb_path in rgb_path_list:
        rgb_str = str(rgb_path)
        for suffix in band_order:
            ms_str = rgb_str.replace("_D.JPG", f"_MS_{suffix}.TIF")
            ms_path_list.append(Path(ms_str))
        if len(ms_path_list) % 4 != 0:
            raise ValueError(f"Number of MS bands is not divisible by 4. Failed at {path.name}")

    return rgb_path_list, ms_path_list

def load_east_kaz(root_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Loader for East Kazakhstan dataset.

    File naming convention:
    - RGB: <id>0.JPG
    - MS bands: <id>2.TIF, <id>3.TIF, <id>4.TIF, <id>5.TIF
      (bands 2-5 correspond to the 4 multispectral channels)

    Args:
        root_path: Directory containing the East Kazakhstan dataset

    Returns:
        tuple: (rgb_paths, ms_paths) where ms_paths contains 4 bands per RGB image

    Example:
        RGB: data/East_Kazakhstan/IMG_0001_0.JPG
        MS:  data/East_Kazakhstan/IMG_0001_2.TIF
             data/East_Kazakhstan/IMG_0001_3.TIF
             data/East_Kazakhstan/IMG_0001_4.TIF
             data/East_Kazakhstan/IMG_0001_5.TIF
    """
    # East Kazakhstan RGB pictures have filenames like: <id>0.JPG
    rgb_path_list = sorted([f for f in root_path.rglob("*.JPG") if f.is_file()])

    ms_path_list = []
    for path in rgb_path_list:
        for x in range(2, 6):
            ms_path = str(path).replace("0.JPG", f"{x}.TIF")
            ms_path_list.append(Path(ms_path))
        if len(ms_path_list) % 4 != 0:
            raise ValueError(f"Number of MS bands is not divisible by 4. Failed at {path.name}")

    return rgb_path_list, ms_path_list

def load_weedy_rice(root_path: Path) -> tuple[list[Path], list[Path]]:
    """
    Loader for Weedy Rice dataset.

    File naming convention:
    - RGB: <id>.JPG
    - MS bands: <id>_<band>.TIF
      where <band> is one of: G (Green), R (Red), RE (Red Edge), NIR (Near Infrared)

    Args:
        root_path: Directory containing the Weedy Rice dataset

    Returns:
        tuple: (rgb_paths, ms_paths) where ms_paths contains 4 bands per RGB image

    Example:
        RGB: data/Weedy_Rice/IMG_0001.JPG
        MS:  data/Weedy_Rice/IMG_0001_G.TIF
             data/Weedy_Rice/IMG_0001_R.TIF
             data/Weedy_Rice/IMG_0001_RE.TIF
             data/Weedy_Rice/IMG_0001_NIR.TIF
    """
    # Weedy Rice RGB pictures have filenames like: <id>.JPG
    rgb_path_list = sorted([f for f in root_path.rglob("*.JPG") if f.is_file()])

    band_order = ["G", "R", "RE", "NIR"]

    ms_path_list = []
    for path in rgb_path_list:
        for suffix in band_order:
            ms_path = str(path).replace(".JPG", f"_{suffix}.TIF")
            ms_path_list.append(Path(ms_path))
        if len(ms_path_list) % 4 != 0:
            raise ValueError(f"Number of MS bands is not divisible by 4. Failed at {path.name}")

    return rgb_path_list, ms_path_list
#endregion

class DataCarrier(Dataset):
    """
    PyTorch Dataset for loading paired RGB and multispectral (MS) images.

    This dataset class provides a flexible interface for working with various
    multispectral imaging datasets. It supports both directory-based loading
    (for entire datasets) and single-file loading (for individual images).

    The dataset expects each RGB image to be paired with exactly 4 multispectral
    band images. All images are automatically normalized to [0, 1] range based
    on their bit depth (uint8 or uint16).

    Args:
        root_path: Path to either:
            - A directory containing the dataset (loads all images)
            - A single RGB image file (loads only that image and its MS bands)
        load_data: Loader function that maps file paths to RGB and MS paths.
            Must return tuple[list[Path], list[Path]] where the second list
            contains 4 MS paths per RGB image.
        resize: If True, resize all images to 256x256. If False, keep original size.

    Attributes:
        rgb_paths: List of paths to RGB images
        ms_paths: List of paths to MS band images (4 per RGB image)
        resize: Whether to resize images to 256x256

    Returns:
        dict: A dictionary with keys:
            - "rgb": torch.FloatTensor of shape [3, H, W], normalized to [0, 1]
            - "ms": torch.FloatTensor of shape [4, H, W], normalized to [0, 1]

    Example:
        # Load entire dataset
        dataset = DataCarrier(
            root_path=Path("data/MS_Sri_Lanka/"),
            load_data=load_sri_lanka,
            resize=True
        )
        print(f"Dataset contains {len(dataset)} images")

        # Load single image
        single_img_dataset = DataCarrier(
            root_path=Path("data/MS_Sri_Lanka/DJI_20230814123320_0001_D.JPG"),
            load_data=load_sri_lanka,
            resize=False
        )

        # Iterate through dataset
        for sample in dataset:
            rgb = sample["rgb"]   # Shape: [3, 256, 256] or [3, H, W]
            ms = sample["ms"]     # Shape: [4, 256, 256] or [4, H, W]
            # Your processing here...

    Note:
        - RGB images are automatically converted from BGR to RGB
        - uint8 images are normalized by dividing by 255
        - uint16 images are normalized by dividing by 65535
        - MS bands with 3 channels are automatically converted to single channel
          by taking the first channel
    """

    def __init__(self,
                 root_path: Path,
                 load_data: Callable[[Path], tuple[list[Path], list[Path]]],
                 resize: bool):
        self.root_dir = root_path
        self.resize = resize

        if self.root_dir.is_file():
            # Single file mode: load all paths from parent directory,
            # then filter to keep only the specified file
            self.rgb_paths, self.ms_paths = load_data(self.root_dir.parent)

            # Find the index of the specified RGB file and extract
            # only that file and its 4 corresponding MS bands
            for i, path in enumerate(self.rgb_paths):
                if path == self.root_dir:
                    self.rgb_paths = [path]
                    self.ms_paths = self.ms_paths[i*4:(i+1)*4]
                    break
        else:
            # Directory mode: load all images in the directory
            self.rgb_paths, self.ms_paths = load_data(self.root_dir)

    def __len__(self):
        """Return the number of RGB images in the dataset."""
        return len(self.rgb_paths)

    @staticmethod
    def _load_and_normalize(path):
        """
        Load an image and normalize it to [0, 1] range.

        Args:
            path: Path to the image file

        Returns:
            np.ndarray: Normalized image as float32 in range [0, 1]

        Raises:
            FileNotFoundError: If the image cannot be read

        Note:
            - uint8 images are divided by 255
            - uint16 images are divided by 65535
            - Uses cv2.IMREAD_UNCHANGED to preserve original bit depth
        """
        path = str(path)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read file: {path}")

        # Normalize to [0, 1] based on bit depth
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        else:
            img = img.astype(np.float32) / 255.0
        return img

    def __getitem__(self, idx):
        """
        Load and return a single RGB-MS image pair.

        Args:
            idx: Index of the image to load

        Returns:
            dict: Dictionary containing:
                - "rgb": torch.FloatTensor [3, H, W], BGR converted to RGB
                - "ms": torch.FloatTensor [4, H, W], 4 multispectral bands

        Note:
            Each MS image corresponds to 4 consecutive band files in ms_paths.
            For example, idx=0 uses ms_paths[0:4], idx=1 uses ms_paths[4:8], etc.
        """
        rgb_path = str(self.rgb_paths[idx])

        # Get paths for all 4 MS bands corresponding to this RGB image
        # ms_paths is structured as [rgb0_b0, rgb0_b1, rgb0_b2, rgb0_b3,
        #                            rgb1_b0, rgb1_b1, rgb1_b2, rgb1_b3, ...]
        ms = [self.ms_paths[idx*4+x] for x in range(4)]

        # Load and optionally resize RGB image
        bands = []
        if self.resize:
            rgb = cv2.resize(self._load_and_normalize(rgb_path), (256, 256), interpolation=cv2.INTER_AREA)
            for path in ms:
                band = cv2.resize(self._load_and_normalize(path), (256, 256), interpolation=cv2.INTER_AREA)
                if band.ndim == 3:
                    band = band[:,:,0]
                bands.append(band)
        else:
            rgb = self._load_and_normalize(rgb_path)
            for path in ms:
                band = self._load_and_normalize(path)
                if band.ndim == 3:
                    band = band[:,:,0]
                bands.append(band)

        # Convert BGR to RGB (OpenCV loads as BGR by default)
        rgb = rgb[:,:,::-1].copy()

        # Stack MS bands into a single array [H, W, 4]
        target = np.stack(bands, axis=-1)

        # Convert to PyTorch tensors and change to channel-first format [C, H, W]
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
        target = torch.from_numpy(target).permute(2, 0, 1).float()

        return {"rgb": rgb, "ms": target, "path": rgb_path}

if __name__ == "__main__":
    print("Testing DataCarrier...")
    root_dir = Path("data/MS_Sri_Lanka/DJI_20230814123320_0001_D.JPG")
    dataset = DataCarrier(root_path=root_dir, load_data=load_sri_lanka, resize=True)
    sample = dataset[0]
    print("rgb patch shape:", sample["rgb"].shape)
    print("ms patch shape:", sample["ms"].shape)

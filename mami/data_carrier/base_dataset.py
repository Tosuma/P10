import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


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

    def __init__(self, root_path: Path, resize: bool):
        self.root_dir = root_path
        self.resize = resize

        if self.root_dir.is_file():
            # Single file mode: load all paths from parent directory,
            # then filter to keep only the specified file
            self.rgb_paths, self.ms_paths = self._load_data(self.root_dir.parent)

            # Find the index of the specified RGB file and extract
            # only that file and its 4 corresponding MS bands
            for i, path in enumerate(self.rgb_paths):
                if path == self.root_dir:
                    self.rgb_paths = [path]
                    self.ms_paths = self.ms_paths[i*4:(i+1)*4]
                    break
        else:
            # Directory mode: load all images in the directory
            self.rgb_paths, self.ms_paths = self._load_data(self.root_dir)

    def _load_data(self, root_path: Path) -> tuple[list[Path], list[Path]]:
        raise NotImplementedError()

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

    def __len__(self):
        """Return the number of RGB images in the dataset."""
        return len(self.rgb_paths)

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
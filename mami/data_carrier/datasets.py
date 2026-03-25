from pathlib import Path

from .base_dataset import DataCarrier

class SriLankaDataset(DataCarrier):
    """
    Data carrier for Sri Lanka dataset.

    File naming convention:
    - RGB: DJI_<timestamp>_<id>_D.JPG
    - MS bands: DJI_<timestamp>_<id>_MS_<band>.TIF
      where <band> is one of: G (Green), R (Red), RE (Red Edge), NIR (Near Infrared)

    Example:
        RGB: data/.../DJI_<yyyymmddhhmmss>_0001_D.JPG
        MS:  data/.../DJI_<yyyymmddhhmmss>_0001_MS_G.TIF
             data/.../DJI_<yyyymmddhhmmss>_0001_MS_R.TIF
             data/.../DJI_<yyyymmddhhmmss>_0001_MS_RE.TIF
             data/.../DJI_<yyyymmddhhmmss>_0001_MS_NIR.TIF
    """
    def _load_data(self, root_path: Path) -> tuple[list[Path], list[Path]]:
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
                raise ValueError(f"Number of MS bands is not divisible by 4. Failed at {root_path.name}")

        return rgb_path_list, ms_path_list
        

class KazDataset(DataCarrier):
    """
    Data carrier for East Kazakhstan dataset.

    File naming convention:
    - RGB: <id>0.JPG
    - MS bands: <id>2.TIF, <id>3.TIF, <id>4.TIF, <id>5.TIF
      (bands 2-5 correspond to the 4 multispectral channels)

    Example:
        RGB: data/East_Kazakhstan/IMG_0001_0.JPG
        MS:  data/East_Kazakhstan/IMG_0001_2.TIF
             data/East_Kazakhstan/IMG_0001_3.TIF
             data/East_Kazakhstan/IMG_0001_4.TIF
             data/East_Kazakhstan/IMG_0001_5.TIF
    """
    def _load_data(self, root_path: Path) -> tuple[list[Path], list[Path]]:
        # East Kazakhstan RGB pictures have filenames like: <id>0.JPG
        rgb_path_list = sorted([f for f in root_path.rglob("*.JPG") if f.is_file()])

        ms_path_list = []
        for root_path in rgb_path_list:
            for x in range(2, 6):
                ms_path = str(root_path).replace("0.JPG", f"{x}.TIF")
                ms_path_list.append(Path(ms_path))
            if len(ms_path_list) % 4 != 0:
                raise ValueError(f"Number of MS bands is not divisible by 4. Failed at {root_path.name}")

        return rgb_path_list, ms_path_list


class WeedyRiceDataset(DataCarrier):
    """
    Data carrier for Weedy Rice dataset.

    File naming convention:
    - RGB: <id>.JPG
    - MS bands: <id>_<band>.TIF
      where <band> is one of: G (Green), R (Red), RE (Red Edge), NIR (Near Infrared)

    Example:
        RGB: data/Weedy_Rice/IMG_0001.JPG
        MS:  data/Weedy_Rice/IMG_0001_G.TIF
             data/Weedy_Rice/IMG_0001_R.TIF
             data/Weedy_Rice/IMG_0001_RE.TIF
             data/Weedy_Rice/IMG_0001_NIR.TIF
    """
    def _load_data(self, root_path: Path):
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


class AndhraDataset(DataCarrier):
    """
    Data carrier for Andhra dataset.
    (Both fields)

    File naming convention:
    - RGB: <timestamp>_<id>_D.JPG
    - MS bands: DJI_<timestamp>_<id>_MS_<band>.TIF
      where <band> is one of: G (Green), R (Red), RE (Red Edge), NIR (Near Infrared)

    Example:
        RGB: data/.../<yyyymmddhhmmss>_0001_D.JPG
        MS:  data/.../<yyyymmddhhmmss>_0001_MS_G.TIF
             data/.../<yyyymmddhhmmss>_0001_MS_R.TIF
             data/.../<yyyymmddhhmmss>_0001_MS_RE.TIF
             data/.../<yyyymmddhhmmss>_0001_MS_NIR.TIF
    """
    def _load_data(self, root_path: Path) -> tuple[list[Path], list[Path]]:
        # Andhra RGB full pictures have filenames like: <id>_D.JPG
        rgb_path_list = sorted([
            f for f in root_path.rglob("*_D.JPG")
            # If Nursery stage should be excluded, comment the following line
            if "Nursery" in str(f)
        ])

        # Andhra MS bands have filenames like: <id>_MS_<band>.TIF
        # Define band naming
        band_order = ["G", "R", "RE", "NIR"]
        ms_path_list: list[Path] = []
        i = 0
        for rgb_path in rgb_path_list:
            rgb_str = str(rgb_path)

            parts = rgb_path.stem.split('_')
           

            image_number = parts[-2]
            if i < 10:
                print(parts)
                print(image_number)
                i += 1
            for suffix in band_order:
                search_pattern = f"*_{image_number}_MS_{suffix}.TIF"

                matching_files = list(rgb_path.parent.glob(search_pattern))

                if not matching_files:
                    print("No matching files!")
                    raise FileNotFoundError(f"Missing {suffix} MS band for {rgb_path.parent}")
                if len(matching_files) > 1:
                    print("Too many matching files")
                    raise FileNotFoundError(f"Too many files found for suffix {suffix} in {rgb_path.parent}, number of files found {len(matching_files)}")
                

                ms_path_list.append(matching_files[0])
            if len(ms_path_list) % 4 != 0:
                raise ValueError(f"Number of MS bands is not divisible by 4. Failed at {root_path.name}")

        print(len(rgb_path_list))
        print(len(ms_path_list))
        return rgb_path_list, ms_path_list


class WestBaddyDataset(DataCarrier):
    """
Data carrier for West-Baddy dataset.
(Healty and Unhealthy images)

File naming convention:
- RGB: <classification>_image<ID>.jpg
where <classification> is one of: h (healthy), u (unhealthy)

Example:
    RGB: data/.../<classification>_image<ID>.jpg

"""
    def _load_data(self, root_path: Path):
        # West-Baddy RGB full pictures have filenames like: <classification>_image<ID>.jpg
        unhealthy_path_list = [f for f in root_path.rglob("Unhealthy/*.jpg")]
        healthy_path_list = [f for f in root_path.rglob("Healthy/h_*.jpg")]

        return healthy_path_list, unhealthy_path_list

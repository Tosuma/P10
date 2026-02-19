# SPDX-License-Identifier: MIT
# Copyright (c) 2025 <Hugin J. Zachariasen, Magnus H. Jensen, Martin C. B. Nielsen, Tobias S. Madsen>.

import torch
import numpy as np
import matplotlib.pyplot as plt
from mstpp.mstpp import MST_Plus_Plus
from data_carrier import DataCarrier
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from data_carrier import load_east_kaz, load_sri_lanka, load_weedy_rice, DataCarrier
import cv2

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu" # Recommended when running full pictures to avoid OOM errors

def run(root_dir="data/",
        data_type="Sri-Lanka",
        save_dir="results",
        single=False,
        single_picture="",
        amount="Full",
        model_path="model_final.pkl",
        save_images=False):
    model = MST_Plus_Plus(in_channels=3, out_channels=4, n_feat=4, stage=3).to(device)
    output_dir= Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(model_path, map_location=device)
    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        key = k
        if key.startswith("module."):
            key = key[len("module."):]

        if key in model_sd and v.size() == model_sd[key].size():
            filtered[key] = v

    missing = set(model_sd.keys()) - set(filtered.keys())
    extra = set(state_dict.keys()) - set(filtered.keys())

    print(f"Loading checkpoint: kept {len(filtered)} params, missing {len(missing)} params, skipped {len(extra)} params")

    model.load_state_dict(filtered, strict=False)
    model.eval()


    if single:
        print("Running single image")
        # Single picture does not care for full or patch
        match data_type:
            case "Sri-Lanka":
                dataset = DataCarrier(Path(root_dir + single_picture), load_sri_lanka, resize=False)
            case "Kazakhstan":
                dataset = DataCarrier(Path(root_dir + single_picture), load_east_kaz, resize=False)
            case "Weedy-Rice":
                dataset = DataCarrier(Path(root_dir + single_picture), load_weedy_rice, resize=False)
            case _:
                print("Unknown dataset type. Defaulting to Sri-Lanka patches.")
                breakpoint() #Dummefejl
    else:
        match data_type:
            case "Sri-Lanka":
                dataset = DataCarrier(Path(root_dir), load_sri_lanka, resize=False)
            case "Kazakhstan":
                dataset = DataCarrier(Path(root_dir), load_east_kaz, resize=False)
            case "Weedy-Rice":
                dataset = DataCarrier(Path(root_dir), load_weedy_rice, resize=False)
            case _:
                print("Unknown dataset type. Defaulting to Sri-Lanka patches.")
                breakpoint() #Dummefejl

    index = 0

    dataset = DataLoader(dataset, batch_size=1, shuffle=False)
    if amount == "Full":
        limit = None
    else:
        limit = int(amount)

    for i, sample in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        rgb = sample["rgb"]
        target = sample["ms"]
        file_path = Path(sample["path"][0])
        if (i % ((len(dataset) if limit is None else limit) / 10) == 0):
            if limit is None:
                print(f"Processing [{i+1}/{len(dataset)}]")
            else: 
                print(f"Processing [{i+1}/{limit}]")


        rgb_vis = rgb.permute(0, 2, 3, 1).cpu().numpy().squeeze(0)
        target = target.squeeze(0).cpu().numpy() if target.dim() == 4 else target.cpu().numpy()
        rgb = rgb.to(device)

        with torch.no_grad():
            output = model(rgb)
            if isinstance(output, list):
                output = output[-1]
            pred = output.squeeze(0).cpu().numpy()

        pred = np.clip(pred, 0, 1)

        # Save numpy file
        np.save(output_dir / file_path.stem, pred.astype(np.float32))

        if save_images:
            file_name = f"validation_result_{str(index)}.png"
            # Save grid image
            _, axes = plt.subplots(2, 5, figsize=(14, 5))
            axes[0, 0].imshow(rgb_vis)
            axes[0, 0].set_title("RGB Input")
            axes[0, 0].axis("off")

            for i in range(4):
                axes[0, i+1].imshow(target[i], cmap='gray')
                axes[0, i+1].set_title(f"GT Band {i+1}")
                axes[0, i+1].axis("off")

            for i in range(4):
                axes[1, i].imshow(pred[i], cmap='gray')
                axes[1, i].set_title(f"Pred Band {i+1}")
                axes[1, i].axis("off")

            axes[1, 4].axis("off")
            plt.tight_layout()
            plt.savefig("validation_result.png", dpi=150, bbox_inches="tight")
            plt.savefig(output_dir / file_name, dpi=150, bbox_inches="tight")
            plt.close()
            # print(f"Saved visualization to {file_name}")

            # Save individual images
            for i in range(4):
                img = (pred[i] * 255).clip(0, 255).astype(np.uint8)
                file_name = f"validation_result_{str(index)}_{i+1}_.JPG"
                cv2.imwrite(output_dir / file_name, img)

        index += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs inference on images.")
    parser.add_argument("--data_path", help="Path to directory with data, default=data/", default="data/")
    parser.add_argument("--single", type=bool, help="One or many pictures, default=many", action=argparse.BooleanOptionalAction)
    parser.add_argument("--jpg", help="path to single picture", default=None)
    parser.add_argument("--amount", help="Amount of pictures the eval should run through, only applies if single=False, default=Full/entire dataset", default="Full")
    parser.add_argument("--save_path", help="Name of save directory", default="results")
    parser.add_argument("--data_type", type=str, choices=["Sri-Lanka", "Kazakhstan", "Weedy-Rice"], help="Which dataset should be used", required=True)
    parser.add_argument("--model", help="Which model to use, and path to the model from project dir, default=model_final.pkl", required=True)
    parser.add_argument("--save_images", help="Save the images predicted", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    root_dir = args.data_path # Root directory of data (data/)
    try:
        data_type = args.data_type #Dataset type (Sri-Lanka or Kazakhstan)
    except:
        print("No data type given, defaulting to Sri-Lanka")
    save_dir = args.save_path #Save path for results (also saves latest result in validation_result.png in main folder)
    single = args.single # One or many pictures
    single_picture = args.jpg #Only one picture
    amount = args.amount #If not single, gives the amount of pictures to process
    model_path = args.model #MST++ model to evaluate
    save_images = args.save_images
    run(
        root_dir=root_dir,
        data_type=data_type,
        save_dir=save_dir,
        single=single,
        single_picture=single_picture,
        amount=amount,
        model_path=model_path,
        save_images=save_images
        )

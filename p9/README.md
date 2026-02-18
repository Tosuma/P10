# P9

## Generating database patches for training
Generate dataset patches from sri lanka dataset images
```bash
python3 utils/generate_dataset.py --data_path data/sri_lanka/
```

Remove dataset patches (not original images)
```bash
python3 utils/remove_dataset.py --data_path data/sri_lanka/ [--dry-run]
```

## Align dataset
As an example with the Sri Lanka dataset in the path `data/sri_lanka` we can 
use the tool to align them. By default they will align by the green ms image.
To use, run the command:
```bash
python utils/align_images.py --input_folder=/path/to/dataset [--output_folder=/path/to/save/aligned_images]
```
By default the aligned images will be saved to data/aligned_images if nothing else is specified.

## Error maps
Script to generate the error maps given predicionts images and ground-truth images.
```bash
python3 utils/error_maps.py --pred_path=<path/to/pred.SUFFIX> --gt_path=<path/to/groud_truth.SUFFIX> [--error_map_type=<all,ndvi,ndre>]
```

- `--pred_path` is the path to the prediction files. If path is `data/pred.jpg` it assumes all the prediction files are .jpg and follows the schema `_g.jpg`, `_r.jpg`, `_re.jpg`, `_nir.jpg`
- `--gt_path` follows the same logic as `--pred_path`.


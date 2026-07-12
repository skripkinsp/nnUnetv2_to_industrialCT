# 3D Industrial CT Segmentation

Adaptation of `nnU-Net` for industrial CT via Few-Shot Transfer Learning.

## Core Modules
- **`new_trainer/`**: Custom training logic with GDL + Top-K CE loss to handle extreme class imbalance.
- **`postcrop/`**: Topological cleaning pipeline (LCC, Visual Hull, Erosion) to remove artifacts.
- **`rotation_file/`**: $SO(3)$ rotation benchmark using the Haar measure.
- **`visualization/`**: Robustness analysis and metric plotting.
- **`create3D_module/`**: Creates 3D model from slices

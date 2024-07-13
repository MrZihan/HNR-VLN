### Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation

#### Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, Junjie Hu, Ming Jiang and Shuqiang Jiang


>Vision-and-language navigation (VLN) enables the agent to navigate to a remote location following the natural language instruction in 3D environments. At each navigation step, the agent selects from possible candidate locations and then makes the move. For better navigation planning, the lookahead exploration strategy aims to effectively evaluate the agent's next action by accurately anticipating the future environment of candidate locations.
To this end, some existing works predict RGB images for future environments, while this strategy suffers from image distortion and high computational cost. To address these issues, we propose the pre-trained hierarchical neural radiance representation model (HNR) to produce multi-level semantic features for future environments, which are more robust and efficient than pixel-wise RGB reconstruction. Furthermore, with the predicted future environmental representations, our lookahead VLN model is able to construct the navigable future path tree and select the optimal path branch via efficient parallel evaluation. Extensive experiments on the VLN-CE datasets confirm the effectiveness of our method.

![image](https://github.com/MrZihan/HNR-VLN/blob/main/demo.gif)

## TODOs

* [X] Release the pre-training code of the Hierarchical Neural Radiance Representation Model.
* [X] Release the checkpoints of the Hierarchical Neural Radiance Representation Model.
* [X] Tidy the pre-training code for easy execution.
* [ ] Release the fine-tuning code of the Lookahead VLN Model.
* [ ] Release the checkpoints of the Lookahead VLN Model.

## Issues
For training speed, see [Issue#7](https://github.com/MrZihan/HNR-VLN/issues/7)

Load only a few scenes for efficient debugging, see [Issue#4](https://github.com/MrZihan/HNR-VLN/issues/4)

## Requirements

1. Install `Habitat simulator`: follow instructions from [ETPNav](https://github.com/MarSaKi/ETPNav) and [VLN-CE](https://github.com/jacobkrantz/VLN-CE).
2. Download the `Habitat-Matterport 3D Research Dataset (HM3D)` from [habitat-matterport-3dresearch](https://github.com/matterport/habitat-matterport-3dresearch)
   ```
   hm3d-train-habitat-v0.2.tar
   hm3d-val-habitat-v0.2.tar
   ```
4. Download annotations (PointNav, VLN-CE) and trained models from [Baidu Netdisk](https://pan.baidu.com/s/1Q511rG-_mJZxufGm4UbAWw?pwd=ih5n) or [TeraBox](https://terabox.com/s/1nSATHQT1b1Cl8HqgEAIHrQ).
5. Download pre-trained `waypoint predictor` from [link](https://drive.google.com/file/d/1goXbgLP2om9LsEQZ5XvB0UpGK4A5SGJC/view?usp=sharing).
6. Install `torch_kdtree` for K-nearest feature search from [torch_kdtree](https://github.com/thomgrand/torch_kdtree).
   ```
   git clone https://github.com/thomgrand/torch_kdtree
   cd torch_kdtree
   git submodule init
   git submodule update
   pip3 install .
   ```
7. Install `tinycudann` for faster multi-layer perceptrons (MLPs) from [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).
   ```
   pip3 install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
   ```

## Pre-train the HNR model

   ```
   bash run_r2r/nerf.bash train 2345
   ```

## Evaluate the HNR model
   Evaluate the cosine similarity between the HNR model's predicted features and the CLIP model's ground truth features.
   ```
   bash run_r2r/nerf.bash eval 2345
   ```
   Set `Visualization` to `True` in line 68 of `HNR-VLN/NeRF/ss_trainer_ETP.py`, visualize and save the images predicted by the HNR model.
   

## Citation

```bibtex
@InProceedings{Wang_2024_CVPR,
    author    = {Wang, Zihan and Li, Xiangyang and Yang, Jiahao and Liu, Yeqi and Hu, Junjie and Jiang, Ming and Jiang, Shuqiang},
    title     = {Lookahead Exploration with Neural Radiance Representation for Continuous Vision-Language Navigation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {13753-13762}
}
  ```

## Acknowledgments
Our code is based on [ETPNav](https://github.com/MarSaKi/ETPNav), [nerf-pytorch](https://github.com/yenchenlin/nerf-pytorch) and [torch_kdtree](https://github.com/thomgrand/torch_kdtree). Thanks for their great works!

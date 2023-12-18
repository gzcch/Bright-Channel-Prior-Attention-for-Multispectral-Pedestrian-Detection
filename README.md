# README

## Project Overview
This repository hosts the code for the research paper "Bright Channel Prior Attention for Multispectral Pedestrian Detection". The paper presents a novel method for enhancing pedestrian detection in low-light conditions by integrating image enhancement and detection within a unified framework. The method leverages the V-channel of the HSV image of the thermal image as an attention map to trigger the unsupervised auto-encoder for visible light images, emphasizing pedestrian features across layers.

## Research Paper
The detailed methodology, experiments, and results are discussed in our paper, which can be accessed [here](https://arxiv.org/abs/2305.12845).

## Key Contributions
- **Bright Channel Prior Algorithm**: Utilizes the V-channel of the HSV image of the thermal image as an attention map, enhancing the focus on pedestrian features.
- **Unsupervised Cross-Modal Illumination Map Estimation**: Employs unsupervised bright channel prior algorithms for light compensation in low-light images.
- **YOLO-based Detection Module**: Integrates the enhancement network with YOLO-v4 for improved object detection in multispectral images.

## Experiments and Results
- **Dataset Used**: LLVIP dataset for low-light visible image person detection.
- **Performance Metrics**: The method's effectiveness is demonstrated through accuracy, recall, and mAP metrics.
- **Comparative Analysis**: Compared with Yolo-v4 and its variants, showing significant improvements.

## Citation
If you find this work useful in your research, please cite the following paper:
```
@article{cui2023bright,
  title={Bright Channel Prior Attention for Multispectral Pedestrian Detection},
  author={Cui, Chenhang and Xie, Jinyu and Yang, Yechenhao},
  journal={arXiv preprint arXiv:2305.12845},
  year={2023}
}
```

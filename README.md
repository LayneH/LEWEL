# Learning Where to Learn in Cross-View Self-Supervised Learning

<p align="center">
    <img src="figs/LEWEL.png" width="850"\>
</p>


This is the official PyTorch implementation of the CVPR'2022 '[Learning Where to Learn in Cross-View Self-Supervised Learning](https://arxiv.org/abs/2203.14898)'.
>
> *ABSTRACT* - Self-supervised learning (SSL) has made enormous progress and largely narrowed the gap with the supervised ones, where the representation learning is mainly guided by a projection into an embedding space. During the projection, current methods simply adopt uniform aggregation of pixels for embedding; however, this risks involving object-irrelevant nuisances and spatial misalignment for different augmentations. In this paper, we present a new approach, Learning Where to Learn (LEWEL), to adaptively aggregate spatial information of features, so that the projected embeddings could be exactly aligned and thus guide the feature learning better. Concretely, we reinterpret the projection head in SSL as a per-pixel projection and predict a set of spatial alignment maps from the original features by this weight-sharing projection head. A spectrum of aligned embeddings is thus obtained by aggregating the features with spatial weighting according to these alignment maps. As a result of this adaptive alignment, we observe substantial improvements on both image-level prediction and dense prediction at the same time: LEWEL improves MoCov2 by 1.6%/1.3%/0.5%/0.4% points, improves BYOL by 1.3%/1.3%/0.7%/0.6% points, on ImageNet linear/semi-supervised classification, Pascal VOC semantic segmentation, and object detection, respectively.
>

## Requirements

Please install the package listed in the `requirements.txt`. By default, we run the experiments on machines with eight Nvidia V100 GPUs, CUDA10.1, and PyTorch 1.8.1 (PyTorch >= 1.6 should work fine).

## Pretraning

We provide several scripts in `scripts/`, which can reproduce our main results. Note that the scripts with name begins with `srun*` are for slurm user. If you are not using slurm, you can easily modified the scripts according to `scripts/run_lewelb_eman.sh`.

### Models

We provide the ImageNet-1K pre-trained ResNet-50 models in the following table:

| Name | Epochs | Top1 ImageNet Val Acc | Model |
| :---: | :---: | :---: | :---: |
| LEWEL_B (EMAN) | 100 | 71.9 | [script](scripts/srun_lewelb_eman.sh) / [model](https://drive.google.com/file/d/12_u_6mR7Fg6YU4lxovL2lBXxXwmQehuK/view?usp=sharing) |
| LEWEL_B (EMAN) | 200 | 72.8 | [script](scripts/srun_lewelb_eman.sh) / [model](https://drive.google.com/file/d/1ofZjBOAS3IB82Hz-Rt8lW4lqd4nR60ZR/view?usp=sharing) |
| LEWEL_B (SyncBN) | 400 | 73.8 | [script](scripts/srun_lewelb.sh) / [model](https://drive.google.com/file/d/13PVJexOZqy3XNEBbmeaVCXdrl9JhAiCN/view?usp=sharing) |

## Evaluation

### Linear Evaluation

See `scripts/srun_lin.sh`.

### Semi-supervised 

- `scripts/srun_semi_1p.sh`: script of finetuning pretrained models using 1% of ImageNet data;

- `scripts/srun_semi_10p.sh`: script of finetuning pretrained models using 10% of ImageNet data;

# MS-COCO Object Detection and Instance Segmentation

We follow the settings of [MoCo](https://github.com/facebookresearch/moco/tree/main/detection) and [InfoMin](https://github.com/HobbitLong/PyContrast/tree/master/pycontrast/detection). Please refer to these two links for installation and running instructions.

| Name | Pretrain Epochs | Detector | Schedule | Config | Box AP | Mask AP |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LEWEL_B (EMAN) | 200 | Mask R-CNN C4 | 1x | [config](det_configs/coco_R_50_C4_1x_infomin.yaml) | 38.5 | 33.7 |
| LEWEL_B (EMAN) | 200 | Mask R-CNN FPN | 1x | [config](det_configs/coco_R_50_FPN_1x_infomin.yaml) | 41.3 | 37.4 |
| LEWEL_B (EMAN) | 200 | Mask R-CNN FPN | 2x | [config](det_configs/coco_R_50_FPN_2x_infomin.yaml) | 42.2 | 38.2 |
| LEWEL_B (SyncBN) | 400 | Mask R-CNN FPN | 1x | [config](det_configs/coco_R_50_FPN_1x_infomin.yaml) | 41.9 | 37.9 |
| LEWEL_B (SyncBN) | 400 | Mask R-CNN FPN | 2x | [config](det_configs/coco_R_50_FPN_2x_infomin.yaml) | 43.4 | 39.1 |

NOTE: we find that using a larger finetuning learning rate (e.g., 2e-4 in our experiments) produces better results for models pretrained with EMAN, while the models with SyncBN work best with the default learning rate (i.e., 1e-4).

## Citation

If you find our work interesting or use our code/models, please cite:

```bibtex
@inproceedings{huang2022learning,
  title={Learning Where to Learn in Cross-View Self-Supervised Learning},
  author={Huang, Lang and You, Shan and Zheng, Mingkai and Wang, Fei and Qian, Chen and Yamasaki, Toshihiko},
  booktitle={CVPR},
  year={2022}
}
```

## License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

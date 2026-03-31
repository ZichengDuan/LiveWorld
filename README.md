<div align="center">

<h1>🌏 LiveWorld: Simulating Out-of-Sight Dynamics in Generative Video World Models</h1>

<div>
    <a href='https://zichengduan.github.io' target='_blank'>Zicheng Duan<sup>1*</sup></a>&emsp;
    <a href='https://jiatongxia.github.io' target='_blank'>Jiatong Xia<sup>1*</sup></a>&emsp;
    <a href='https://steve-zeyu-zhang.github.io' target='_blank'>Zeyu Zhang<sup>2*</sup></a>&emsp;
    <a href='https://zwbx.github.io' target='_blank'>Wenbo Zhang<sup>1</sup></a>&emsp;
    <a href='https://gengzezhou.github.io' target='_blank'>Gengze Zhou<sup>1</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=tlhShPsAAAAJ' target='_blank'>Chenhui Gou<sup>3</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=CTEQwwwAAAAJ' target='_blank'>Yefei He<sup>4</sup></a>&emsp;
    <a href='https://scholar.google.com/citations?user=fwpY_HoAAAAJ' target='_blank'>Feng Chen<sup>1†</sup></a>&emsp;
    <a href='https://zhangxinyu-xyz.github.io' target='_blank'>Xinyu Zhang<sup>5‡</sup></a>&emsp;
    <a href='https://researchers.adelaide.edu.au/profile/lingqiao.liu' target='_blank'>Lingqiao Liu<sup>1†‡</sup></a>
</div>

<sup>1</sup>Adelaide University&emsp;
<sup>2</sup>The Australian National University&emsp;
<sup>3</sup>Monash University&emsp;
<sup>4</sup>Zhejiang University&emsp;
<sup>5</sup>University of Auckland

<sub>* Equal contribution&emsp;† Project lead&emsp;‡ Corresponding author</sub>

<br>

<div>
    <a href='https://zichengduan.github.io/LiveWorld' target='_blank'><img src="https://img.shields.io/badge/Project-Page-blue"></a>
    <a href='https://arxiv.org/abs/2603.07145' target='_blank'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/ZichengD/LiveWorld' target='_blank'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
</div>

</div>

## 📋 TODOs

- [x] Release inference code and pretrained weights
- [x] Release training code
- [x] Release data preparation pipeline
- [ ] Release LiveBench benchmark and evaluation scripts
- [ ] Release demo data and examples
- [ ] Add detailed documentation

## 🔧 Installation

**1. Clone the repository**
```bash
git clone https://github.com/ZichengDuan/LiveWorld.git
cd LiveWorld
```

**2. Install PyTorch**
```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

**3. Setup environment (install dependencies + download model weights)**
```bash
bash setup_env.sh
```

This script will:
- Install Python dependencies from `setup/requirements.txt`
- Install LiveWorld and local packages (SAM3, Stream3R)
- Download all pretrained weights (~100GB) into `ckpts/`

<details>
<summary>📦 Downloaded model weights</summary>

| Model | Source | Purpose |
|---|---|---|
| LiveWorld State Adapter + LoRA | [ZichengD/LiveWorld](https://huggingface.co/ZichengD/LiveWorld) | Core LiveWorld weights |
| Wan2.1-T2V-14B | [Wan-AI/Wan2.1-T2V-14B](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B) | Backbone |
| Wan2.1-Fun-1.3B-InP | [alibaba-pai/Wan2.1-Fun-1.3B-InP](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) | VAE (data preparation) |
| Wan2.1-T2V-14B-StepDistill | [lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill) | Distilled backbone (fast inference) |
| Qwen3-VL-8B-Instruct | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Entity detection |
| SAM3 | [facebook/sam3](https://huggingface.co/facebook/sam3) | Video segmentation |
| STream3R | [yslan/STream3R](https://huggingface.co/yslan/STream3R) | 3D reconstruction |
| DINOv3 (optional) | [facebook/dinov3-vith16plus-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m) | Entity matching |

</details>

## 🚀 Inference

Edit `infer.sh` to set your config path and GPU, then run:
```bash
bash infer.sh
```

## 🏋️ Training

Edit `train.sh` to set your GPU configuration, then run:
```bash
bash train.sh
```

Training config: `configs/train_liveworld_14B.yaml`

## 📊 Data Preparation

Edit `configs/data_preparation.yaml` to set input video paths and output directory, then run:
```bash
bash dataset_preparation.sh
```

The pipeline runs 4 steps automatically:
1. **Build samples** — clip extraction, entity detection (Qwen3-VL), segmentation (SAM3), geometry estimation (Stream3R), sample construction
2. **Captioning** — generate text descriptions with Qwen3-VL
3. **VAE encode** — encode videos to latent space
4. **Pack LMDB** — package into sharded LMDB for training

## 📁 Project Structure

```
LiveWorld/
├── infer.sh                    # Inference entry point
├── train.sh                    # Training entry point
├── dataset_preparation.sh      # Data preparation entry point
├── setup_env.sh                # One-click environment setup
├── setup/                      # Installation scripts & requirements
├── configs/
│   ├── infer_system_config.yaml
│   ├── train_liveworld_14B.yaml
│   └── data_preparation.yaml
├── liveworld/                  # Core package
│   ├── trainer.py              # Task definition + training loop
│   ├── wrapper.py              # Model wrappers (VAE, text encoder, State Adapter)
│   ├── dataset.py              # LMDB dataset loader
│   ├── utils.py                # Utilities
│   ├── geometry_utils.py       # Geometry & projection utilities
│   ├── pipelines/
│   │   ├── pipeline_unified_backbone.py   # Unified Backbone
│   │   ├── pointcloud_updater.py          # Stream3R point cloud handler
│   │   └── monitor_centric/               # Monitor-Centric Evolution Pipeline
│   └── wan/                    # Wan2.1 model architecture
├── scripts/
│   ├── infer.py
│   ├── train.py
│   └── dataset_preparation/    # Data processing steps
├── misc/
│   ├── sam3/                   # SAM3 (local package)
│   └── STream3R/               # Stream3R (local package)
├── ckpts/                      # Model weights (download separately)
└── examples/                   # Example data
```

## 🙏 Acknowledgements

We thank the authors of [Wan2.1](https://github.com/Wan-Video/Wan2.1), [SAM3](https://github.com/facebookresearch/sam3), [STream3R](https://github.com/NIRVANALAN/STream3R), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), and [DINOv3](https://github.com/facebookresearch/dinov2) for their outstanding open-source contributions.

We also acknowledge that the concept of reasoning about out-of-sight world shares a similar spirit with Out of Sight, Not Out of Mind (Plizzari et al., 3DV 2025) [[paper](https://arxiv.org/abs/2404.05072), [code](https://github.com/Chiaraplizz/OSNOM)], which explored spatial cognition of off-screen objects in egocentric video perception — a totally different domain but a kindred high-level insight.

## 📝 Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{duan2026liveworld,
  title={LiveWorld: Simulating Out-of-Sight Dynamics in Generative Video World Models},
  author={Duan, Zicheng and Xia, Jiatong and Zhang, Zeyu and Zhang, Wenbo and Zhou, Gengze and Gou, Chenhui and He, Yefei and Chen, Feng and Zhang, Xinyu and Liu, Lingqiao},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

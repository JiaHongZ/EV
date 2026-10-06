# EV

Supplementary code for **Embodied Vision for Closed-loop Perception-Decision Making in Intelligent Agents**.

The associated manuscript is currently under revision at *Nature Communications*.

We provide the core experimental code of EV for retina visualization, static image recognition on ImageNet, and embodied navigation. These implementations are provided for review and reference.

## Repository layout

The ImageNet implementation, training scripts, evaluation scripts, and supporting code are in [`EV-ImageNet/`](EV-ImageNet/). The navigation code is undergoing reproduction checks before release. Code for additional experiments, including the retina visualization example and its `environment.yml`, is being organized for release.

## Dependencies

This project is built on [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models). The required source snapshot is included under `EV-ImageNet/timm`.

## 1. Retina Visualization

The following instructions apply once `environment.yml` and `example/` are released.

### Environment Setup

```bash
conda env create -f environment.yml
cd example
```

Modify the following parameters in `show_retina.py`:

- `l_t`: fixation location, in the range [-1, 1].
- `s_t`: focal modulation parameter.

Run the visualization:

```bash
python show_retina.py
```

## 2. ImageNet Recognition

The shared environment setup follows Section 1 once `environment.yml` is released. The ImageNet dependency list is in [`EV-ImageNet/requirements.txt`](EV-ImageNet/requirements.txt).

Prepare the ImageNet-1K dataset and replace `/zjh/data/imagenet1-k` in the commands below with its local path.

### Pretrained Weights

Download `ev-resnet18-em4-224.tar` from [the pretrained model archive](https://pan.baidu.com/s/16bm45WwHXMvL9zOWXXcxfQ?pwd=msuu). Extraction code: `msuu`.

Place it in `EV-ImageNet/weight/`, then run all ImageNet commands from the project directory:

```bash
cd EV-ImageNet
```

### 2.1 Testing ResNet-18

```bash
CUDA_VISIBLE_DEVICES=0 python test_ev.py \
    --data-dir /zjh/data/imagenet1-k \
    -b 64 --epochs 1 --num-classes 1000 --img-size 224 \
    --model-kwargs patch_number=4 retina_size=224 use_residual=True \
    --model evc_resnet18 \
    --initial-checkpoint ./weight/ev-resnet18-em4-224.tar
```

### 2.2 Occlusion Results and Visualization

```bash
CUDA_VISIBLE_DEVICES=0 python test_ev_occ.py \
    --data-dir /zjh/data/imagenet1-k \
    --num-classes 1000 --img-size 224 \
    --model-kwargs patch_number=4 retina_size=224 \
    --model evc_resnet18 \
    --lr 1e-5 --opt adamw --opt-eps 1e-8 --weight-decay 0.05 \
    --warmup-epochs 2 --warmup-lr 1e-6 --min-lr 1e-7 \
    --epochs 1 --sched cosine -b 16 -j 1 --amp --dist-bn reduce \
    --initial-checkpoint ./weight/ev-resnet18-em4-224.tar
```

### 2.3 Training

The supplied training example uses two eye movements (`patch_number=2`); the evaluation checkpoint above uses four.

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 --master_port 29516 train_ev.py \
    --data-dir /zjh/data/imagenet1-k \
    --num-classes 1000 --img-size 224 \
    --model-kwargs patch_number=2 retina_size=224 use_residual=True \
    --model evc_resnet18 \
    --lr 1e-3 --opt adamw --opt-eps 1e-8 --weight-decay 0.0001 \
    --warmup-epochs 0 --warmup-lr 1e-6 --min-lr 1e-7 \
    --epochs 100 --sched cosine --aa rand-m9-mstd0.5-inc1 \
    -b 16 -j 8 --amp --dist-bn reduce --model-ema True \
    --mixup 0.2 --cutmix 1.0 --drop-path 0.2 --smoothing 0.1 \
    --save-eyemove-figs --eyemove-save-interval 10
```

## 3. Navigation

The navigation implementation will be released under `EV-Nav/` after reproduction checks are complete.

### Pretrained Weights

Download `XGX-EV.pth` and `hm3d_rednet.pt` from [the pretrained model archive](https://pan.baidu.com/s/16bm45WwHXMvL9zOWXXcxfQ?pwd=msuu). Extraction code: `msuu`.

Place both files in `EV-Nav/models/`. Detailed environment, dataset, training, and evaluation instructions will be provided in `EV-Nav/README.md` when that directory is released.

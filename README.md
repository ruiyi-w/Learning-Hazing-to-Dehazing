# Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing [:link:](https://arxiv.org/abs/2503.19262)

![Python 3.10](https://img.shields.io/badge/python-3.10-g) ![pytorch 2.2.2](https://img.shields.io/badge/pytorch-2.2.2-blue.svg)

This repository presents the implementation of the paper

>**Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing**<br> Ruiyi Wang, Yushuo Zheng, Zicheng Zhang, Chunyi Li, Shuaicheng Liu, Guangtao Zhai, Xiaohong Liu<br>The IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR), 2025

We present a novel hazing-dehazing pipeline consisting of a Realistic Hazy Image Generation framework (HazeGen) and a Diffusion-based Dehazing framework (DiffDehaze).

![teaser](assets/teaser.png)
![teaser](assets/result.png)

## 🛠️ Setup

### 📦 Repository

Clone the repository (requires git):

```bash
git clone https://github.com/ruiyi-w/Learning-Hazing-to-Dehazing.git
cd Learning-Hazing-to-Dehazing
```

### 💻 Dependencies
Using [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). After the installation, create the environment and install dependencies into it:

```bash
conda create -n env_name python=3.10
conda activate env_name
pip install -r requirements.txt
```

Keep the environment activated before running the inference script. 
Activate the environment again after restarting the terminal session.

## 🏃 Testing

### 📷 Prepare input images

Place your images in the `inputs/` directory. To set a different source directory, you can edit configuration files in `configs/inference/`.

### ⬇ Download Checkpoints

Download pre-trained models and place them to folder `weights/`, but you can always edit configuration files in `configs/inference/`.

|        Model Name        |                         Description                          |                             Link                             |
| :----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| v2-1_512-ema-pruned.ckpt | Pretrained Stable Diffusion v2.1 from stabilityai, providing generative priors | [download](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) |
|        stage1.pt         |               IRControlNet trained for HazeGen               | [download](https://pan.baidu.com/s/1vbxEwftJC9nUaMXJ-t9sww?pwd=8egg) |
|        stage2.pt         |             IRControlNet trained for DiffDehaze              | [download](https://pan.baidu.com/s/1vbxEwftJC9nUaMXJ-t9sww?pwd=8egg) |

### 🚀 Run inference

To perform dehazing with standard spaced sampler, please run

```bash
python inference_stage2.py --config configs/inference/stage2.yaml
```

By default, results will be saved to `outputs/`. **Enjoy**!

To perform dehazing with AccSamp sampler, please run

```bash
python inference_accsamp.py --config configs/inference/stage2_accsamp.yaml
```

To generate realisitic hazy images with HazeGen, please run

```bash
python inference_stage1.py --config configs/inference/stage1.yaml
```

To use a different hyperparameter settings, e.g., $\tau$ and $\omega$, please edit the corresponding `.yaml` configuration file.

## 🏋️ Training

1. Training data preparation. 
   - The training of HazeGen requires real-world hazy data from the URHI split of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-β) dataset and the synthetic hazy data from [RIDCP](https://github.com/RQ-Wu/RIDCP_dehazing). 
   - To train DiffDehaze, you need to generate realistic hazy data from HazeGen based on clean images, e.g., the clean images from the OTS split of [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-β) dataset, using the inference script above.

2. Accelerate configuration. The training is supported by the huggingface [Accelerate](https://huggingface.co/docs/transformers/accelerate) library. Before running training scripts, create and save a configuration file to help Accelerate correctly set up training based on your setup by running 

```bash
accelerate config
```

3. Fill in the training configuration files in `configs/train/` with appropriate values, especially for the paths to the training data. Please find specific instructions there.
4. Start training! To train stage1 HazeGen model, run

```bash
accelerate launch train_stage1.py --config configs/train/stage1.yaml
```

5. To train stage2 DiffDehaze model, run

```bash
accelerate launch train_stage2.py --config configs/train/stage2.yaml
```

## ✏️ Acknowledgment

A large part of the implementation is based on [DiffBIR](https://github.com/XPixelGroup/DiffBIR?tab=readme-ov-file#inference). We sincerely appreciate their wonderful work.

## 🎓 Citation

If you find our work useful, please consider cite our paper:

```bibtex
@misc{wang2025learninghazingdehazingrealistic,
      title={Learning Hazing to Dehazing: Towards Realistic Haze Generation for Real-World Image Dehazing}, 
      author={Ruiyi Wang and Yushuo Zheng and Zicheng Zhang and Chunyi Li and Shuaicheng Liu and Guangtao Zhai and Xiaohong Liu},
      year={2025},
      eprint={2503.19262},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.19262}, 
}
```

## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and models you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)

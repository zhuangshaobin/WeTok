<div align="center">
<h1>🚀 WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction</h1>

</div>

This project introduces **WeTok**, a powerful discrete visual tokenizer designed to resolve the long-standing conflict between compression efficiency and reconstruction fidelity. WeTok achieves state-of-the-art reconstruction quality, surpassing previous leading discrete and continuous tokenizers. <br><br>

> <a href="https://github.com/zhuangshaobin/WeTok">WETOK: POWERFUL DISCRETE TOKENIZATION FOR HIGH-FIDELITY VISUAL RECONSTRUCTION</a><br>
> [Shaobin Zhuang], [Yiwei Guo], [Canmiao Fu], [Zhipeng Huang], [Zeyue Tian], [Ying Zhang], [Chen Li], [Yali Wang]<br>
> Shanghai Jiao Tong University, WeChat Vision (Tencent Inc.), Shenzhen Institutes of Advanced Technology (Chinese Academy of Sciences), Hong Kong University of Science and Technology, Shanghai AI Laboratory<br>
> <a href="./docs/WeTok.md">📚WeTok.md</a>
> ```
> @inproceedings{zhuang2026wetok,
>   title={WETOK: POWERFUL DISCRETE TOKENIZATION FOR HIGH-FIDELITY VISUAL RECONSTRUCTION},
>   author={Zhuang, Shaobin and Guo, Yiwei and Fu, Canmiao and Huang, Zhipeng and Tian, Zeyue and Zhang, Ying and Li, Chen and Wang, Yali},
>   booktitle={International Conference on Learning Representations},
>   year={2026}
> }
> ```

<p align="center">
  <img src="https://i.imgur.com/K3p0pYp.png" width="90%">
  <br>
  <em>WeTok achieves a new state-of-the-art in reconstruction fidelity, surpassing both discrete and continuous tokenizers, while offering high compression ratios.</em>
</p>

## 📰 News
* **[2026.01.28]**:fire::fire::fire: **WeTok is accepted by ICLR 2026.**
* **[2025.12.15]** We release the **WeTok-AR-XL** model, achieving SOTA performance (**2.31 FID**) on ImageNet 256x256 class-conditional generation.
* **[2025.11.20]** We release WeTok tokenizers trained on a 400M general-domain dataset, achieving a record-low zero-shot rFID of **0.12** on ImageNet, surpassing top continuous tokenizers like FLUX-VAE and SD-VAE 3.5.
* **[2025.10.30]** We are excited to release **WeTok**, a powerful discrete tokenizer featuring our novel **Grouped Lookup-Free Quantization (GFQ)** and a **generative decoder**. Code and pretrained models are now available!

## 📖 Implementations

### 🛠️ Installation
- **Env**: We have tested on `Python 3.9+`, `PyTorch 2.0+` and `CUDA 11.8+` (other versions may also be fine).
- **Dependencies**: `pip install -r requirements.txt`

### Datasets

- **Image Dataset (e.g., ImageNet)**

We use ImageNet-1K for in-distribution training and evaluation. The dataset should be organized as follows:
```
imagenet
└── train/
    ├── n01440764
        ├── n01440764_10026.JPEG
        ├── ...
    ├── n01443537
    └── ...
└── val/
    ├── ...
```

- **General-Domain Dataset**

For large-scale pre-training, we recommend organizing the data in the WebDataset `tar` format for efficient loading.
```
general_domain_data
└── dataset_1/
    ├── webdataset
        ├── 00000.tar
        ├── 00001.tar
        └── ...
└── dataset_2/
    ├── webdataset
        ├── 00000.tar
        ├── 00001.tar
        └── ...
```

### ⚡ Training & Evaluation
The detailed scripts for training and evaluation can be found in <a href="docs/WeTok.md">WeTok.md</a>.

## ❤️ Acknowledgement
Our work builds upon the foundations laid by many excellent projects in the field. We would like to thank the authors of [VQGAN](https://github.com/CompVis/taming-transformers), [MAGVIT-v2](https://github.com/google-research/magvit), [Open-MAGVIT2](https://github.com/MCG-NJU/Open-MAGVIT2), [LlamaGen](https://github.com/Vchitect/LlamaGen), and [VideoGPT](https://github.com/wilson1yan/VideoGPT). We also drew inspiration from the methodologies presented in [LFQ](https://arxiv.org/abs/2310.12797), [BSQ](https://arxiv.org/abs/2402.13039), and recent state-of-the-art tokenizers like [FLUX-VAE](https://github.com/black-forest-labs/flux) and [SD-VAE](https://github.com/stability-ai/sd-vae-3-5). We are grateful for their contributions to the community.
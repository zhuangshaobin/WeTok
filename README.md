<div align="center">
<h1>🚀 WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2505.12489-b31b1b.svg)](https://arxiv.org/abs/2505.12489)
[![Github](https://img.shields.io/badge/Github-WeTok-blue)](https://github.com/zhuangshaobin/WeTok)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/GrayShine/WeTok)

</div>

This project introduces **WeTok**, a powerful discrete visual tokenizer designed to resolve the long-standing conflict between compression efficiency and reconstruction fidelity. WeTok achieves state-of-the-art reconstruction quality, surpassing previous leading discrete and continuous tokenizers. <br><br>

> <a href="https://github.com/zhuangshaobin/WeTok">WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction</a><br>
> [Shaobin Zhuang](https://scholar.google.com/citations?user=PGaDirMAAAAJ&hl=zh-CN&oi=ao), [Yiwei Guo](https://scholar.google.com/citations?user=HCAyeJIAAAAJ&hl=zh-CN&oi=ao), [Canmiao Fu](), [Zhipeng Huang](), [Zeyue Tian](https://scholar.google.com/citations?user=dghq4MQAAAAJ&hl=zh-CN&oi=ao), [Ying Zhang](https://scholar.google.com/citations?user=R_psgxkAAAAJ&hl=zh-CN&oi=ao), [Chen Li](https://scholar.google.com/citations?hl=zh-CN&user=WDJL3gYAAAAJ), [Yali Wang](https://scholar.google.com/citations?hl=zh-CN&user=hD948dkAAAAJ)<br>
> Shanghai Jiao Tong University, WeChat Vision (Tencent Inc.), Shenzhen Institutes of Advanced Technology (Chinese Academy of Sciences), Hong Kong University of Science and Technology, Shanghai AI Laboratory<br>
> ```
> @article{zhuang2026wetok,
>   title={WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction},
>   author={Zhuang, Shaobin and Guo, Yiwei and Fu, Canmiao and Huang, Zhipeng and Tian, Zeyue and Zhang, Ying and Li, Chen and Wang, Yali},
>   journal={arXiv preprint arXiv:2409.04410},
>   year={2025}
> }
> ```

<p align="center">
  <img src="./assets/teaser.png" width="90%">
  <br>
  <em>WeTok achieves a new state-of-the-art in reconstruction fidelity, surpassing both discrete and continuous tokenizers, while offering high compression ratios.</em>
</p>

## 📰 News
* **[2025.08.05]**:fire::fire::fire: We release a series of WeTok models, achieving a record-low zero-shot rFID of **0.12** on ImageNet, surpassing top continuous tokenizers like FLUX-VAE and SD-VAE 3.5.
* **[2025.08.05]** We are excited to release **WeTok**, a powerful discrete tokenizer featuring our novel **Grouped Lookup-Free Quantization (GFQ)** and a **generative decoder**. Code and pretrained models are now available!

## 📖 Implementations

### 🛠️ Installation
- **Dependencies**: 
```
bash env.sh
```

### Evaluation

- **Evaluation on ImageNet 50K Validation Set**

The dataset should be organized as follows:
```
imagenet
└── val/
    ├── ...
```

Run the 256×256 resolution evaluation script:
```
bash scripts/evaluation/imagenet_evaluation_256_dist.sh
```

Run the original resolution evaluation script:
```
bash scripts/evaluation/imagenet_evaluation_original_dist.sh
```

- **Evaluation on MS-COCO Val2017**

The dataset should be organized as follows:
```
MSCOCO2017
└── val2017/
    ├── ...
```

Run the evaluation script:
```
bash scripts/evaluation/mscocoval_evaluation_256_dist.sh
```

Run the original resolution evaluation script:
```
bash scripts/evaluation/mscoco_evaluation_original_dist.sh
```


### Inference

Simply test the effect of each model reconstruction:
```
bash scripts/inference/reconstruct_image.sh
```

<p align="center">
  <img src="./assets/compare.png" width="90%">
  <br>
  <em>Qualitative comparison of 512 × 512 image reconstruction on TokBench.</em>
</p>

<p align="center">
  <img src="./assets/gen.png" width="90%">
  <br>
  <em>WeTok-AR-XL generated samples at 256 × 256 resolution.</em>
</p>



## ❤️ Acknowledgement
Our work builds upon the foundations laid by many excellent projects in the field. We would like to thank the authors of [Open-MAGVIT2](https://arxiv.org/abs/2409.04410). We also drew inspiration from the methodologies presented in [LFQ](https://arxiv.org/abs/2310.05737), [BSQ](https://arxiv.org/abs/2406.07548). We are grateful for their contributions to the community.
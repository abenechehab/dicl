<div align="center">
<h1>DICL: Disentangled In-Context Learning for Multivariate Time-Series Forecasting 📊</h1>

[![paper](https://img.shields.io/static/v1?label=arXiv&message=2402.03885&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/2410.11711)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/papers/2410.11711)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://opensource.org/license/MIT)
[![Python: 3.9](https://img.shields.io/badge/Python-3.9-blue)]()

</div>

## 🔥 **Exciting News!** 🔥
Our paper has been **accepted at ICLR 2025**! 🎉📄🎓  
👉 **[Zero-shot Model-based Reinforcement Learning using Large Language Models](https://openreview.net/forum?id=uZFXpPrwSh)**  

---

This repository contains the **official implementation** of the paper:  

> 📖 Abdelhakim Benechehab, Youssef Attia El Hili, Ambroise Odonnat, Oussama Zekri, Albert Thomas, Giuseppe Paolo, Maurizio Filippone, Ievgen Redko, Balázs Kégl.  
> **[Zero-shot Model-based Reinforcement Learning using Large Language Models](https://openreview.net/forum?id=uZFXpPrwSh)**  

### 📌 Abstract  
The emerging zero-shot capabilities of Large Language Models (LLMs) have led to their applications in areas extending well beyond natural language processing tasks. In reinforcement learning, while LLMs have been extensively used in text-based environments, their integration with continuous state spaces remains understudied. In this paper, we investigate how pre-trained LLMs can be leveraged to predict in context the dynamics of continuous Markov decision processes. We identify handling multivariate data and incorporating the control signal as key challenges that limit the potential of LLMs' deployment in this setup and propose **Disentangled In-Context Learning (DICL)** to address them. We present proof-of-concept applications in two reinforcement learning settings: model-based policy evaluation and data-augmented off-policy reinforcement learning, supported by theoretical analysis of the proposed methods. Our experiments further demonstrate that our approach produces well-calibrated uncertainty estimates.

![main figure](figures/main_figure_for_repo.PNG)  

---

## ⚙️ Installation  

🔹 **Create a conda environment**:  
```bash
conda create -n dicl python=3.9
```  
🔹 **Activate the environment**:  
```bash
conda activate dicl
```  
🔹 **Install the package**:  
```bash
pip install .
```  
🔹 **Install RL dependencies** (for **DICL-SAC**):  
```bash
pip install .[rl]
```  
🔹 **For developers** (includes pre-commit hooks 🛠️):  
```bash
pip install -e .[dev]
pre-commit install
```  

---

## 🚀 Getting Started  

### **DICL**  
- Try our **multivariate time series forecasting** method using the [getting started notebook](getting_started.ipynb).  

### **DICL-SAC**  
- Install the RL dependencies and run **DICL-SAC** with:  
```bash
dicl-sac --seed $RANDOM --env-id HalfCheetah --total-timesteps 1000000 --exp_name "test_5p_vicl" --batch_size 128 --llm_batch_size 7 --llm_learning_frequency 256 --context_length 500 --interact_every 1000 --learning_starts 5000 --llm_learning_starts 10000 --llm_model 'meta-llama/Llama-3.2-1B' --method 'vicl'
```  
**Arguments**:  
![dicl_sac_args](figures/dicl_sac_args.PNG)  

### **SAC (Baseline)**  
- Run **Soft Actor-Critic (SAC)** as a baseline:  
```bash
sac --seed $RANDOM --env-id "HalfCheetah" --total-timesteps 1000000 --exp_name "test_baseline" --interact_every 1000 --batch_size 128 --learning_starts 5000
```  
**Arguments**:  
![sac_args](figures/sac_args.PNG)  

---

## 📂 Directory Structure  

Overview of the repository’s contents (inside `/src/dicl/`):  

- `dicl.py` → The main class `DICL` that orchestrates the framework
- `icl/` → Core LLM usage classes such as `ICLTrainer` 
- `rl/` → Scripts for **SAC baseline** & **DICL-SAC** 
- `data/` → Sample dataset from **D4RL** for testing
- `utils/` → Helper functions

---

## 📜 License  

🔓 This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.  

---

## 🌍 Open-source Participation  

We **welcome contributions**!
Feel free to **submit issues, PRs, or feedback**—your ideas help improve DICL. 

---

## 🙏 Credits  

A big **thank you** to:  

- **[CleanRL](https://github.com/vwxyzjn/cleanrl)** for their **SAC implementations** 
- The authors of the paper **[LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law](https://github.com/AntonioLiu97/llmICL)** for access to their codebase

---

## 📚 Citing DICL  

If you use DICL in your work, please cite **our ICLR 2025 paper**:  

```bibtex
@inproceedings{
benechehab2025zeroshot,
title={Zero-shot Model-based Reinforcement Learning using Large Language Models},
author={Abdelhakim Benechehab and Youssef Attia El Hili and Ambroise Odonnat and Oussama Zekri and Albert Thomas and Giuseppe Paolo and Maurizio Filippone and Ievgen Redko and Bal{\'a}zs K{\'e}gl},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=uZFXpPrwSh}
}
```

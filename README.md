# Tencent Advertising Algorithm Competition — Generative Recommender

## Project overview

This repository contains code and experiments for a generative sequential recommender built for the Tencent Advertising Algorithm Competition. This is a state-space model specifically for generative recommender trained on large-scale interaction data (~1M users, ~4M items). The implementation uses PyTorch, and NumPy, and follows a Hydra configuration-driven architecture inspired by the paper: https://arxiv.org/html/2504.07398v1.

The goal is to improve standard ranking metrics (HR@10, Recall@K) through feature engineering, architecture experiments (Hydra-style multi-branch generative encoder), negative sampling, and careful model fine-tuning.

---

## Highlights

- Technology: **PyTorch**, **Transformers**, **NumPy**, **Hydra** 
- Dataset scale: ~1,000,000 users and ~4,000,000 items (interaction logs)
- Model: Generative sequential recommender using Hydra-inspired architecture
- Evaluation: Hit Rate @ 10 (HR@10), Recall @ K (Recall@K)





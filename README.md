# EvoVLA - Vision–Language Alignment via Evolutionary Semantic Embeddings (CLIP-Style CIFAR-100)

Designed EvoVLA, a **CLIP-style zero-shot image–text alignment model** where CIFAR-100 images are classified via nearest-neighbour retrieval in a shared embedding space. Trained Skip-Gram text embeddings on Visual Genome and enhanced class semantics using evolutionary anchor insertion, achieving ~90% Recall@10 with stronger semantic clustering across related labels. The images and class-label words are mapped into a shared embedding space. Unlike standard CLIP pipelines that rely on pre-trained language models, this work builds a **visually grounded semantic embedding space** and aligns image features to it using contrastive learning.

The system achieves **~90% Recall@10** on image→text retrieval and demonstrates strong class-wise semantic alignment, stable neighbourhood structures, and meaningful OOD retrieval behaviour.

---

## Table of Contents

- [Overview](#-project-overview)
- [Architecture](#-system-architecture)
- [Training Pipeline](#-training-pipeline)
- [Evaluation & Results](#-evaluation--results)
- [Future Work](#-future-work)

---

## Project Overview

This project explores **vision–language alignment without using large pretrained text encoders**.

Instead:

1. A **Skip-Gram model** is trained on a Visual Genome co-occurrence network  
2. CIFAR-100 class labels are inserted into the embedding space using an  
   **Evolutionary Anchor-based semantic insertion method (Weighted-Anchor (1+lambda) GA Insertion)**
3. A frozen CNN image encoder (MobileNetV3) is trained with a  
   **projection head + symmetric InfoNCE loss**  
   to align images with semantically coherent word embeddings

The result is a **structured multimodal embedding space** that is:

- visually grounded  
- semantically interpretable  
- robust across classes and unseen examples

   Nuance: The Skip-Gram model learns semantic relationships from Visual Genome co-occurrence data, and an evolutionary anchor-insertion process integrates CIFAR-100 class labels into this space. The projection head is then trained to align CNN image embeddings to these semantically-enriched word representations, enabling the model to map images into a concept space shaped by real-world visual context.

---

## System Architecture

### 1) Text Embedding Space — Skip-Gram over Visual Co-Occurrence Graph

A co-occurrence network was built from **Visual Genome captions**.

A Skip-Gram model was trained to learn:

- contextual visual relationships
- semantic similarity structure
- neighbourhood-preserving embedding geometry

The resulting vocabulary reflects **how objects co-occur in real scenes**, rather than purely linguistic usage.

Rationale for the need of co-occurence network:
The co-occurrence graph was used instead of training Skip-Gram directly on the raw text corpus because it provides a structured, globally consistent representation of how objects appear together in the visual world. Rather than relying only on local sentence windows, the graph aggregates co-occurrence statistics across all Visual Genome region descriptions, forming a weighted network where nodes represent object concepts and edges encode the strength of their visual relationships. This produces a smoother and more semantically meaningful topology — grouping objects such as apple–orange–pear or lizard–snake–crocodile according to shared context — which aligns more closely with image features than purely linguistic similarity. The graph also makes neighbourhood structure explicit, enabling us to evaluate, preserve, and optimise semantic clusters during evolutionary anchor insertion. This is crucial for robust alignment in our contrastive learning stage, as the resulting text embedding space is not only linguistically coherent, but also visually grounded and structurally stable for zero-shot retrieval and generalisation.

---

### 2) Evolutionary Anchor-Based Label Insertion (EA)

CIFAR-100 labels were inserted into the embedding space using an **evolutionary search with weighted-anchors**, instead of random or mean-based placement.

EA optimisation preserved:

- local semantic neighbourhoods  
- cosine-similarity structure  
- class-cluster continuity

Examples:

- reptiles form a stable cluster (lizard, crocodile, snake)
- vehicle classes align together
- trees + plants form meaningful hierarchies

This produces a **523-word enriched semantic vocabulary space**.

---

### 3) Vision Encoder + InfoNCE Projection Head

- Backbone: **MobileNetV3-Small (frozen)**
- Output features: 576-dim
- Trainable projection head maps features → text embedding space

Training objective:

✔ symmetric **InfoNCE contrastive loss**  
✔ aligns image & text embeddings  
✔ pushes apart negative pairs

This encourages:

- cross-modal consistency
- interpretable clusters
- robust image→text retrieval

---

## Training Pipeline (End-to-End)
Visual Genome captions → build co-occurrence graph → Train Skip-Gram word embeddings → Evolutionary Anchor Insertion for CIFAR-100 labels → Freeze text embedding space (semantic prior) → Train projection head using contrastive InfoNCE → Evaluate retrieval, alignment & OOD behaviour

The projection head learns to **adapt image features into a fixed, semantically structured space**, rather than learning text + image jointly — reducing overfitting risk and improving interpretability.

---

## Evaluation & Results

Evaluation is retrieval-based, not classification-based.

Metrics include:

- Image→Text Recall@K
- Text→Image Recall@K
- Per-class similarity distribution
- Neighbourhood structure preservation
- OOD retrieval behaviour

### Achieved Performance

- **~90% Recall@10 (image→text retrieval)**
- Strong alignment in vehicles, plants, structures, tools
- Meaningful class clusters & similarity hierarchies

<img width="1558" height="1204" alt="image" src="https://github.com/user-attachments/assets/fbb432bb-16fb-4c88-828d-9a607bda82d5" />

This indicates:

- stable semantic geometry
- visually grounded embedding relationships
- cross-modal correspondence

## Future Work

Planned extensions:

- Joint training of both modalities
- Few-shot novel class embedding insertion
- Zero-shot learning with relational prompts
- Graph neural networks over embedding neighbourhoods
- Compare against CLIP text encoder representations

---

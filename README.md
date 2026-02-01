# E-FineR: Vocabulary-free Fine-grained Visual Recognition

**E-FineR** is a training-free, fully automated framework for **vocabulary-free fine-grained visual recognition**. This repository accompanies the research paper:

> **Vocabulary-free Fine-grained Visual Recognition via Enriched Contextually Grounded Vision-Language Model** (ICCV '25)
> Dmitry Demidov, Zaigham Zaheer, Omkar Thawakar, Salman Khan, Fahad Shahbaz Khan
> Mohamed bin Zayed University of Artificial Intelligence
> [[arXiv]](https://arxiv.org/abs/2507.23070v1)
> [[ICCV]](https://openaccess.thecvf.com/content/ICCV2025W/MMFM/html/Demidov_Vocabulary-free_Fine-grained_Visual_Recognition_via_Enriched_Contextually_Grounded_Vision-Language_Model_ICCVW_2025_paper.html)

The method achieves state-of-the-art results on multiple fine-grained datasets without requiring predefined class labels, expert annotations, prompt engineering, or training.

---

## Overview

Fine-grained visual recognition (FGVR) is challenging due to subtle inter-class differences (e.g., bird species, car models). Traditional methods rely on fixed vocabularies and supervision. **E-FineR** removes these constraints by:
* Automatically discovering class names from unlabelled images.
* Generating rich, class-specific contextual descriptions using VLMs.
* Coupling language and vision representations for robust classification.
* Operating fully **training-free**, **vocabulary-free**, and **human-free**.

The framework supports:
* Vocabulary-free recognition
* Zero-shot classification
* Few-shot classification

All within a single, unified pipeline.

---

## Key Contributions

* **Class-specific Contextual Grounding**
  Automatically generates rich, descriptive, and diverse class-specific in-context sentences using large vision-language models.
* **Advanced Class Name Filtration**
  Retains multiple semantically plausible class candidates instead of forcing top-1 selection, improving robustness and recall.
* **Vision-Language Prompt Coupling**
  Combines text and visual embeddings to form stronger class representations without retraining.
* **Fully Automated & Training-free**
  No manual prompts, no training, no expert annotations, no predefined class lists.

---

## Usage

### 1. Setup Environment


1.1 Set up environment using PIP:

```bash
pip install -r envs/pip_requirements.txt
```

1.2 Alternatively, set up environment using Conda:

```bash
conda env create -f envs/conda_environment.yml
# or
conda create --name e_finer --file envs/conda_requirements.txt
```

```bash
conda activate e-finer
```

### 2. Prepare Datasets
For dataset download and preparation, please follow a beautifully written guide available [here](https://github.com/OatmealLiu/FineR?tab=readme-ov-file#-datasets-preparation).
All meta data needed for the supported datasets is provided in the `data/data_stats.py` file.

### 3. Classname Discovery
The discovered class names are provided in the `data/guessed_classnames/` directory for all supported datasets.
For classname discovery in a custom dataset, please utilize FineR approach [here](https://github.com/OatmealLiu/FineR?tab=readme-ov-file#%EF%B8%8F-full-pipeline).

### 4. Generate In-Context Sentences
The generated in-context sentences are provided in the `data/generated_context/` directory for all supported datasets.
To re-generate class-specific in-context sentences (or generate for a custom dataset), modify the generation config in `generate_context.py` and run:

```bash
python generate_context.py
```

### 5.1 Vocabulary-free Classification
To perform vocabulary-free classification on supported datasets, run the corresponding evaluation scripts:

```bash
sh run/eval_birds.sh
sh run/eval_cars.sh
sh run/eval_dogs.sh
sh run/eval_flowers.sh
sh run/eval_pets.sh
```

### 5.2 Zero-shot Classification

```bash
TOADD
```

### 5.3 Few-shot Classification

```bash
TOADD
```

---

## Repository Structure

```
e-finer/
├── configs/               # Configuration files for experiments
├── data/                  # Dataset loaders, preprocessing, generated in-context sentences
├── datasets/              # Fine-grained datasets
├── envs/                  # Environment setup files
├── models/                # Vision-language interfaces
├── run/                   # Entry-point scripts for experiments
├── utils/                 # Helper utilities
└── README.md
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{Demidov_2025_ICCV,
    author    = {Demidov, Dmitry and Zaheer, Muhammad Zaigham and Thawakar, Omkar and Khan, Salman and Khan, Fahad Shahbaz},
    title     = {Vocabulary-free Fine-grained Visual Recognition via Enriched Contextually Grounded Vision-Language Model},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2025},
    pages     = {4216-4225}
}
```

---

## Contacts

For questions or collaborations:

* **Dmitry Demidov** – [dmitry.demidov@mbzuai.ac.ae](mailto:dmitry.demidov@mbzuai.ac.ae)

---

## ⭐ Acknowledgements

This project builds upon and integrates ideas from [CLIP](https://github.com/openai/CLIP), [FineR](https://github.com/OatmealLiu/FineR), and recent advances in LLM-based vision-language modeling. 
We are thankful to the corresponding authors for making their code public.
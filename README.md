# Brain Tumor MRI Segmentation with Deep Convolutional Neural Networks

**Group Number:** 10  
**Group Members:** Adriaan Jacquet, Matvey Sivashinskiy, Arne Wiese

---

## 🧠 Purpose

The goal of this project is to develop and evaluate deep learning models for **automated segmentation of brain tumors in MRI scans**.  
Accurate tumor segmentation is critical for **diagnosis**, **treatment planning**, and **prognosis** in neuro-oncology.

We aim to compare **2D and 3D U-Net-based convolutional neural networks (ConvNets)** for this task and explore how architectural and methodological choices affect segmentation quality.  
As an optional extension, we will investigate using segmentation outputs for **patient survival prediction**.

---

## 📚 Dataset

We use the **BraTS 2020 dataset** ([Kaggle link](https://www.kaggle.com/)), which contains **multimodal MRI scans** and **expert-annotated tumor segmentation masks** for 369 patients.

Each patient includes:

- **Four MRI modalities:** T1, T1c, T2, and FLAIR
- **Ground-truth label volume:** 240 × 240 × 155
- **Pixel values:**
  - 0 → Background
  - 1 → Necrotic/non-enhancing tumor
  - 2 → Edema
  - 3 → Enhancing tumor

---

## ⚙️ Methods

### **Baseline**

- **Model:** 2D U-Net
- **Input:** Each axial slice with 4 MRI modalities as channels
- **Pros:** Fast, low memory usage, can leverage pretrained encoders
- **Cons:** Lacks 3D spatial context, may struggle with small or fragmented regions

### **Variants**

1. **2.5D U-Net** – Stack neighboring slices as input channels
2. **Multi-view Ensemble** – Combine predictions from axial, coronal, and sagittal views
3. **3D U-Net** – Use true 3D convolutions on patches for richer spatial context
   - _(Higher memory and compute requirements)_

### **Evaluation Metrics**

- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU / Jaccard Index)
- Pixel-wise Accuracy (on held-out validation set)

---

## 🔬 Planned Experiments

1. Compare **2D vs. 3D U-Net** architectures for tumor segmentation
2. Test **2.5D** and **multi-view ensemble** strategies to enhance 2D performance
3. Investigate **data augmentation**, **regularization**, and **transfer learning** (e.g., pretrained ImageNet encoders)
4. _(Optional)_ Use segmentation outputs for **survival prediction** using radiomic or deep features

---

## 📖 Key References

- Menze et al., _The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)_, **IEEE TMI**, 2015.  
  DOI: [10.1109/TMI.2014.2377694](https://doi.org/10.1109/TMI.2014.2377694)

- Bakas et al., _Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features_, **Nature Scientific Data**, 2017.  
  DOI: [10.1038/sdata.2017.117](https://doi.org/10.1038/sdata.2017.117)

- Bakas et al., _Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge_, **arXiv:1811.02629**, 2018.  
  [arXiv link](https://arxiv.org/abs/1811.02629)

---

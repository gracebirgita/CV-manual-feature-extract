# Computer Vision: Manual Feature Extraction

### Links : 
🔗 **Deployed App**: https://manual-feat-extract.streamlit.app/

🎥 **Demo Video**: https://youtu.be/kgpmrzNLvqI?si=cYbvgshvjJo3VGwp

📄 **Project Report**: https://drive.google.com/file/d/1OyySj7PCRhPQvbjKJ6OG3vkzq4qkKXkv/view?usp=sharing

<br>

## 📝 Description
This project presents a practical implementation of manual feature extraction techniques in Computer Vision, focusing on classical, handcrafted visual descriptors rather than Deep Learning–based approaches.
The main objective is to analyze the effectiveness of HOG and LBP features for object classification under limited data and class imbalance conditions, using a filtered subset of the COCO2017 dataset. Unlike end-to-end deep models, this pipeline emphasizes:
- Feature interpretability
- Lightweight computation
- Robustness on small and imbalanced datasets

## Objectives
- Implement Histogram of Oriented Gradients (HOG) and Local Binary Patterns (LBP) from scratch pipelines
- Reduce background noise using bounding-box–based cropping
- Compare multiple experimental configurations involving:
  - PCA
  - Data augmentation
- Evaluate fairness and per-class generalization using classical metrics

<br>

## 📦 Dataset

**Dataset Source**: COCO2017 (filtered subset)

### Selected Object Classes

| Class Name | COCO ID |
|-----------|---------|
| Person    | 1       |
| Bicycle   | 2       |
| Car       | 3       |
| Dog       | 18      |

### Dataset Filtering Strategy

- Only images containing at least one object from the selected classes are included.
- Each image is assigned **one single label** using:
  - max of bounding box area among valid classes
- Images are cropped to the selected bounding box so that:
  - Feature extraction focuses on object regions
  - Background noise is minimized
  - The SVM classifier learns object-centric visual representations

<br>

## 📊 Evaluation Metrics

The following metrics are used to evaluate model performance:

- Precision
- Recall
- F1-score
- Accuracy
- Support
- Macro-average F1-score
- Weighted-average F1-score

Special attention is given to **per-class recall**, emphasizing robustness on underrepresented classes.



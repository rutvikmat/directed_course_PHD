# Dataset description

**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), available through `sklearn.datasets.load_breast_cancer`.

**Task:** Binary classification of breast tumour observations as malignant or benign.

**Inputs:** 30 numeric features derived from digitised images of fine needle aspirate (FNA) samples, including radius, texture, perimeter, area, smoothness, compactness, concavity and symmetry summaries.

**Target:** `0 = malignant`, `1 = benign` (the original scikit-learn target encoding).

**Source:** UCI Machine Learning Repository, Wisconsin Diagnostic Breast Cancer dataset; accessed through scikit-learn's built-in dataset loader. No external download is required.

**Limitations:** This is a historical benchmark dataset, not representative of all current clinical populations, scanners, hospitals, or care pathways. The features are image-derived measurements rather than a complete patient record. It is suitable for an educational classification/XAI exercise, not clinical deployment.

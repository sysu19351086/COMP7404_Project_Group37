# COMP7404 Group Project

## 1 Introduction
We choose paper **Efﬁcient kNN Classiﬁcation With Different Numbers of Nearest Neighbors** for this group project. The paper introduces two innovative kNN classification algorithms: **kTree** and **k*Tree**. Unlike traditional kNN methods that assign a fixed k value to all test samples, these methods learn different optimal k values for different test samples by incorporating a training stage. The kTree method constructs a decision tree during the training phase, where each leaf node stores the optimal k value for the training samples in that node. In the testing phase, the kTree quickly outputs the optimal k value for each test sample, allowing efficient kNN classification. The k*Tree method further enhances efficiency by storing additional information in the leaf nodes, such as the training samples and their nearest neighbors, enabling kNN classification using a subset of training samples rather than the entire dataset. These methods address the limitations of traditional kNN approaches, which either assign a fixed k value or are computationally expensive when assigning different k values to different samples. The proposed methods were compared with other state-of-the-art kNN methods using **20 UCI data sets**, demonstrating their effectiveness and efficiency.

## 2 Environment Setup
The experimental environment for reproducing the code of the paper involves the following configurations and tools:

### **Hardware Environment**

-   **Processor**: Intel i7 CPU
    
-   **GPU**: NVIDIA RTX 4060 (used for accelerated computations)
    
-   **Memory**: Sufficient RAM to handle dataset loading and processing (exact specifications not provided).
    

### **Software Environment**

-   **Programming Language**: Python 3.12
    
-   **Deep Learning Framework**: PyTorch (for efficient tensor operations and GPU acceleration)
    
-   **Machine Learning Libraries**:
    
    -   `scikit-learn` (for implementing kNN and other baseline algorithms)
        
    -   `pandas` (for data manipulation and preprocessing)
        
    -   `numpy` (for numerical computations)
        
-   **Utility Libraries**:
    
    -   `tqdm` (for tracking the progress of algorithms during execution)
        
    -   `matplotlib` (for visualizing results, such as classification accuracy and runtime comparisons).
        

### **Key Features**

-   The environment leverages PyTorch for GPU acceleration, enabling faster computations, especially for large datasets.
    
-   `scikit-learn` provides a robust framework for implementing and comparing traditional kNN methods.
    
-   `tqdm` ensures efficient monitoring of algorithm performance during training and testing.
    
-   `matplotlib` is used to generate visualizations for analyzing the results of the proposed methods (kTree and k*Tree) against other state-of-the-art kNN algorithms.

## 3 Working on the project
The experimental setup for reproducing the code involves implementing various kNN algorithms and evaluating their performance on 20 UCI datasets. Below is a detailed description of the implementation and experimental design:

### **Implemented Algorithms**

1.  **Proposed Methods**:
    
    -   **kTree**: Implemented in `ktree.py`
        
    -   **k*Tree**: Implemented in `kstartree.py`
        
2.  **Baseline and Comparison Methods**:
    
    -   **Traditional kNN**: Implemented in `knn.py`
        
    -   **AD-kNN**: Implemented in `adknn.py`
        
    -   **S-kNN**: Implemented in `sknn.py` (based on the paper "1_kNNAlgorithmwithData-DrivenkValue")
        
    -   **GS-kNN**: Implemented in `gsknn.py` (based on the paper "Efficient kNN algorithm based on graph sparse reconstruction")
        
    -   **FASBIR**: Implemented in `fasbir.py` (based on the paper "Efficient kNN Classification Algorithm for Big Data")
        
    -   **LC-kNN**: Implemented in `lcknn.py` (based on the paper "Ensembling local learners ThroughMultimodal perturbation")
        

All algorithms were independently implemented in Python, as the original code for these methods is not publicly available.

### **Datasets**

The experiments were conducted on 20 datasets from the UCI Machine Learning Repository, divided into two groups:

1.  **Datasets for Evaluating Performance Under Different Sample Sizes**:
    
    -   `abalone`, `australian`, `balance`, `blood`, `breast`, `car`, `client`, `climate`, `german`, `mice`
        
2.  **Datasets for Evaluating Performance Under Different Feature Counts**:
    
    -   `arcene`, `arrhythmia`, `cnae`, `dbworld`, `gisette`, `hill`, `libras`, `lsvt`, `madelon`, `musk`
        

All datasets are stored in a `datasets` folder.

### **Training and Evaluation**

-   For each dataset, a dedicated Python script was written to train and evaluate all algorithms. These scripts are named after the datasets (e.g., `abalone.py`, `arcene.py`), resulting in a total of 20 training scripts.
    
-   Each script performs the following tasks:
    
    1.  Loads the dataset.
        
    2.  Runs each algorithm (kTree, k*Tree, traditional kNN, AD-kNN, S-kNN, GS-kNN, FASBIR, LC-kNN).
        
    3.  Records the classification accuracy and runtime for each algorithm.
        
    4.  Saves the results for further analysis and visualization.
        

### **Result Visualization**

-   The results (classification accuracy and runtime) for each algorithm on each dataset are printed during execution.
    
-   These results are saved and visualized using `matplotlib`, with plots stored in an `outputs` folder.
    
-   Visualizations include comparisons of classification accuracy and runtime across different sample sizes and feature counts.

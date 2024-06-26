# geo5017_A2

# Point Cloud Classification README

This guide provides detailed instructions on running the point cloud classification script, managing data and output files, and customizing training/testing parameters.

## Introduction

The provided Python script is designed for the classification of point cloud data into distinct categories such as buildings, cars, fences, poles, and trees. It employs various machine learning algorithms, including SVM and Random Forest, and provides tools for feature extraction, model tuning, and performance evaluation.

## Prerequisites

Ensure the following Python libraries are installed:

- matplotlib
- numpy
- pandas
- scikit-learn

Installation can be done via pip:

```bash
pip install matplotlib numpy pandas scikit-learn
```

## File Structure

Before running the script, place it inside the code directory. The script and associated modules should reside here. All input and output directories will be relative to this code directory, maintaining a coherent project hierarchy.

- **Input Data**: Place your point cloud data files in .xyz format within the ../pointclouds-500 directory, relative to the code folder. The script expects these files to be named sequentially (e.g., 0.xyz, 1.xyz, etc.) and categorizes them based on their filenames.
- **Output Directories**: Upon execution, the script will generate the following directories alongside the code directory:
  - `result_both`: This directory is used for storing numpy arrays generated during feature extraction and model predictions.
  - `figures`: This directory is intended for saving visual output such as confusion matrices and learning curves.
  
If the directories do not exist, the code will create them.

## Running the Code

Execute the script with the following command:

```bash
python main.py
```

## Configuration and Output

### Data Path

Modify the `folder_path` variable to point to your input `.xyz` data directory, if you locate them elsewhere. The script expects files named sequentially (e.g., `0.xyz`, `1.xyz`, etc.) and categorizes them based on their filenames.

### Output Files

- Feature data and labels are saved in `result_both/X_{i}_{n}.npy` and `result_both/y.npy`, where `{i}` and `{n}` represent iteration indices and slice numbers, respectively.
- Figures such as confusion matrices and learning curves are stored in the `figures` directory with descriptive filenames.

### Modifying Training/Testing Splits

The script includes parameters `train_size` and `test_size` for adjusting the proportions of data used for training and testing:

```python
train_size = 0.7
test_size = 0.3
```

By default, 70% of the data is used for training and 30% for testing. Alter these values to explore different training/testing configurations and potentially impact the model's performance.

### Model Parameters and Feature Selection

The script contains sections for tuning model parameters and selecting features. Users can modify these sections to experiment with different configurations and improve classification results.

## Result Analysis

After execution, the script provides:

- Accuracy metrics and confusion matrices to evaluate model performance.
- Learning curves to understand model behavior with varying data sizes.
- Cross-validation scores to assess the robustness of the models. Per default set to 5.

## Conclusion

This README aims to assist users in navigating and utilizing the point cloud classification script effectively. Users are encouraged to modify the script to suit their specific data sets and research needs, exploring different model configurations and parameters for optimal results.
```

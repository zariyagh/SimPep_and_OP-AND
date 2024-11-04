# SimPep and OP-AND
Overview

OP-AND is a curated public database of osteogenic peptides, and SimPep is a deep learning framework for predicting Osteogenic properties of peptides using OP-AND.

## Notebook Structure

The notebook consists of several key sections:

1. Configuration Setup

This initial part of the notebook includes importing necessary libraries and setting up the environment, such as mounting drives and loading essential packages like TensorFlow and scikit-learn.

2. Function Definitions

Custom functions that power the SimPep framework are defined here. They include:

	•	makedatasetPos(): Generates positive datasets.
	•	makedatasetNeg(): Creates negative datasets.
	•	InternalTest(), ExternalTest(), RealTest(): Validation of SimPep-Net, Prediction of OPD, and case study.
	•	Accuracy(): Computes metrics like AUC and AUPR.
	•	SelectName(): Handles naming and cutoff for prediction results.

3. Data Preparation

This section involves:

	•	Loading positive and negative protein embeddings using ProtBert-based vectorization.
	•	Combining datasets and splitting them into training and testing subsets.

4. Training Dataset Construction

Balanced datasets are built by concatenating samples for training and testing the model.

5. SimPep-Net Framework Overview

This section describes the core structure of the SimPep-Net, which includes defining the model architecture:

	•	A Siamese network setup using paired inputs for analysis.

6. Training and Optimization

Training the SimPep-Net is optimized using various callback functions, training sets, and hyperparameters.

7. Validation and Testing

The model’s performance is evaluated using internal and external test datasets to validate its predictive capabilities.

8. Case Study Evaluation

SimPep tests real-world peptide data loaded from files and processes these data points for interaction predictions.

9. Additional Test Sets

Further assessments include external negative datasets and new operational peptide data from recent years.

Requirements

	•	Python 3.x
	•	TensorFlow for deep learning operations
	•	scikit-learn for dataset manipulation and metrics
	•	NumPy and Pandas for data handling
	•	Access to protein embedding files formatted for input

Usage Instructions

	1.	Set Up Your Environment:
Ensure that the necessary files (e.g., vec_pos.txt, vec_neg_Q5T9C2.txt, etc.) are available and paths are correctly set.
	2.	Run the Notebook Sequentially:
Execute the cells in sequence to load libraries, define functions, and process the datasets.
	3.	Train the Model:
Use the provided training code to train SimPep-Net. Adjust the Epoch and other parameters as needed.
	4.	Predict and Analyze:
Use the trained model to run predictions on test datasets and analyze the outcomes.

Key Considerations

	•	Ensure your machine has sufficient RAM for handling large protein datasets.
	•	Modify paths as per your directory structure for loading datasets and saving results.

Customization

You can extend the notebook by:

	•	Modifying the network structure.
	•	Adding custom data processing steps.
	•	Integrating other deep learning models for comparative studies.

Contact and Support

For issues or questions regarding the SimPep framework, please contact z_ghorbanali@aut.ac.ir or f.zare@aut.ac.ir.
<img width="468" alt="image" src="https://github.com/user-attachments/assets/4c8dece8-7cd1-4d28-8eab-46010e7b4029">

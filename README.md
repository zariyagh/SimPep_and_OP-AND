# SimPep and OP-AND

# SimPep and OP-AND

In this work, we introduce **OP-AND**, a curated public database of **osteogenic peptides**.  
We also propose a novel hypothesis: peptides derived from proteins involved in **osteoclast formation** may serve as **non-osteogenic**.  

Considering the limited availability of OP data, we present **SimPep**, a deep learning framework achieving:

- **Accuracy**: 92.93%
- **Specificity**: 96.64%
- **Sensitivity**: 70.58%  
(using 5-fold cross-validation)

---

## 📁 Repository Contents

| File / Folder | Description |
|---------------|-------------|
| `OP-AND.xlsx` | Contains osteogenic peptides collected from literature. The second sheet includes references for each peptide. |
| `PPP_ProtBERT_embeddings.txt` | ProtBERT embeddings of osteogenic peptides (OPs). |
| `NPP_O88942_ProtBERT_embeddings.txt` | Embeddings of peptides derived from **O88942** protein, representing non-osteogenic peptides (NPPs) under our osteoclast-based hypothesis. |
| `NPP_Q5T9C2_ProtBERT_embeddings.txt` | Same as above, but from protein **Q5T9C2**. |
| `NPP_Q9CWT3_ProtBERT_embeddings.txt` | Same as above, but from protein **Q9CWT3**. |
| `NPP_random.txt` *(if present)* | Peptides derived from random proteins — assumed non-osteogenic with no known osteogenic function. |
| `weights_only.weights.h5` | Trained weights of the final **Siamese model** used in SimPep. |
| `requirements.txt` | Python package dependencies for running the code and models. |
| `SimPep_Compelet_Model.ipynb` | Contains the full model architecture, training pipeline, and parameters — easily modifiable. |
| `SimPep_App.ipynb` | Interactive notebook that loads the trained model and predicts the **osteogenic score** of a given peptide. |

---

## 🚀 Getting Started

### To use the model:

#### 1. Clone the repo:
   ```bash
   git clone https://github.com/zariyagh/SimPep_and_OP-AND.git
```

#### 2. Install dependencies:
```
pip install -r requirements.txt
```
#### 3.Run the prediction notebook:
	•	Open SimPep_App.ipynb in Google Colab or Jupyter
	•	Enter a peptide sequence
	•	Get its osteogenic score prediction
## 📬 Contact & Submission Info

This work has been submitted for peer review to the journal **PLOS Computational Biology**.

For feedback or collaboration inquiries, feel free to contact:

- Zahra Ghorbanali: [z_ghorbanali@aut.ac.ir](mailto:z_ghorbanali@aut.ac.ir)
- Fatemeh Zare-Mirakabad (corresponding author): [f.zare@aut.ac.ir](mailto:f.zare@aut.ac.ir)

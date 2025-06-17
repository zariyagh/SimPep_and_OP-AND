import keras
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

from transformers import BertTokenizer, BertModel
import torch


app = Flask(__name__)


tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
modelb = BertModel.from_pretrained("Rostlab/prot_bert")
modelb.eval()  # inference mode

def embed_peptide_with_protbert(seq):
    sequence = ' '.join(seq)
    inputs = tokenizer(sequence, return_tensors='pt')

    with torch.no_grad():
        outputs = modelb(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)

    # گرفتن میانگین تمام توکن‌ها → (hidden_dim,)
    peptide_vector = torch.mean(embeddings, dim=0)

    return peptide_vector.numpy()

  # vec_pos: the ProtBert-Based of OPs collected from OP-AND

df_pos = pd.read_csv('https://github.com/zariyagh/SimPep_and_OP-AND/blob/main/PPP_ProtBERT_embeddings.txt', sep=",", header=None)
df_pos = df_pos.iloc[:, 1:-1]
print(df_pos.shape)
df_pos = df_pos.drop_duplicates()
print(df_pos.shape)

# vec_neg_Q5T9C2: the ProtBert-Based of NPP collected from Q5T9C2

df_neg1 = pd.read_csv('https://github.com/zariyagh/SimPep_and_OP-AND/blob/main/NPP_Q5T9C2_ProtBERT_embeddings.txt', sep=",", header=None)
df_neg1 = df_neg1.iloc[:, 1:-1]
df_neg1 = df_neg1.drop_duplicates()

# vec_neg_Q5T9C2: the ProtBert-Based of NPP collected from Q9CWT3

df_neg2 = pd.read_csv('https://github.com/zariyagh/SimPep_and_OP-AND/blob/main/NPP_Q9CWT3_ProtBERT_embeddings.txt', sep=",", header=None)
df_neg2 = df_neg2.iloc[:, 1:-1]
df_neg2 = df_neg2.drop_duplicates()

# vec_neg_O88942: the ProtBert-Based of NPP collected from O88942

df_neg3 = pd.read_csv('https://github.com/zariyagh/SimPep_and_OP-AND/blob/main/NPP_O88942_ProtBERT_embeddings.txt', sep=",", header=None)
df_neg3 = df_neg3.iloc[:, 1:-1]
df_neg3 = df_neg3.drop_duplicates()

# merging all neg data of bone loss proteins

df_neg = pd.concat([df_neg1, df_neg2, df_neg3], ignore_index=True)

X1 = df_pos.to_numpy(dtype='float')
X0 = df_neg.to_numpy(dtype='float')
Y1 = [1 for i in range(X1.shape[0])]
Y0 = [0 for i in range(X0.shape[0])]

def build_siamese_model(input_dim):
    shared_model = tf.keras.models.Sequential([
        Dense(512, input_shape=(input_dim,), activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
    ])

    left_input = Input(shape=(input_dim,))
    right_input = Input(shape=(input_dim,))

    encoded_l = shared_model(left_input)
    encoded_r = shared_model(right_input)

    L1 = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=lambda input_shape: input_shape[0])([encoded_l, encoded_r])
    L1_D = Dropout(0.2)(L1)

    prediction = Dense(16, activation='relu')(L1_D)
    prediction = Dropout(0.2)(prediction)
    prediction = Dense(16, activation='relu')(prediction)
    prediction = Dropout(0.2)(prediction)
    prediction = Dense(16, activation='relu')(prediction)
    prediction = Dropout(0.2)(prediction)
    prediction = Dense(8, activation='relu')(prediction)
    prediction = Dropout(0.2)(prediction)
    prediction = Dense(1, activation='sigmoid')(L1_D)

    model = Model(inputs=[left_input, right_input], outputs=prediction)
    return model

input_dim = 1024  
model = build_siamese_model(input_dim)
model.load_weights('https://github.com/zariyagh/SimPep_and_OP-AND/blob/main/weights_only.weights.h5')

x0_train = X0
x1_train = X1

def RealTest(X, x0_train, x1_train):
    dataset_test = []
    Indexsample = []
    for i in range(len(X)):
        for j in range(len(x0_train)):
            dataset_test.append(np.concatenate((X[i], x0_train[j])))
            Indexsample.append(f"{i},0")
        for j in range(len(x1_train)):
            dataset_test.append(np.concatenate((X[i], x1_train[j])))
            Indexsample.append(f"{i},1")
    return np.asarray(dataset_test), Indexsample

def RealPredict(X, test_pred, Indexsample):
    Pred = []
    for i in range(len(X)):
        cnt0Pred, cnt1Pred, cnt0, cnt1 = 0, 0, 0, 0
        for j in range(len(Indexsample)):
            triplet = Indexsample[j].split(",")
            if int(triplet[0]) == i:
                if int(triplet[1]) == 1:
                    cnt1Pred += test_pred[j]
                    cnt1 += 1
                else:
                    cnt0Pred += test_pred[j]
                    cnt0 += 1
        pos = (1 - (cnt0Pred / cnt0)) + (cnt1Pred / cnt1)
        neg = (cnt0Pred / cnt0) + (1 - (cnt1Pred / cnt1))
        Pred.append(float(pos / (pos + neg)))
    return Pred

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        peptide_seq = data["peptide_sequence"] 
        peptide_vector = embed_peptide_with_protbert(peptide_seq).reshape(1, -1)
        main_test, index_sample = RealTest(peptide_vector, x0_train, x1_train)
        left = main_test[:, :input_dim]
        right = main_test[:, input_dim:]

        y_pred = model.predict([left, right])
        final_score = RealPredict(peptide_vector, y_pred, index_sample)

        return jsonify({
            "osteogenic_score": final_score[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

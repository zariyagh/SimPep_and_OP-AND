{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "18Pezvf5T7zsIkWlKlaCFf-yvebVvHBKz",
      "authorship_tag": "ABX9TyPNWntCfxfowDx3JxyN/Be/",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/zariyagh/SimPep_and_OP-AND/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import keras\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from flask import Flask, request, jsonify\n",
        "\n",
        "from transformers import BertTokenizer, BertModel\n",
        "import torch\n",
        "\n",
        "\n",
        "app = Flask(__name__)\n",
        "\n",
        "\n",
        "# فقط یک بار بارگذاری مدل و توکنایزر\n",
        "tokenizer = BertTokenizer.from_pretrained(\"Rostlab/prot_bert\", do_lower_case=False)\n",
        "modelb = BertModel.from_pretrained(\"Rostlab/prot_bert\")\n",
        "modelb.eval()  # inference mode\n",
        "\n",
        "def embed_peptide_with_protbert(seq):\n",
        "    \"\"\"\n",
        "    تبدیل پپتید به وکتور با مدل ProtBERT\n",
        "    خروجی: وکتور numpy به طول 1024 (بسته به مدل می‌تونه متفاوت باشه)\n",
        "    \"\"\"\n",
        "    # آماده‌سازی ورودی (ProtBERT نیاز به فاصله بین اسید آمینه‌ها داره)\n",
        "    sequence = ' '.join(seq)\n",
        "    inputs = tokenizer(sequence, return_tensors='pt')\n",
        "\n",
        "    with torch.no_grad():\n",
        "        outputs = modelb(**inputs)\n",
        "        embeddings = outputs.last_hidden_state.squeeze(0)  # (seq_len, hidden_dim)\n",
        "\n",
        "    # گرفتن میانگین تمام توکن‌ها → (hidden_dim,)\n",
        "    peptide_vector = torch.mean(embeddings, dim=0)\n",
        "\n",
        "    return peptide_vector.numpy()\n",
        "\n",
        "  # vec_pos: the ProtBert-Based of OPs collected from OP-AND\n",
        "\n",
        "df_pos = pd.read_csv('/content/drive/MyDrive/Osta/PPP_ProtBERT_embeddings.txt', sep=\",\", header=None)\n",
        "df_pos = df_pos.iloc[:, 1:-1]\n",
        "print(df_pos.shape)\n",
        "df_pos = df_pos.drop_duplicates()\n",
        "print(df_pos.shape)\n",
        "\n",
        "# vec_neg_Q5T9C2: the ProtBert-Based of NPP collected from Q5T9C2\n",
        "\n",
        "df_neg1 = pd.read_csv('/content/drive/MyDrive/Osta/NPP_Q5T9C2_ProtBERT_embeddings.txt', sep=\",\", header=None)\n",
        "df_neg1 = df_neg1.iloc[:, 1:-1]\n",
        "df_neg1 = df_neg1.drop_duplicates()\n",
        "\n",
        "# vec_neg_Q5T9C2: the ProtBert-Based of NPP collected from Q9CWT3\n",
        "\n",
        "df_neg2 = pd.read_csv('/content/drive/MyDrive/Osta/NPP_Q9CWT3_ProtBERT_embeddings.txt', sep=\",\", header=None)\n",
        "df_neg2 = df_neg2.iloc[:, 1:-1]\n",
        "df_neg2 = df_neg2.drop_duplicates()\n",
        "\n",
        "# vec_neg_O88942: the ProtBert-Based of NPP collected from O88942\n",
        "\n",
        "df_neg3 = pd.read_csv('/content/drive/MyDrive/Osta/NPP_O88942_ProtBERT_embeddings.txt', sep=\",\", header=None)\n",
        "df_neg3 = df_neg3.iloc[:, 1:-1]\n",
        "df_neg3 = df_neg3.drop_duplicates()\n",
        "\n",
        "# merging all neg data of bone loss proteins\n",
        "\n",
        "df_neg = pd.concat([df_neg1, df_neg2, df_neg3], ignore_index=True)\n",
        "\n",
        "X1 = df_pos.to_numpy(dtype='float')\n",
        "X0 = df_neg.to_numpy(dtype='float')\n",
        "Y1 = [1 for i in range(X1.shape[0])]\n",
        "Y0 = [0 for i in range(X0.shape[0])]\n",
        "\n",
        "def build_siamese_model(input_dim):\n",
        "    shared_model = tf.keras.models.Sequential([\n",
        "        Dense(512, input_shape=(input_dim,), activation='relu'),\n",
        "        Dropout(0.2),\n",
        "        Dense(128, activation='relu'),\n",
        "        Dropout(0.2),\n",
        "        Dense(64, activation='relu'),\n",
        "        Dropout(0.2),\n",
        "        Dense(32, activation='relu'),\n",
        "    ])\n",
        "\n",
        "    left_input = Input(shape=(input_dim,))\n",
        "    right_input = Input(shape=(input_dim,))\n",
        "\n",
        "    encoded_l = shared_model(left_input)\n",
        "    encoded_r = shared_model(right_input)\n",
        "\n",
        "    L1 = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=lambda input_shape: input_shape[0])([encoded_l, encoded_r])\n",
        "    L1_D = Dropout(0.2)(L1)\n",
        "\n",
        "    prediction = Dense(16, activation='relu')(L1_D)\n",
        "    prediction = Dropout(0.2)(prediction)\n",
        "    prediction = Dense(16, activation='relu')(prediction)\n",
        "    prediction = Dropout(0.2)(prediction)\n",
        "    prediction = Dense(16, activation='relu')(prediction)\n",
        "    prediction = Dropout(0.2)(prediction)\n",
        "    prediction = Dense(8, activation='relu')(prediction)\n",
        "    prediction = Dropout(0.2)(prediction)\n",
        "    prediction = Dense(1, activation='sigmoid')(L1_D)\n",
        "\n",
        "    model = Model(inputs=[left_input, right_input], outputs=prediction)\n",
        "    return model\n",
        "\n",
        "# ---- بخش 3: بارگذاری وزن مدل و داده‌های مرجع ----\n",
        "input_dim = 1024  # خروجی ProtBERT\n",
        "model = build_siamese_model(input_dim)\n",
        "model.load_weights('/content/drive/MyDrive/Osta/weights_only.weights.h5')\n",
        "\n",
        "# فرض بر این است که این فایل‌ها در کنار app.py قرار دارند\n",
        "x0_train = X0\n",
        "x1_train = X1\n",
        "\n",
        "# ---- بخش 4: توابع پردازش ورودی ----\n",
        "def RealTest(X, x0_train, x1_train):\n",
        "    dataset_test = []\n",
        "    Indexsample = []\n",
        "    for i in range(len(X)):\n",
        "        for j in range(len(x0_train)):\n",
        "            dataset_test.append(np.concatenate((X[i], x0_train[j])))\n",
        "            Indexsample.append(f\"{i},0\")\n",
        "        for j in range(len(x1_train)):\n",
        "            dataset_test.append(np.concatenate((X[i], x1_train[j])))\n",
        "            Indexsample.append(f\"{i},1\")\n",
        "    return np.asarray(dataset_test), Indexsample\n",
        "\n",
        "def RealPredict(X, test_pred, Indexsample):\n",
        "    Pred = []\n",
        "    for i in range(len(X)):\n",
        "        cnt0Pred, cnt1Pred, cnt0, cnt1 = 0, 0, 0, 0\n",
        "        for j in range(len(Indexsample)):\n",
        "            triplet = Indexsample[j].split(\",\")\n",
        "            if int(triplet[0]) == i:\n",
        "                if int(triplet[1]) == 1:\n",
        "                    cnt1Pred += test_pred[j]\n",
        "                    cnt1 += 1\n",
        "                else:\n",
        "                    cnt0Pred += test_pred[j]\n",
        "                    cnt0 += 1\n",
        "        pos = (1 - (cnt0Pred / cnt0)) + (cnt1Pred / cnt1)\n",
        "        neg = (cnt0Pred / cnt0) + (1 - (cnt1Pred / cnt1))\n",
        "        Pred.append(float(pos / (pos + neg)))\n",
        "    return Pred\n",
        "\n",
        "# ---- بخش 5: API Endpoint ----\n",
        "@app.route(\"/predict\", methods=[\"POST\"])\n",
        "def predict():\n",
        "    try:\n",
        "        data = request.get_json()\n",
        "        peptide_seq = data[\"peptide_sequence\"]  # رشته پپتید\n",
        "\n",
        "        # مرحله 1: تبدیل به وکتور\n",
        "        peptide_vector = embed_peptide_with_protbert(peptide_seq).reshape(1, -1)\n",
        "\n",
        "        # مرحله 2: آماده‌سازی داده برای Siamese\n",
        "        main_test, index_sample = RealTest(peptide_vector, x0_train, x1_train)\n",
        "        left = main_test[:, :input_dim]\n",
        "        right = main_test[:, input_dim:]\n",
        "\n",
        "        # مرحله 3: پیش‌بینی\n",
        "        y_pred = model.predict([left, right])\n",
        "        final_score = RealPredict(peptide_vector, y_pred, index_sample)\n",
        "\n",
        "        return jsonify({\n",
        "            \"osteogenic_score\": final_score[0]\n",
        "        })\n",
        "\n",
        "    except Exception as e:\n",
        "        return jsonify({\"error\": str(e)}), 500\n",
        "\n",
        "# ---- اجرای سرور ----\n",
        "if __name__ == \"__main__\":\n",
        "    app.run(host=\"0.0.0.0\", port=5000, debug=True)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-9SlPTUtGfkJ",
        "outputId": "91c9d5d4-8fb8-4af2-f175-91037e40307c"
      },
      "execution_count": null,
      "outputs": [
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "(81, 1024)\n",
            "(81, 1024)\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.11/dist-packages/keras/src/layers/core/dense.py:87: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stdout",
          "output_type": "stream",
          "text": [
            " * Serving Flask app '__main__'\n",
            " * Debug mode: on\n"
          ]
        },
        {
          "metadata": {
            "tags": null
          },
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "INFO:werkzeug:\u001b[31m\u001b[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\u001b[0m\n",
            " * Running on all addresses (0.0.0.0)\n",
            " * Running on http://127.0.0.1:5000\n",
            " * Running on http://172.28.0.12:5000\n",
            "INFO:werkzeug:\u001b[33mPress CTRL+C to quit\u001b[0m\n",
            "INFO:werkzeug: * Restarting with stat\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "idF_km0YrWxv"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}
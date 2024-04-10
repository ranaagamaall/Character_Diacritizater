from nltk.tokenize import sent_tokenize
import nltk
import re
import pyarabic.araby as araby
import torch
import torch.nn as nn
from torch import tensor
import numpy as np
from tqdm import tqdm
import pandas as pd
import itertools
from transformers import AutoTokenizer, AutoModel
import pickle
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

nltk.download('punkt')


class Dataset(torch.utils.data.Dataset):

    def __init__(self, x, y, pad=None):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def load_data(data, labels):
    data = data.explode().to_list()
    labels = labels.explode().to_list()
    train_data = zip(data, labels)

    cleaned_data = []
    for word in train_data:
        try:
            if isinstance(word[0], list):
                cleaned_data.append(word)
        except:
            pass
    data, labels = zip(*cleaned_data)

    max_word_len = max(len(word[0]) for word in data)
    X = torch.zeros((len(data), max_word_len, 36 + 15 + 2),
                    dtype=torch.float32).cpu()
    y = torch.zeros((len(data), max_word_len), dtype=torch.int64).cpu()
    y = torch.fill(y, 15)

    for i, word in enumerate(data):
        for j, char in enumerate(word):
            try:
                X[i, j, tensor(char, dtype=torch.long)] = 1
                y[i, j] = labels[i][j]
            except Exception as e:
                y[i, j] = 14

    train_data = Dataset(X, y)

    return train_data


class Model(nn.Module):
    def __init__(self, input_size=768, hidden_size=50, n_classes=100, bidirectional=False):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True, bidirectional=bidirectional)

        self.linear = nn.Linear(
            hidden_size * (2 if bidirectional else 1), n_classes)

    def forward(self, X, hidden=None):

        final_output, hidden = self.lstm(X, hidden)
        final_output = self.linear(final_output)
        return final_output, hidden


def train(model, train_dataset, n_classes, batch_size=512, epochs=5, learning_rate=0.01):
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_input = train_input.to(device)

            train_label = train_label.to(device)

            output, _ = model(train_input)
            output = output.to(device)

            batch_loss = criterion(
                output.view(-1, n_classes), train_label.view(-1))

            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=-1) == train_label).sum().item()
            total_acc_train += acc

            optimizer.zero_grad()

            batch_loss.backward(retain_graph=True)

            optimizer.step()

        epoch_loss = total_loss_train / len(train_dataset)

        epoch_acc = total_acc_train / \
            (len(train_dataset) * train_dataset[0][0].shape[0])

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss} \
        | Train Accuracy: {epoch_acc}\n')


with open("./dataset/train.txt", "r", encoding="utf-8") as f:
    val = f.readlines()


def clean_text(text):
    text = re.sub(r"[]{}[:()'\"]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"،", ",", text)
    text = re.sub(r"؟", "?", text)
    text = re.sub(r"؛", ";", text)
    return text


val = " ".join([clean_text(text) for text in val])
val = sent_tokenize(val)
sentences = []
for sent in val:
    sentences.extend(araby.sentence_tokenize(sent))
df = pd.DataFrame(sentences)
print(df.shape)


# import unicodedata


def condition(word):
    if len(word) == 1:
        if word == "و":
            return True
        return False
    return araby.is_arabicrange(word)


diacritic_to_id = pickle.load(open("./assets/diacritic2id.pickle", "rb"))
arabic_letters = list(pickle.load(
    open("./assets/arabic_letters.pickle", "rb")))


def match_diacritics(text, idx):
    for idx, key in enumerate(diacritic_to_id.keys()):
        match = True
        for i in range(len(key)):
            if idx + i >= len(text) - 1:
                match = False
                break
            if text[idx + i] != key[i]:
                match = False

        if match:
            return idx

    return diacritic_to_id[""]
# [(' َ', 0),
#  (' ً', 1),
#  (' ُ', 2),
#  (' ٌ', 3),
#  (' ِ', 4),
#  (' ٍ', 5),
#  (' ْ', 6),
#  (' ّ', 7),
#  (' َّ', 8),
#  (' ًّ', 9),
#  (' ُّ', 10),
#  (' ٌّ', 11),
#  (' ِّ', 12),
#  (' ٍّ', 13),
#  (' ', 14)]


def extract_diacritics(text):
    diacritics_list = []
    for word in text:
        word_list = []
        for idx, char in enumerate(word):
            if char in diacritic_to_id:
                continue
            if char not in arabic_letters:
                continue
            # if idx == len(word) - 1:
            #     word_list.append(diacritic_to_id[""])
            #     continue

            # word_list.append(match_diacritics(word, idx+1))
            # ana 7arf

            if idx + 2 >= len(word):  # last char
                if idx == len(word) - 1:
                    word_list.append(diacritic_to_id[""])
                    break
                if word[idx+1] in diacritic_to_id:
                    word_list.append(diacritic_to_id[word[idx+1]])
                    break
                else:
                    word_list.append(diacritic_to_id[""])
                    continue

            if word[idx+1] in diacritic_to_id and word[idx+2] in diacritic_to_id:
                if diacritic_to_id[word[idx+1]] == 7:
                    word_list.append(diacritic_to_id[word[idx+2]]+8)
                else:
                    word_list.append(diacritic_to_id[""])
            elif word[idx+1] in diacritic_to_id and word[idx+2] not in diacritic_to_id:
                word_list.append(diacritic_to_id[word[idx+1]])
            else:
                word_list.append(diacritic_to_id[""])
        diacritics_list.append(word_list)

    return diacritics_list


df["tokenized"] = df[0].apply(
    lambda sent: araby.tokenize(sent, conditions=condition))
df["tokenized_cleaned"] = df[0].apply(lambda sent: araby.tokenize(
    sent, conditions=condition, morphs=araby.strip_tashkeel))
df["diacritics"] = df["tokenized"].apply(extract_diacritics)


def diacritics_probability_per_char(tokenized_cleaned, diacritics):
    probability = {}

    for i in range(36):
        probability[i] = []
        for j in range(15):
            probability[i].append(0)

    counts = [0]*36

    for i in range(len(tokenized_cleaned)):
        for j in range(len(tokenized_cleaned[i])):
            for k in range(len(tokenized_cleaned[i][j])):
                probability[arabic_letters.index(
                    tokenized_cleaned[i][j][k])][diacritics[i][j][k]] += 1
                counts[arabic_letters.index(tokenized_cleaned[i][j][k])] += 1

    for i in range(36):
        for j in range(15):
            probability[i][j] /= counts[i]

    return probability


def create_bef_after(text):
    input_vectors = []
    if len(text) == 1:
        letter_vec = []
        letter_vec.append(36)
        letter_vec.append(36)
        input_vectors.append(letter_vec)
        return input_vectors
    for i in range(len(text)):
        letter_vec = []
        if i == 0:
            letter_vec.append(36)
            letter_vec.append(arabic_letters.index(text[i+1]))
            input_vectors.append(letter_vec)
            continue
        elif i == len(text)-1:
            letter_vec.append(arabic_letters.index(text[i-1]))

            letter_vec.append(36)
            input_vectors.append(letter_vec)
            break
        letter_vec.append(arabic_letters.index(text[i-1]))

        letter_vec.append(arabic_letters.index(text[i+1]))
        input_vectors.append(letter_vec)

    return input_vectors


prob_per_char = diacritics_probability_per_char(
    df["tokenized_cleaned"], df["diacritics"])
df["prob_per_char"] = df["tokenized_cleaned"].apply(
    lambda x: [[prob_per_char[list(arabic_letters).index(char)] for char in word] for word in x])

df["bef_after"] = df["tokenized_cleaned"].apply(
    lambda x: [create_bef_after(word) for word in x])

print("Concatinating features..")


def concat_features(tokenized_cleaned, prob, bef_after):
    features = []
    for idx, word in enumerate(tokenized_cleaned):
        word_features = []
        for letter_idx, letter in enumerate(word):
            letter_id = arabic_letters.index(letter)
            feature = np.zeros(36)
            feature[letter_id] = 1
            feature = list(feature)
            feature.extend(prob[idx][letter_idx])
            feature.extend(bef_after[idx][letter_idx])
            word_features.append(feature)
        features.append(word_features)
    return features


df["features"] = df.apply(lambda x: concat_features(
    x["tokenized_cleaned"], x["prob_per_char"], x["bef_after"]), axis=1)


labels = df["diacritics"]
data = df["features"]

print("Creating Dataset...")
train_data = load_data(data, labels)

tashkeel_model = Model(
    input_size=36+15+2, hidden_size=512, n_classes=16).to(device)
train(tashkeel_model, train_data, n_classes=16,
      batch_size=200, epochs=5, learning_rate=0.05)

pickle.dump(tashkeel_model, open("./models/ely_mayetsamma.pkl", "wb"))

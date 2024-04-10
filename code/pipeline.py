from dataset import Letter_Train_Dataset
from model import Letter_Model, Model as Context_Model
from utils import load_train_data, train_letter_model
import torch
from torch import tensor
import pandas as pd


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


if __name__ == '__main__':
    # Load data
    df = pd.read_csv("./processed/train.csv")

    labels = df['diacritics'].apply(eval)
    data = df["word_with_context"].apply(lambda x: eval(x))

    print("Data loaded")

    train_data = load_train_data(data, labels)

    print("Data processed")

    # Train model

    tashkeel_model = Letter_Model(
        input_size=36, hidden_size=128, n_classes=16).to(device)

    train_letter_model(tashkeel_model, train_data,
                       n_classes=16, epochs=10, learning_rate=0.01)

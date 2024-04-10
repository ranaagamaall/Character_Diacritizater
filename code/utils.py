import pickle
import torch
from torch import tensor
from dataset import Letter_Train_Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

context_model = pickle.load(
    open("../models/context_model.pkl", "rb")).to(device)
context_model.lstm.flatten_parameters()
tokenizer = AutoTokenizer.from_pretrained(
    'CAMeL-Lab/bert-base-arabic-camelbert-ca')
embedder = AutoModel.from_pretrained(
    'CAMeL-Lab/bert-base-arabic-camelbert-ca').to(device)


def get_contextualized_embeddings(sent):
    if len(sent) == 0:
        return []
    tokens = tokenizer(sent, return_tensors="pt", padding=True)
    tokens = tokens.to(device)
    embeddings = embedder(**tokens).last_hidden_state[:, 1, :]
    hidden_layers = []
    with torch.no_grad():
        hidden = None
        for embedding in embeddings:
            _, hidden = context_model(embedding.unsqueeze(0), hidden)
            hidden_layers.append(hidden[0])

    return list(zip(sent, hidden_layers))


def load_train_data(data, labels):
    arabic_letters = list(pickle.load(
        open("../assets/arabic_letters.pickle", "rb")))

    data = data.apply(lambda sent: [([arabic_letters.index(
        letter) for letter in word[0]], word[1]) for word in sent])

    data = data.explode().to_list()
    labels = labels.explode().to_list()
    train_data = zip(data, labels)

    cleaned_data = []
    for word in train_data:
        try:
            if isinstance(word[0][0], list):
                cleaned_data.append(word)
        except:
            pass

    data, labels = zip(*cleaned_data)

    data = [(list(word[0]), word[1]) for word in data]
    labels = [list(diacritic) for diacritic in labels]

    max_word_len = max(len(word[0]) for word in data)
    X = torch.zeros((len(data), max_word_len, len(
        arabic_letters)), dtype=torch.float32).cpu()
    y = torch.full((len(data), max_word_len), 15, dtype=torch.int64).cpu()
    for i, word in enumerate(data):
        for j, char in enumerate(word[0]):
            try:
                X[i, j, tensor(char, dtype=torch.long)] = 1
                y[i, j] = labels[i][j]
            except Exception as e:
                y[i, j] = 14
                print(char)
                # print(labels[i])
                # pass
    E = torch.cat([word[1] for word in data])

    train_data = Letter_Train_Dataset(X, E, y)

    return train_data


def train_letter_model(model, train_dataset, n_classes, batch_size=1, epochs=5, learning_rate=0.01):
    """
    This function implements the training logic
    Inputs:
    - model: the model ot be trained
    - train_dataset: the training set of type NERDataset
    - n_classes: integer represents the number of classes (tags)
    - batch_size: integer represents the number of examples per step
    - epochs: integer represents the total number of epochs (full training pass)
    - learning_rate: the learning rate to be used by the optimizer
    """
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

        for train_input, hidden, train_label in tqdm(train_dataloader):

            train_input = train_input.to(device)

            hidden = hidden.to(device)

            train_label = train_label.to(device)

            output, _ = model(train_input, (hidden.unsqueeze(0),
                                            torch.zeros_like(hidden.unsqueeze(0))))
            output = output.to(device)

            batch_loss = criterion(
                output.view(-1, n_classes), train_label.view(-1))

            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=-1) == train_label).sum().item()
            total_acc_train += acc

            optimizer.zero_grad()

            batch_loss.backward(retain_graph=True)

            optimizer.step()

        # epoch loss
        epoch_loss = total_loss_train / len(train_dataset)

        # calculate the accuracy
        epoch_acc = total_acc_train / \
            (len(train_dataset) * train_dataset[0][0].shape[0])

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {epoch_loss} \
        | Train Accuracy: {epoch_acc}\n')


def eval_letter_model(model, train_dataset, batch_size=512):

    test_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    out_preds = []

    total_acc_test = 0

    with torch.no_grad():
        for test_input, hidden, test_label in tqdm(test_dataloader):

            test_label = test_label.to(device)

            hidden = hidden.to(device)

            test_input = test_input.to(device)

            output, _ = model(test_input, (hidden.unsqueeze(0),
                                           torch.zeros_like(hidden.unsqueeze(0))))

            outputs = output.argmax(dim=-1)
            index = test_input
            for i in range(len(test_input)):
                if test_input[i].sum() > 0:
                    out_preds.append(outputs[i])

            acc = (output.argmax(dim=-1) == test_label).sum().item()
            total_acc_test += acc

    print(
        f'\nTest Accuracy: {total_acc_test/(len(train_dataset) * train_dataset[0][0].shape[0])}')


def pred_letter_model(model, eval_dataset):

    test_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1)

    out_preds = []
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        model = model.cuda()

    with torch.no_grad():
        for test_input, hidden, test_label in tqdm(test_dataloader):

            test_label = test_label.to(device)

            hidden = hidden.to(device)

            test_input = test_input.to(device)

            output, _ = model(test_input, (hidden.unsqueeze(0),
                                           torch.zeros_like(hidden.unsqueeze(0))))

            outputs = output.argmax(dim=-1)

            for i in range(len(test_input)):
                if test_input[i].sum() > 0:
                    out_preds.append(outputs[i])

    return out_preds

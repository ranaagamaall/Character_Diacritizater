import pickle
from nltk.tokenize import sent_tokenize
import pandas as pd
import pyarabic.araby as araby
import re
import nltk
nltk.download('punkt')

diacritic_to_id = pickle.load(open("./assets/diacritic2id.pickle", "rb"))
arabic_letters = pickle.load(open("./assets/arabic_letters.pickle", "rb"))


def condition(word):
    if len(word) == 1:
        if word == "و":
            return True
        return False
    return araby.is_arabicrange(word)


def extract_diacritics(text):
    diacritics_list = []
    for word in text:
        word_list = []
        for idx, char in enumerate(word):
            if char in diacritic_to_id:
                continue
            if char not in arabic_letters:
                continue

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
                if word[idx+1] == 7:
                    word_list.append(diacritic_to_id[word[idx+2]]+8)
                else:
                    word_list.append(diacritic_to_id[""])
            elif word[idx+1] in diacritic_to_id and word[idx+2] not in diacritic_to_id:
                word_list.append(diacritic_to_id[word[idx+1]])
            else:
                word_list.append(diacritic_to_id[""])
        diacritics_list.append(word_list)

    return diacritics_list


def clean_text(text):
    text = re.sub(r"[]{}[:()'\"]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"،", ",", text)
    text = re.sub(r"؟", "?", text)
    text = re.sub(r"؛", ";", text)
    return text


if __name__ == "__main__":

    with open("./dataset/train.txt", "r", encoding="utf-8") as f:
        val = f.readlines()

    val = " ".join([clean_text(text) for text in val])
    val = sent_tokenize(val)
    sentences = []
    for sent in val:
        sentences.extend(araby.sentence_tokenize(sent))
    df = pd.DataFrame(sentences)
    print(df.shape)

    df["tokenized"] = df[0].apply(
        lambda sent: araby.tokenize(sent, conditions=condition))
    df["tokenized_cleaned"] = df[0].apply(lambda sent: araby.tokenize(
        sent, conditions=condition, morphs=araby.strip_tashkeel))
    df["cleaned"] = df["tokenized"].apply(" ".join)
    df["diacritics"] = df["tokenized"].apply(extract_diacritics)

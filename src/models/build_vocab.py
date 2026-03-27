import pandas as pd
from collections import Counter
import json
import re


class Vocabulary:
    def __init__(self, freq_threshold=1, max_size=5000):
        self.freq_threshold = freq_threshold
        self.max_size = max_size

        self.itos = {}  # index -> word
        self.stoi = {}  # word -> index

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        """
        tokenize đơn giản, lowercase + remove ký tự lạ
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        return text.split()

    def build_vocabulary(self, sentence_list):
        """
        sentence_list: list[str]
        """

        frequencies = Counter()

        # đếm tần suất từ
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        # lọc theo threshold
        freq_words = [
            word for word, freq in frequencies.items()
            if freq >= self.freq_threshold
        ]

        # sort theo tần suất
        freq_words = sorted(freq_words, key=lambda x: frequencies[x], reverse=True)

        # cắt max vocab
        freq_words = freq_words[:self.max_size]

        # special tokens
        self.itos = {
            0: "<PAD>",
            1: "<SOS>",
            2: "<EOS>",
            3: "<UNK>"
        }

        idx = 4

        for word in freq_words:
            self.stoi[word] = idx
            self.itos[idx] = word
            idx += 1

        # thêm special tokens vào stoi
        for k, v in self.itos.items():
            if v not in self.stoi:
                self.stoi[v] = k

    def numericalize(self, text):
        """
        text → list[int]
        """
        tokens = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokens
        ]


def build_vocab_from_csv(csv_path, save_path):
    """
    csv phải có cột 'caption'
    """

    df = pd.read_csv(csv_path)

    captions = df["caption"].tolist()

    vocab = Vocabulary(freq_threshold=3, max_size=5000)
    vocab.build_vocabulary(captions)

    # save
    with open(save_path, "w") as f:
        json.dump({
            "stoi": vocab.stoi,
            "itos": vocab.itos
        }, f)

    print(f"Vocab size: {len(vocab)}")
    print(f"Saved to: {save_path}")

import os
basedir  = os.path.dirname(os.path.dirname(__file__))
if __name__ == "__main__":
    build_vocab_from_csv(
        csv_path= r"D:\Paper\Traffic_Risk_Project\data\data\processed_train.csv",
        save_path=r"D:\Paper\Traffic_Risk_Project\data\data\vocab.json"
    )
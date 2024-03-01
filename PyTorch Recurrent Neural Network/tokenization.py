import nltk
import numpy as np


class Word_Preprocess:
    def __init__(self, data) -> None:
        self.vocabs = self.vocabulary(data)
    
    def remove_stop_words(self, data):
        stop_words = nltk.stopwords.words('english')

        removed_stop_words_list = []
    
        for i, _ in enumerate(data):
            removed_stop_words_list.append([word for word in data[i].replace("\n", "").split(" ") if word not in stop_words])

        return removed_stop_words_list
                
    def bigrams(self, data):
        bigram_list = []

        for _, word in enumerate(data):
            for j in range(len(word) - 1):
                bigram_list.append(word[j : j+2])
        
        return bigram_list

    def vocabulary(self, bigram_data):
        vocab_list = []
        for bigram in bigram_data:
            vocab_list.extend(bigram)

        return list(set(vocab_list))

    def one_hot_encoder(self, bigram_data):
        one_hot_values = {}
        bigram_one_hot_list = []
        for i, key in enumerate(self.vocab):
            one_hot_values[key] = [0 if i != j else 1 for j in range(len(self.vocab))]

        print("One Hot Encoder\n------------\n")
        print(np.array([f"{key}: {value}" for key, value in one_hot_values.items()]))
        for (X,y) in enumerate(bigram_data):
            bigram_one_hot_list.append([one_hot_values[X], one_hot_values[y]])

        return np.array(bigram_one_hot_list)
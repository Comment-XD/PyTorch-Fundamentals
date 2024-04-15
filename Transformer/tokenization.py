import re
import nltk
import numpy as np

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    mag_vec1 = np.sqrt(sum(vec1 ** 2))
    mag_vec2 = np.sqrt(sum(vec2 ** 2))
		
    return dot_product / (mag_vec1 * mag_vec2)


class Word_Preprocess:
    
    def __init__(self) -> None:
	    self.stop_words = nltk.corpus.stopwords.words('english')
	
    def remove_tags(self, text):
		
        """Removes HTML tags: replaces anything between opening and closing <> with empty space"""
        
        
        TAG_RE = re.compile(r'<[^>]+>')
        return TAG_RE.sub('', text)
	
    def text_preprocess(self, sen):
	    
        sen = sen.lower()
	        
	    # Remove html tags
        sentence = self.remove_tags(sen)
	
	    # Remove punctuations and numbers
        sentence = re.sub('[^a-zA-Z]', ' ', sentence)
	        
	    # Single character removal
        sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, 
	
	    # Remove multiple spaces
        sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text.
	        
	    # Remove Stopwords
        pattern = re.compile(r'\b(' + r'|'.join(self.stop_words) + r')\b\s*')
        sentence = pattern.sub('', sentence)
	
        return sentence
	
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
	
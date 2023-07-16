import pickle
import numpy as np
import n_gram_config
import config

class FeatureExtractor:

    def __init__(self):
        self.n = n_gram_config.N
        self.sentences = self.__load_processed_training_data__()

        if n_gram_config.PAD_SENTENCES:
            self.sentences = self.__pad_sentences__()
    def __load_processed_training_data__(self):
        return pickle.load(open(config.PROCESSED_TRAIN_DATA_PATH, 'rb'))
    def __pad_sentences__(self):
        return list(map(lambda x: ['<sos>' * (self.n - 1)] + x + ['<eos>' * (self.n - 1)], self.sentences))
    def __extract_n_grams__(self):
        self.n_grams = []
        average_sentence_length = 0
        for sentence in self.sentences:
            sentence_ngrams = []
            average_sentence_length += len(sentence)
            for i in range(len(sentence) - self.n):
                sentence_ngrams.append(np.array([sentence[i] for i in range(i + self.n)]))
            self.n_grams.append(np.array(sentence_ngrams))



extractor = FeatureExtractor()
print(extractor.sentences[0])



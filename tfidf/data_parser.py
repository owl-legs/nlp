import pickle
import numpy as np
import config
import re

class DataParser():
    def __init__(self):

        self.tokens = {}

        self.__load_data__()
        self.__clean_data__()
    def __load_data__(self):
        self.data = []
        with open(config.RAW_DATA_PATH, "r", encoding='UTF-8') as file:
            self.data = file.read()
    def __update_token_counts__(self, tokens):
        for token in tokens:
            if token in self.tokens:
                self.tokens[token] += 1
            else:
                self.tokens[token] = 1
    def __extract_tokens_and_sentences__(self):
        if self.sentences:
            for i, sentence in enumerate(self.sentences):
                tokens = re.findall("[a-zA-Z'-]+", sentence)
                tokens = list(map(lambda x: x.lower(), tokens))
                tokens = ['<bos>'] + ["".join(t.split("'")) for t in tokens] + ['<eos>']
                self.__update_token_counts__(tokens)
                self.sentences[i] = tokens
        return
    def __write_token_dic__(self):
        pickle.dump(self.tokens, open(config.TOKEN_DIC_OUTPUT, "wb"), True)
    def __write_tokenized_sentences__(self):
        pickle.dump(self.sentences, open(config.TOKENIZED_SENTENCE_OUTPUT, "wb"), True)

    def __clean_data__(self):
        start = self.data.find('One morning, when Gregor Samsa woke from troubled dreams,')
        end = self.data.find('first to get up and stretch out her young body.')
        self.data = self.data[start:end+1]

        self.sentences = self.data.split(".")
        self.__extract_tokens_and_sentences__()
        self.__write_tokenized_sentences__()
        self.__write_token_dic__()

    def write_out_corpus_stats(self):
        corpus_config = {}

        if self.sentences:
            corpus_config['avg_sentence_length'] = sum(list(map(len, self.sentences)))/len(self.sentences)
        if self.tokens:
            corpus_config['total_tokens'] = len(self.tokens)
            corpus_config['average_token_len'] = sum(list(map(len, self.tokens)))/len(self.tokens)

        pickle.dump(corpus_config, open('corpus_config', "wb"), True)

    def print_text(self):
        print(pickle.load(open('corpus_config', "rb")))

dp = DataParser()
dp.write_out_corpus_stats()
dp.print_text()
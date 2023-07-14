import config
import pickle
from gensim.models import word2vec

class Embedding:
    def __init__(self, embedding_size=300):
        self.corpus_config = pickle.load(open('./corpus_config', 'rb'))
        self.tokens = pickle.load(open(config.TOKEN_DIC_OUTPUT, 'rb'))
        self.sentences = pickle.load(open(config.TOKENIZED_SENTENCE_OUTPUT, 'rb'))
        self.average_word_size = self.corpus_config.get('average_token_len', 7)
        self.vocab_size = len(self.tokens)
        self.embedding_size = embedding_size

    def __most_common_tokens__(self):
        maxThresh = int(len(self.tokens)//4)
        for token in self.tokens:
            if self.tokens[token] < maxThresh:
                del self.tokens[token]
    def create_embeddings(self):
        self.__create_one_hot_embeddings__()
        self.__create_word2vec_embeddings__()
    def embed_text(self, text):
        if type(text) == str:
            text = [text]
        embeddedText = []
        return embeddedText
    def __create_one_hot_embeddings__(self):
        print('\n creating one-hot embedding map')
        embeddingMap = {t:i for i, t in enumerate(self.tokens.keys())}
        embeddedSentences = []
        for sentence in self.sentences:
            embedding = list(map(lambda x: embeddingMap.get(x, '<unk>'), sentence))
            embeddedSentences += [embedding]
        pickle.dump(embeddingMap, open('embeddings/maps/oneHotEmbeddingMap', 'wb'), True)
        pickle.dump(embeddedSentences, open('embeddings/embeddedSentences', 'wb'), True)
    def __create_word2vec_embeddings__(self):
        print('\n creating word2vec embedding map')
        word_vector = word2vec.Word2Vec(sentences=self.sentences, vector_size=self.embedding_size)
        word_vector.wv.save_word2vec_format('embeddings/maps/word2vecEmbeddingMap.txt', binary=False)
        word_vector.wv.save_word2vec_format('embeddings/maps/word2vecEmbeddingMap.bin', binary=True)
        self.wv = word_vector.wv

embedding = Embedding()
embedding.create_embeddings()
import collections
import config
import pickle
from gensim.models import word2vec, KeyedVectors
import numpy as np

class FeatureExtractor:
    def __init__(self):

        self.min_frequency = 5
        self.max_frequency = 10000

        self.stopwords = self.__load_stop_words__()
        self.__load_corpus__()
        self.trainingWordDic = self.__count_tokens__(pickle.load(open(config.PROCESSED_TRAIN_DATA_PATH, "rb")))
    def __load_corpus__(self):
        print("\n loading corpus")
        self.wordDic = open(config.CORPUS_PATH, 'r').read()
        self.wordDic = self.wordDic.replace('--', " ")
        self.wordDic = collections.Counter(self.wordDic.split(" "))
    def extract_features(self,
                         embedding_type,
                         training=True):
        self.__test_sentence_stats__()

        if training:
            self.__create_embeddings__(output_path=config.EMBEDDED_TRAIN_PATH, option=embedding_type, train=training)
        else:
            self.__create_embeddings__(output_path=config.EMBEDDED_TEST_PATH, option=embedding_type, train=training)
    def __load_stop_words__(self):
        print("\n loading stop words")
        return set(open(config.STOP_WORDS_PATH, "r").read().split(" "))

    def __count_tokens__(self, sentences):
        tokenCounts = collections.Counter()
        for sentence in sentences:
            for token in sentence:
                tokenCounts[token] += 1
        return tokenCounts

    def __filter_words__(self):
        self.filteredWords = {}
        for word in self.wordDic.keys():
            if self.wordDic[word] >= self.min_frequency:
                self.filteredWords[word] = self.wordDic[word]

    def __test_sentence_stats__(self):

        print("\n calculating corpus statistics")
        test_data = pickle.load(open(config.PROCESSED_TEST_DATA_PATH, 'rb'))
        n = len(test_data)
        textLen = 0
        minTextLen, maxTextLen = float('inf'), float('-inf')
        spaceIndex = 0

        for testPoint in test_data:
            textLen += len(testPoint['question'])
            minTextLen = min(minTextLen, len(testPoint['question']))
            maxTextLen = max(maxTextLen, len(testPoint['question']))
            spaceIndex += testPoint['question'].index("_____")

        self.averageTestSentenceLength = textLen//n
        self.minTestSentenceLength = minTextLen
        self.maxTestSentenceLength = maxTextLen
        self.averageSpaceIndex = spaceIndex//n

    def __build_word2vec_embeddings__(self):
        corpus = word2vec.Text8Corpus(config.CORPUS_PATH)
        word_vector = word2vec.Word2Vec(corpus, vector_size=config.EMBEDDING_SIZE)
        word_vector.wv.save_word2vec_format(config.WORD2VEC_MAP_PATH, binary=False)
        word_vector.wv.save_word2vec_format(config.WORD2VEC_BIN_PATH, binary=True)
    def __build_one_hot_mappings__(self):
        oneHotMap = {}
        self.__filter_words__()
        for i, word in enumerate(list(self.filteredWords.keys())):
            oneHotMap[word] = i
        oneHotMap['<unk>'] = len(self.filteredWords)
        pickle.dump(oneHotMap, open(config.ONE_HOT_MAP_PATH, 'wb'), True)

    def __embed_pre_context__(self, sentence, index, embeddingMap, option):
        if option == 'one-hot':
            return np.array([embeddingMap[word] if word in embeddingMap \
                                                   and word != "" \
                                 else embeddingMap['<unk>'] \
                             for word in sentence[:index]])
        elif option == 'word2vec':
            return np.array([embeddingMap[word] if word in embeddingMap \
                                            and word != "" \
                          else np.zeros(config.EMBEDDING_SIZE) \
                      for word in sentence[:index]])
    def __embed_mid_context__(self, sentence, midIndex, embeddingMap, option):
        if option == 'one-hot':
            if sentence[midIndex] in embeddingMap:
                return embeddingMap[sentence[midIndex]]
            else:
                return embeddingMap['<unk>']
        elif option == 'word2vec':
            if sentence[midIndex] in embeddingMap:
                return embeddingMap[sentence[midIndex]]
            else:
                return np.zeros(config.EMBEDDING_SIZE)

    def __embed_post_context__(self, sentence, index, embeddingMap, option):
        if option == 'one-hot':
            return np.array([embeddingMap[word] if word in embeddingMap \
                                                   and word != "" \
                                 else embeddingMap['<unk>'] \
                             for word in sentence[index+1:]])
        elif option == 'word2vec':
            return np.array([embeddingMap[word] if word in embeddingMap \
                                            and word != "" \
                          else np.zeros(config.EMBEDDING_SIZE) \
                      for word in sentence[index+1:]])

    def __create_embeddings__(self, output_path, option='one-hot', train=True):

        filePath = {'word2vec':config.WORD2VEC_MAP_PATH,
                    'skip-gram':"",
                    'one-hot':config.ONE_HOT_MAP_PATH}

        print("\n creating embedding maps")

        if option == 'word2vec':
            embeddingMap = KeyedVectors.load_word2vec_format(config.WORD2VEC_BIN_PATH,
                                                             binary=True)
        else:
            self.__build_one_hot_mappings__()
            embeddingMap = pickle.load(open(filePath[option], "rb"))

        embeddedX, embeddedY = [], []
        batchX, batchY = [], []
        counter = 0
        postfix = -1

        if train:
            print('\n embedding training data')
            sentences = pickle.load(open(config.PROCESSED_TRAIN_DATA_PATH, "rb"))
            self.training_data_size = len(sentences)
            print(f'''\n training data size: {self.training_data_size}''')
            for sentence in sentences:

                midIndex = int(len(sentence) // 2)

                preContextVector = self.__embed_pre_context__(sentence, midIndex, embeddingMap, option)
                preContextVector = np.sum(preContextVector, axis=0)

                postContextVector = self.__embed_post_context__(sentence, midIndex, embeddingMap, option)
                postContextVector = np.sum(postContextVector, axis=0)

                midContextVector = self.__embed_mid_context__(sentence, midIndex, embeddingMap, option)

                batchX.append([preContextVector, midContextVector])
                batchY.append(postContextVector)

                counter += 1

                if counter == (self.training_data_size // 8) - 1:
                    postfix += 1

                    print(f'''\n\nwriting train data, postfix is {postfix}''')
                    pickle.dump((np.array(batchX), np.array(batchY), len(batchX)), \
                                open(f'''data/embedded_train_data_{postfix}''', "wb"), True)

                    counter = 0
                    embeddedX += batchX
                    embeddedY += batchY

                    batchX, batchY = [], []

            pickle.dump((np.array(embeddedX), np.array(embeddedY), len(embeddedX)), \
                         open('data/embedded_train_full', 'wb'), True)

        else:
            sentences = pickle.load(open(config.PROCESSED_TEST_DATA_PATH, "rb"))
            for testCase in sentences:
                question = testCase['question']
                index = question.index("_____")
                preContextVector = self.__embed_pre_context__(question, index, embeddingMap, option)
                preContextVector = np.sum(preContextVector, axis=0)

                postContextVector = self.__embed_post_context__(question, index, embeddingMap, option)
                postContextVector = np.sum(postContextVector, axis=0)

                for option in {'a', 'b', 'c', 'd', 'e'}:

                    if testCase[option] in embeddingMap:
                        midContextVector = embeddingMap[testCase[option]]
                    else:
                        midContextVector = np.zeros(config.EMBEDDING_SIZE)

                    embeddedX.append([preContextVector, midContextVector])
                embeddedY.append([postContextVector])

            pickle.dump((np.array(embeddedX), np.array(embeddedY), len(embeddedX)), open(config.EMBEDDED_TEST_PATH, 'wb'), True)



featureExtractor = FeatureExtractor()
featureExtractor.extract_features(embedding_type='one-hot')

from data_parser import DataParser
import config
import pickle
import collections

dp = DataParser()
dp.write_out_corpus_stats()

tokenized_sentences = pickle.load(open(config.TOKENIZED_SENTENCE_OUTPUT, 'rb'))
n_documents = len(tokenized_sentences)

tf = collections.defaultdict(int)
df = collections.defaultdict(int)

def update_df(sentence):
    sentence_set = set(sentence)
    for word in sentence_set:
        df[word] += 1

for doc_id in tokenized_sentences:
    update_df(tokenized_sentences[doc_id])
    for word in tokenized_sentences[doc_id]:
        tf[(doc_id, word)] += 1

tfidf = {}
for doc_id, word in tf.keys():
    pass


from data_parser import DataParser
import config
import pickle
import collections
import math

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

for doc_id in range(n_documents):
    update_df(tokenized_sentences[doc_id])
    for word in tokenized_sentences[doc_id]:
        tf[(doc_id, word)] += 1

tfidf = {}
for doc_id, word in tf.keys():
    tfidf[(doc_id, word)] = tf[(doc_id, word)] * math.log2(n_documents / df[word])

tfidf_sentences = []
for doc_id in range(n_documents):
    tfidf_sentences.append(list(map(lambda x: tfidf[(doc_id, x)], tokenized_sentences[doc_id])))
pickle.dump(tfidf_sentences, open(config.TFIDF_EMBEDDED_OUTPUT, 'wb'), True)




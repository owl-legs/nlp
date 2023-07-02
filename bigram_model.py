def get_bigram(sentence):
  tokens = sentence.split(' ')
  n = len(tokens)
  bigrams = [tokens[i:i+2] for i in range(n-1)]
  bigrams = ['<bos>', tokens[0]] + bigrams + [tokens[n-1] + '<eos>']
  return bigrams

def get_bigram_counts(corpus):
  bigram_counts = {}
  for sentence in corpus:
    bigrams = get_birgrams(sentence)
    for bigram in bigrams:
      if bigram in bigram_counts:
        bigram_counts[bigram] += 1
      else:
        bigram_counts[bigram] = 1
  return bigram_counts

def get_bigram_probabilities(corpus):
  bigram_counts = get_bigram_counts(corpus)
  return bigram_counts =/ (sum(bigram_counts.values()))

def load_corpus(file_path):
  pass
  
                  


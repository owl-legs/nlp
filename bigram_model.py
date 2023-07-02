import json

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
  n = sum(bigram_counts.values())
  for bigram in bigram_counts:
    bigram_counts[bigram] /= n
  return bigram_counts

def get_optional_bigrams(lastword, bigram_probabilites):
  return [(bigram, p) for bigram, p in bigram_probabilities.items() if bigram[-1] == lastword]

def generate(model, seed = None):
  generatedText = '<bos>'
  
  if seed:
    generatedText += (' ' + seed)
    
  lastWord = '<bos>'
  while lastWord != '<eos':
    next_bigram = sorted(get_optional_bigrams(lastword, model))[0][1]
    generatedText += (' ' + next_bigram)
    lastWord = next_bigram
                          
  return generatedText

def load_corpus(file_path):
  pass
  
                  


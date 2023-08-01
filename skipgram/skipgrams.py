import pickle
import config
import collections
import tensorflow as tf
class Generator:
    def __init__(self):
        self.sequences = pickle.load(open('data/trainingData.txt', 'rb'))
        self.tokens = self.__generate_tokens__()
    def __filter__(self):
        pass
    def __generate_tokens__(self):
        with open('model/all_words.txt', 'rb') as file:
            all_words = file.readline()
            all_words = str(all_words)
            all_words = all_words.split(' ')
            all_words = list(set(all_words))
            return {w:i for i,w in enumerate(all_words)}

    def __tokenize_sequences__(self):

        tokenized_sequences = []
        for sequence in self.sequences:
            yield list(map(lambda x: self.tokens[x], sequence))

    def generate_training_data(self):

        window_size = 2

        vocab_size = len(self.tokens)
        targets, contexts, labels = [], [], []

        i = 1
        n = len(self.sequences)

        sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)
        num_ns = 4

        for tokenized_sequence in self.__tokenize_sequences__():

            print(f'''{(i/n)*100.00}% complete''')

            positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
                tokenized_sequence,
                vocabulary_size=vocab_size,
                sampling_table=sampling_table,
                window_size=window_size,
                negative_samples=0)

            for target_word, context_word in positive_skip_grams:

                context_class = tf.expand_dims(
                    tf.constant([context_word], dtype="int64"), 1)

                negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=vocab_size,
                    seed=55,
                    name="negative_sampling")

                context = tf.concat([tf.squeeze(context_class, 1), negative_sampling_candidates], 0)
                label = tf.constant([1] + [0] * num_ns, dtype="int64")

                targets.append(target_word)
                contexts.append(context)
                labels.append(label)

            i += 1

        return targets, contexts, labels


gen = Generator()
gen.generate_training_data()
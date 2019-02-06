import pickle

import numpy as np

WORDS = ['respiratory', 'vomiting', 'urine', 'pulse']

num_topic_to_models = pickle.load(open("num_topic_to_models.dict", "rb"))
model = num_topic_to_models[20]
id2word = model.id2word
word2id = {id2word[key]: key for key in id2word.keys()}

output_file = open("part4b.txt", "w")
for word in WORDS:
    output_file.write(word + "\n")
    word_id = word2id[word]
    temp = np.array(model.get_term_topics(word_id, minimum_probability = 0.0))
    topic_id = temp[temp[:, 1].argmax()][0]
    for term_id, prob in model.get_topic_terms(int(topic_id), topn=100):
        output_file.write(id2word[term_id] + " ")
    output_file.write("\n\n")
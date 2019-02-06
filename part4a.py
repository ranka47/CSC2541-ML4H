import pandas as pd

from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

from functions import *

import pickle

def compute_coherence_values(dictionary, corpus, limit, start=2, step=3, coherence_score = 'u_mass', iterations = 5):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : Max num of topics
    coherence_score : Type of Coherence Score to be used

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    for num_topics in range(start, limit, step):
        print(num_topics)
        model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, iterations=iterations)
        coherencemodel = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence=coherence_score)
        coherence_values.append(coherencemodel.get_coherence())

    return coherence_values

data = pd.read_csv('datasets/part2/adult_notes.gz', compression='gzip')
data.chartext = data.chartext.fillna('')
data = data['chartext']

dictionary = corpora.Dictionary()
corpus = []
num_topics = [20, 50, 100]
num_topic_to_topics = {}
num_topic_to_models = {}

print("Creating corpus and dictionary...")
for index in range(len(data)):
    tokens = tokenize(data[index], return_as_list=True, lowercase=True, regex=re.compile(r"[a-zA-Z0-9]+"))
    dictionary.add_documents([tokens])
    corpus.append(dictionary.doc2bow(tokens))
print("Corpus and dictionary created!")

for num_topic in num_topics:
    print("Number of topics: ", num_topic)
    ldamodel = LdaModel(corpus=corpus, num_topics = num_topic, id2word=dictionary, iterations=20)
    topics = ldamodel.print_topics(num_words=10)
    
    num_topic_to_topics[num_topic] = topics
    num_topic_to_models[num_topic] = ldamodel

    coherence_model_lda = CoherenceModel(model=ldamodel, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_lda = coherence_model_lda.get_coherence()
    print('Coherence Score (U_Mass): ', coherence_lda)

pickle.dump(num_topic_to_models, open("num_topic_to_models.dict", "wb"))

"""
Number of topics:  20
Coherence Score (U_Mass):  -0.42236587843071566

Number of topics:  50
Coherence Score (U_Mass):  -0.5597070191943092

Number of topics:  100
Coherence Score (U_Mass):  -0.6293096290358676
--------------------------------------------------------------------------
Number of iterations = 10
Number of topics:  20
Coherence Score (U_Mass):  -0.30862722788878283

Number of topics:  50
Coherence Score (U_Mass):  -0.33037421658476146

Number of topics:  100
Coherence Score (U_Mass):  -0.4703434539780698
--------------------------------------------------------------------------
Number of topics:  20
Coherence Score (U_Mass):  -0.23571937598224274

Number of topics:  50
Coherence Score (U_Mass):  -0.24170986787080043

Number of topics:  100
Coherence Score (U_Mass):  -0.32032350480440847
"""
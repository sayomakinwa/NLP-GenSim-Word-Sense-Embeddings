from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from typing import Tuple
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
from my_utils import *



def cos_similarity(word1: str, word2: str, w2v_model:KeyedVectors) -> float:
	"""
	:param word1; First of the words whose similarity is to be checked
    :param word2; Second of the words whose similarity is to be checked
    :param w2v_model; Model of the sense embeddings to check for associated senses
    :return score; A real scalar score value for the similarity between both words
	"""

	word1 = word1.lower()
	word2 = word2.lower()
	senses1 = []
	senses2 = []
	
	word1_words = []
	for synset in wn.synsets(word1):
		for lem in synset.lemma_names():
			word1_words.append(lem)

	word2_words = []
	for synset in wn.synsets(word2):
		for lem in synset.lemma_names():
			word2_words.append(lem)

	for sense in w2v_model.vocab.keys():
		sense_word = sense.split("_bn:")[0]

		if sense_word in word1_words:
			senses1.append(w2v_model[sense])

		if sense_word in word2_words:
			senses2.append(w2v_model[sense])

	score = 0.0
	for sense1 in senses1:
		for sense2 in senses2:
			score = max(score, cosine_similarity(sense1.reshape(1,-1), sense2.reshape(1,-1))[0,0])

	return score



def spearman(test_data_path: str, w2v_path: str) -> Tuple[float, ...]:
	"""
	:param test_data_path; Path to combined.tab data
    :param w2v_path; Model of the sense embeddings to check for associated senses
    :return [corr, p, data]; Corr is the correlation value, p is the p-value, and data is the data as structured on page 17 of the homework pdf
	"""
	model = KeyedVectors.load_word2vec_format(w2v_path, binary=False)

	data = []
	with open(test_data_path, 'r') as handle:
		for i, line in enumerate(handle):
			if 1 <= i:
				if (test_data_path.endswith(".csv")):
					line = line.strip().split(",")
				elif (test_data_path.endswith(".tab")):
					line = line.strip().split("\t")

				if (line):
					cosine = cos_similarity(line[0], line[1], model)
					line.append(cosine)
					data.append(line)

	corr, p = spearmanr([gold[2] for gold in data], [cos[3] for cos in data])
	return corr, p, data

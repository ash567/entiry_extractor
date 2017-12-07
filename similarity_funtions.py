# A whole new approach can can be
# construct the matrix (by taking the tensor product) and then learn the parameters for finding similarites of matrices. ie. use ANN or RNN to learn the similarity of these statements (Sanchit told that it is what is done in siamis net)



import numpy as np
import scipy as sp
import copy as cp

# Assumed things
# 1. Lines are the list of words and not string format [[word]]
# 2. Dictionaries are not empty



# TODO things
# 1. Use a threshold on the similarity measures.
# 4. Bug in ordered similarity (word_similarity)

def word_similarity(word1, word2):
	return model.similarity(word1, word2)
	# can implement your own similarity 
	# vec1 = word_to_vec(word1)
	# vec2 = word_to_vec(word2)
	# return vec_similarity(vec1, vec2)


def word_to_vec(word):
	vec = model[word]
	return vec


def vec_similarity(vec1, vec2):
	return sp.spatial.distance.cosine(vec1, vec2)


# returns the most similar token (in a dictionary) and the similarity score of the token to the word
# (can also try averaging out)
def most_similar_word(dictionary, word):
	if (dictionary == []):
			print "ERROR: Dictionary can not be empty"
	most_similar = dictionary[0]
	similarity_score = word_similarity(word, most_similar)
	avg_similarity = similarity_score
	for element in dictionary[1:]:
		similarity = word_similarity(word, element)
		avg_similarity += similarity
		if(similarity > similarity_score):
			similarity_score = similarity
			most_similar = element
	avg_similarity = avg_similarity/(len(dictionary) + 0.0)
	return((avg_similarity, most_similar))
	# return((similarity_score, most_similar))


# given a word returns the list of most similar word in each dictionary
def find_similar(list_dictionary, word):
	similar_list = []
	for dictionary in list_dictionary:
		similarity_score, most_similar = most_similar_word(dictionary, word)
		similar_list.append((similarity_score, most_similar))
	return similar_list


# for each dictionay finds out the most similar word in the line.
# One a word in line is matched, it can be removed from line
# ensure that line is not empty and all the words in line are in vocab
def line_similarity(list_dictionary, line):
	# line = []
	# for word in line1:
	# 	if word in model.vocab:
	# 		line.append(word)
	similarity_list = []
	for dictionary in list_dictionary:
		best_word = line[0]
		best_index = 0
		best_similarity_score, best_similar_word = most_similar_word(dictionary, best_word)
		for idx, word in enumerate(line[1:]):
			similarity_score, similar_word = most_similar_word(dictionary, word)
			if(similarity_score > best_similarity_score):
				best_similarity_score = similarity_score
				best_word = word
				best_similar_word = similar_word
				best_index = idx
		similarity_list.append((best_index, best_word, best_similar_word, best_similarity_score))
		# This is optional
		line.remove(best_word)
		# list of tokens returned will be
	token_list = [b for (a, b, c, d) in similarity_list]
	unordered_score = sum([d for (a, b, c, d) in similarity_list])
	# sort and find the index
	# index = np.asarray([idx for (a, b, c, d) in similarity_list])
	# order = np.arange(0, len(list_dictionary))
	# ordered_similarity = 1.0/np.linalg.norm((index - order), ord = 1)
	return(unordered_similarity, token_list)


# operator for combining the atomic similarity in dp_similarity
# for high number of dictionaries * may not a good operator
def operator(score1, score2):
	return score1 * score2


# objecive: say the input is (D1, D2, D3, D4,... Dm) and (W1, W2, W3 .... Wn)
# Find [(D1, Wi1,), (D2, Wi2) ..... (Dm, Wim)] such that i1 < i2 < i3 ...< im
# so as to maximize s(D1, Wi1,) op s(D2, Wi2) .... op s(Dm, Wim)
# properties The words matched with the dictionary follows the same order in the sentence as to the the corresponding order of the matched dictionaries and then for all such sequences it maximizes the similarity score as defined (operator can be changed)

# ensure that line is not empty and all the words in line are in vocab
def dp_similarity(list_dictionary, line):
	# line = []
	# for word in line1:
	# 	if word in model.vocab:
	# 		line.append(word)
	similarity_list = []
	for i, word in enumerate(line):
		tem_list = []
		for idx, dictionary in enumerate(list_dictionary):
			similarity_score, similar_word = most_similar_word(dictionary, word)
			tem_list.append((similarity_score, similar_word))
		similarity_list.append(tem_list)
	arr = np.empty((len(list_dictionary), len(line)))
	idx_arr = np.empty((len(list_dictionary), len(line)))
	arr.fill(0)
	idx_arr.fill(-1)
	arr[0][0] = similarity_list[0][0][0]
	idx_arr[0][0] = 1
	for i in xrange(1, len(line)):
		arr[0][i] = max(similarity_list[i][0][0], arr[0][i-1])
		idx_arr[0][i] = int (similarity_list[i][0][0] >= arr[0][i-1])
	for i in xrange(1, len(list_dictionary)):
		for j in xrange(i, len(line)):
			new_score = operator(arr[i-1][j-1],most_similar_word(list_dictionary[i], line[j])[0])
			arr[i][j] = max(new_score, arr[i][j-1])
			idx_arr[i][j] = int(new_score >= arr[i][j-1])
	i = len(list_dictionary) - 1
	j = len(line) - 1
	max_score = arr[i][j]
	word_list = []
	while(i >= 0 and j >= 0 and idx_arr[i][j] != -1):
		if(idx_arr[i][j]):
			word_list.append(line[j])
			i = i - 1
			j = j - 1
		else:
			j = j - 1
	word_list.reverse()
	return(max_score, word_list)


# -----------------------------
# Unused Code

# def get_item(a):
# 	return a[0]

# line is a list of words
# construct a vector for the line


# def construct_vec(list_dictionary, line):
# 	# filtering out all the words not in the dictionary
# 	line = [word for word in line if word in model.vocab]
# 	if(line ==[])
# 		print "No similar words"
# 	# for each dictionary finding the most similar word in the line

# 	most_similar_list = []

# 	for word in line:
# 		similar_list = find_similar(list_dictionary, word)
# 		most_similar = max(similar_list, key = lambda item:item[0])
# 		most_similar_list.append((word, most_similar[1]))
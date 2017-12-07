import numpy as np
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Word2Vec
from random import shuffle
import multiprocessing
import tensorflow as tf

# from tensorflow.python.ops import rnn, rnn_cell
# from tensorflow.python.ops.seq2seq import rnn_decoder

from tensorflow.models.rnn import rnn, rnn_cell
from tensorflow.models.rnn.seq2seq import rnn_decoder

import sys

# run python lstm11_c.py 1 MSRvid

# Lambda
# if len(sys.argv) < 2:
# 	print 'Lambda not specified.'
# 	exit(-1)
# else:
# 	lda = float(sys.argv[1])

lda = 1.0
print 'Lambda:', lda


# if len(sys.argv) < 3:
# 	print 'Test set not specified. Choosing default.'
# 	testDomain = 'SICK'
# else:
# 	domains = [
# 	'answers-forums',
# 	'answers-students',
# 	'belief',
# 	'deft-forum',
# 	'headlines',
# 	'MSRpar',
# 	'MSRvid',
# 	'OnWN',
# 	'SICK',
# 	'SMTeuroparl',
# 	'surprise.OnWN',
# 	'surprise.SMTnews',
# 	'tweet-news'
# 	]
# 	testDomain = ''
# 	for d in domains:
# 		if sys.argv[2].lower() in d.lower():
# 			testDomain = d

testDomain =  "MSRvid"

# if testDomain == '':
# 	print 'Test set specified, but not found.'
# 	print 'Please choose one of the following:'
# 	for d in domains:
# 		print d
# 	sys.exit(1)

print 'Test Set:', testDomain

pretrain = []
train = []
test = []

# TODO: change the path name
path = "/home/ishu/Documents/run_saved_model_lstm/"
resultPath = path + 'results/'
modelPath = path + 'models/'
lstm_model = 'model_1.ckpt'

file = open(path + 'sts.csv', 'r')

string = file.read()
ls = string.split('\n')
for l in ls[1:]:
	l = l.split('\t')
	if l[2] == '':
		continue
	tup = (list(utils.tokenize(l[3], lower = True, deacc = True)), list(utils.tokenize(l[4], lower = True, deacc = True)), float(l[2])*4.0/5.0+1)
	if l[1] == testDomain:
		test.append(tup)
	elif testDomain == 'SICK':
		pretrain.append(tup)
	else:
		train.append(tup)

file.close()


sick = np.genfromtxt(path + 'SICK.txt', delimiter = '\t', dtype = str, usecols = [1, 2, 4, -1])

sickTrain = sick[sick[:, -1] == 'TRAIN'][:, :-1]
sickValid = sick[sick[:, -1] == 'TRIAL'][:, :-1]
sickTest = sick[sick[:, -1] == 'TEST'][:, :-1]

sickTrain1 = [list(utils.tokenize(firstSentence, lower = True, deacc = True)) for firstSentence in sickTrain[:, 0]]
sickTrain2 = [list(utils.tokenize(secondSentence, lower = True, deacc = True)) for secondSentence in sickTrain[:, 1]]
sickTrainScore = sickTrain[:, 2]
sickTrain = [(s1, s2, float(sc)) for (s1, s2), sc in zip(zip(sickTrain1, sickTrain2), sickTrainScore)]

sickValid1 = [list(utils.tokenize(firstSentence, lower = True, deacc = True)) for firstSentence in sickValid[:, 0]]
sickValid2 = [list(utils.tokenize(secondSentence, lower = True, deacc = True)) for secondSentence in sickValid[:, 1]]
sickValidScore = sickValid[:, 2]
sickValid = [(s1, s2, float(sc)) for (s1, s2), sc in zip(zip(sickValid1, sickValid2), sickValidScore)]

sickTest1 = [list(utils.tokenize(firstSentence, lower = True, deacc = True)) for firstSentence in sickTest[:, 0]]
sickTest2 = [list(utils.tokenize(secondSentence, lower = True, deacc = True)) for secondSentence in sickTest[:, 1]]
sickTestScore = sickTest[:, 2]
sickTest = [(s1, s2, float(sc)) for (s1, s2), sc in zip(zip(sickTest1, sickTest2), sickTestScore)]


if testDomain == 'SICK':
	test = sickTest
	train = sickTrain + sickValid
else:
	train = train + sickTrain + sickValid

shuffle(train)
valid = train[:int(0.1*len(train))]
train = train[int(0.1*len(train)):]


# -----------------

# f = open(path + "test", "r")
# ap_data = f.read().split("\n")
# ap_data = [line.split() for line in ap_data]
# f.close()


# maxWords = 100
# ap_data = [line for line in ap_data if len(line) < maxWords]

# # my dats set is done
# ap_test = []

# #  here I am doubling the tuples

# for idx, l1 in enumerate(ap_data):
# 	sentence = []
# 	for l2 in ap_data[0:idx]:
# 		sentence.append((l1, l2))
#   	ap_test =  ap_test + sentence


# represen = ["Harbinger bought SkyTerra a satellite earlier this year",
# "IBM said Friday it has agreed to buy Unica Corp a marketing services for $num million or $num per share",
# "In October Adobe will be aquiring Web analytics firm Omniture for $num billion",
# "Business maker Deltek said Thursday that it will acquire database and market information Input Inc for $num million in an allcash transaction"]

# unrelated = ["Thousands of pending home sales may be in jeopardy unless Congress extends the June num deadline for buyers to close on their deals and claim a tax credit",
# "Brian Bonime num has a contract on a home in Margate Fla but is worried the short sale wont close in time to get the $num tax credit he was counting on",
# "The company which is based in China and had its initial public offering earlier this year earned $num million or num cents per share during the threemonth period",
# "The company also must pay $num as a civil penalty",
# "In afternoon trading Vical stock dropped $num or num percent to $num"]

# unwanted = ["Dells firstquarter results had showed businesses started replacing aging technology by buying new servers and other behindthescenes technology",
# "Longerterm however Nvidia will have to compete with Intel and AMD when it comes to selling chips to tablet makers",
# "Schnitzer Steel Industries said Wednesday it bought substantially all of the assets of Waco Inc in to expand its auto parts business",
# "CBS is selling WGNT in Norfolk to Local TV Holdings LLC"]

# para2vecSen = "Earlier this year Exxon Mobil bought XTO Energy to become Americas largest producer of natural gas"

# para2vec = ["References to the neerdowell farmer spill into the margins of Dallas Countys fledgling court logs",
# "If you are seeking an avenue to gamble its there for you at a very close proximity Greeley said",
# "Analysts had expected $num per share",
# "BristolMyerss stock dropped num percent or $num to $num"]

# represen = represen + [para2vecSen]

# represen = [line.split() for line in represen]
# unrelated = [line.split() for line in unrelated]
# unwanted = [line.split() for line in unwanted]
# para2vec = [line.split() for line in para2vec]

# ap_data =  represen + unrelated + unwanted + para2vec

# maxWords = 100
# ap_data = [line for line in ap_data if len(line) < maxWords]

# ap_test = []

# for sen1 in represen:
#     for sen2 in unrelated:
#         ap_test = ap_test + [(sen1, sen2)]

# for sen1 in represen:
#     for sen2 in unwanted:
#         ap_test = ap_test + [(sen1, sen2)]

# for sen1 in unwanted:
#     for sen2 in unrelated:
#         ap_test = ap_test + [(sen1, sen2)]

# for idx, sen1 in enumerate(represen):
#     for sen2 in represen[0:idx]:
#         ap_test = ap_test + [(sen1, sen2)]

# for sen in para2vec:
#     ap_test = ap_test + [(para2vecSen.split(), sen)]

seed_sentences = ["Intel buying German chipmakers wireless unit",
"Harbinger bought SkyTerra a satellite earlier this year",
"CBS is selling WGNT in Norfolk to Local TV Holdings LLC",
"Aug num Dell announces offer to buy numPar for $num per share or $num billion",
"In October Adobe will be aquiring Web analytics firm Omniture for $num billion",
"Thermo Fisher Scientific said it will buy Dionex for $num billion or $num per share",
"Bonfante wont name the but says it was bought a year ago by a large publicly traded company",
"RuralMetro will buy Pridemark Paramedic Services in order to expand its services in Colorado",
"Sprint also owns most of Clearwire which is building a nationwide wireless broadband network",
"Earlier this year Exxon Mobil bought XTO Energy to become Americas largest producer of natural gas",
"Motorola on Thursday said it has acquired Aloqa GmbH a German developer of locationbased for smart",
"IBM said Friday it has agreed to buy Unica Corp a marketing services for $num million or $num per share",
"The paper stayed in the family until num when Avis Tucker sold the business to the St Josephbased NewsPress Gazette",
"In num US Internet giant Yahoo acquired Jordans Maktoob then the Arab worlds largest online media for an undisclosed fee",
"Covidien PLC which makes drugs and medical devices said Wednesday it is buying its distribution partner Somanetics for almost $num million",
"ShoreTel acquired the intellectual property customer base and distribution network of the privately held Agito Networks in the allcash deal",
"Business maker Deltek said Thursday that it will acquire database and market information Input Inc for $num million in an allcash transaction",
"Biotechnology Celgene said Wednesday it is expanding its array of cancer treatments with a deal to buy Abraxis BioScience for $num billion in cash and stock",
"General Electric is paying $num billion to buy British oilfield Wellstream Holdings PLC and Dell is spending $num million for network storage company Compellent Inc",
"The EnnisKnupp deal was closed with Hewitt in the process of coming under the umbrella of insurance conglomerate Aon which agreed to buy the Lincolnshirebased July num for $num billion in cash and stock"]

seed_sentences = [line.split() for line in seed_sentences]


min_size = 4
max_words = 100
# test_size = 10000

f = open(path + "sentences", "r")
ap_data = f.read().split("\n")
ap_data = [line.split() for line in ap_data]
# ap_data = [token for token in ap_data if (len(token) > min_size and len(token) <= 100 )]
f.close()

ap_test = []
for l1 in seed_sentences:
	for l2 in ap_data:
		ap_test.append((l1, l2))

ap_data = ap_data + seed_sentences
# --------------------------------- GENSIM BEGINS ------------------------------------- #

epochs = 10
alpha = 0.04

model = Word2Vec(
	size=300,
	alpha=alpha,
	window=5,
	min_count=1,
	workers=multiprocessing.cpu_count(),
	min_alpha=alpha,
	sg=1
)

# here add the test set too
model.build_vocab(
	[t[0] for t in train] +
	[t[1] for t in train] +
	[t[0] for t in valid] +
	[t[1] for t in valid] +
	[t[0] for t in test] +
	[t[1] for t in test] +
	[t[0] for t in pretrain] +
	[t[1] for t in pretrain] +
	ap_data
)

model.intersect_word2vec_format(path + 'google_vectors.bin.gz', binary = True)
model.intersect_word2vec_format(path+ 'word_vectors_AP_word2vec_decay_300', binary = True)

# word2vec_model = gensim.models.word2vec.Word2Vec.load(path + model_directory + word2vec_model)
# Fine tuning for our dataset

# for epoch in range(epochs):
# 	shuffle(rawSentences)
# 	model.train(rawSentences)
# 	model.alpha = model.alpha / (1 + epoch)
# 	model.min_alpha = model.alpha

# ------------------------------------ GENSIM ENDS ---------------------------------------- #


# --------------------------------- TENSORFLOW BEGINS ------------------------------------- #

lstmSize = 50
batchSize = 10
maxWords = 100
inputVectorSize = 300

lamb = tf.placeholder("float", shape=())
X1 = tf.placeholder("float", [maxWords, None, inputVectorSize])
X2 = tf.placeholder("float", [maxWords, None, inputVectorSize])

earlyStop1 = tf.placeholder(tf.int32, [None])
earlyStop2 = tf.placeholder(tf.int32, [None])

Y = tf.placeholder("float", [None])

with tf.variable_scope('share_var') as scope:
	lstmLayer = rnn_cell.LSTMCell(lstmSize, initializer = tf.contrib.layers.xavier_initializer(), forget_bias = 2.5)
	lstmLayer = rnn_cell.InputProjectionWrapper(lstmLayer, lstmSize)
	X11 = [tf.reshape(i, (-1, inputVectorSize)) for i in tf.split(0, maxWords, X1)]
	X22 = [tf.reshape(i, (-1, inputVectorSize)) for i in tf.split(0, maxWords, X2)]
	_, h1 = rnn.rnn(lstmLayer, X11, dtype = 'float', sequence_length = earlyStop1)
	scope.reuse_variables()
	_, h2 = rnn.rnn(lstmLayer, X22, dtype = 'float', sequence_length = earlyStop2)

with tf.variable_scope('share_var2') as scope:
	decoderCell = rnn_cell.LSTMCell(lstmSize, initializer = tf.contrib.layers.xavier_initializer(), num_proj = inputVectorSize, forget_bias = 2.5)
	out1, _ = rnn.rnn(decoderCell, [h1]*maxWords, dtype = 'float', sequence_length = earlyStop1)
	scope.reuse_variables()
	out2, _ = rnn.rnn(decoderCell, [h2]*maxWords, dtype = 'float', sequence_length = earlyStop2)
	out1 = tf.pack(out1)
	out2 = tf.pack(out2)

with tf.variable_scope('share_var3') as scope:
	l1 = tf.reduce_mean(tf.square(tf.sub(out1, X1)))
	l2 = tf.reduce_mean(tf.square(tf.sub(out2, X2)))
	unsupLoss = (l1 + l2)*inputVectorSize

g = tf.exp(tf.neg(tf.reduce_sum(tf.abs(tf.sub(h1, h2)), 1)))

out = 4*g+1

supLoss = tf.reduce_mean(tf.square(tf.sub(Y, out)))

loss = supLoss + lamb*unsupLoss

trainStep = tf.train.AdamOptimizer().minimize(loss)
# Computing gradient
# gradsAndVars = opt.compute_gradients(loss)
# # Clipping gradient
# gradAndVarsCapped = [(tf.clip_by_value(grad, -1.0, 1.0), var) if grad != None for grad, var in gradsAndVars]
# # Applying gradient
# trainStep = opt.apply_gradients(gradAndVarsCapped)

init = tf.initialize_all_variables()

saver = tf.train.Saver()
# ---------------------------------- TENSORFLOW ENDS -------------------------------------- #

if pretrain:
	shuffle(pretrain)
	pretrainFeed = []
	for batch in [pretrain[i:i+batchSize] for i in range(0, len(pretrain)-batchSize+1, batchSize)]:	
		x1 = np.zeros([maxWords, batchSize, inputVectorSize])
		x2 = np.zeros([maxWords, batchSize, inputVectorSize])
		for word in range(maxWords):
			for index in range(min(batchSize, len(batch))):
				sentence1 = batch[index][0]
				sentence2 = batch[index][1]
				score = batch[index][2]
				if len(sentence1) >= word + 1:
					x1[word, index] = model[sentence1[word]]
				if len(sentence2) >= word + 1:
					x2[word, index] = model[sentence2[word]]
		es1 = np.array([len(batchElement[0]) for batchElement in batch])
		es2 = np.array([len(batchElement[1]) for batchElement in batch])
		y = np.array([float(batchElement[2]) for batchElement in batch])
		pretrainFeed.append((x1, x2, es1, es2, y))

trainFeed = []
for batch in [train[i:i+batchSize] for i in range(0, len(train)-batchSize+1, batchSize)]:	
	x1 = np.zeros([maxWords, batchSize, inputVectorSize])
	x2 = np.zeros([maxWords, batchSize, inputVectorSize])
	for word in range(maxWords):
		for index in range(min(batchSize, len(batch))):
			sentence1 = batch[index][0]
			sentence2 = batch[index][1]
			score = batch[index][2]
			if len(sentence1) >= word + 1:
				x1[word, index] = model[sentence1[word]]
			if len(sentence2) >= word + 1:
				x2[word, index] = model[sentence2[word]]
	es1 = np.array([len(batchElement[0]) for batchElement in batch])
	es2 = np.array([len(batchElement[1]) for batchElement in batch])
	y = np.array([float(batchElement[2]) for batchElement in batch])
	trainFeed.append((x1, x2, es1, es2, y))


vx1 = np.zeros([maxWords, len(valid), inputVectorSize])
vx2 = np.zeros([maxWords, len(valid), inputVectorSize])
for word in range(maxWords):
	for index in range(len(valid)):
		sentence1 = valid[index][0]
		sentence2 = valid[index][1]
		score = valid[index][2]
		if len(sentence1) >= word + 1:
			vx1[word, index] = model[sentence1[word]]
		if len(sentence2) >= word + 1:
			vx2[word, index] = model[sentence2[word]]


ves1 = np.array([len(batchElement[0]) for batchElement in valid])
ves2 = np.array([len(batchElement[1]) for batchElement in valid])
vy = np.array([float(batchElement[2]) for batchElement in valid])

tx1 = np.zeros([maxWords, len(test), inputVectorSize])
tx2 = np.zeros([maxWords, len(test), inputVectorSize])
for word in range(maxWords):
	for index in range(len(test)):
		sentence1 = test[index][0]
		sentence2 = test[index][1]
		score = test[index][2]
		if len(sentence1) >= word + 1:
			tx1[word, index] = model[sentence1[word]]
		if len(sentence2) >= word + 1:
			tx2[word, index] = model[sentence2[word]]


tes1 = np.array([len(batchElement[0]) for batchElement in test])
tes2 = np.array([len(batchElement[1]) for batchElement in test])
ty = np.array([float(batchElement[2]) for batchElement in test])
###############################

ttx1 = np.zeros([maxWords, len(ap_test), inputVectorSize])
ttx2 = np.zeros([maxWords, len(ap_test), inputVectorSize])

for word in range(maxWords):
	for index in range(len(ap_test)):
		sentence1 = ap_test[index][0]
		sentence2 = ap_test[index][1]
		if len(sentence1) >= word + 1:
			ttx1[word, index] = model[sentence1[word]]
		if len(sentence2) >= word + 1:
			ttx2[word, index] = model[sentence2[word]]


ttes1 = np.array([len(batchElement[0]) for batchElement in ap_test])
ttes2 = np.array([len(batchElement[1]) for batchElement in ap_test])
ttout = []

###############################

outString = ''
try:
	with tf.Session() as session:
		# saver.restore(session, modelPath + lstm_model)
		session.run(init)
		if pretrain:
			for epoch in range(10):
				shuffle(pretrainFeed)
				trainLoss = 0
				trainLoss2 = 0
				for tfe in pretrainFeed:
					x1 = tfe[0]
					x2 = tfe[1]
					es1 = tfe[2]
					es2 = tfe[3]
					y = tfe[4]
					feed = {X1: x1, X2: x2, earlyStop1: es1, earlyStop2: es2, Y: y, lamb: lda}
					l, l2, _ = session.run([supLoss, unsupLoss, trainStep], feed_dict = feed)
					trainLoss = trainLoss + l
					trainLoss2 = trainLoss2 + l2
				trainLoss = trainLoss/len(pretrainFeed)
				trainLoss2 = trainLoss2/len(pretrainFeed)
				currStr = 'PT Epoch: {}\nRMSE= {}\tUnsupRMSE={}\n'.format(epoch, np.sqrt(trainLoss), np.sqrt(trainLoss2))
				outString = outString + currStr
				print currStr

		for epoch in range(20):
			shuffle(trainFeed)
			trainLoss = 0
			trainLoss2 = 0
			for tfe in trainFeed:
				x1 = tfe[0]
				x2 = tfe[1]
				es1 = tfe[2]
				es2 = tfe[3]
				y = tfe[4]
				feed = {X1: x1, X2: x2, earlyStop1: es1, earlyStop2: es2, Y: y, lamb: lda}
				l, l2, _ = session.run([supLoss, unsupLoss, trainStep], feed_dict = feed)
				trainLoss = trainLoss + l
				trainLoss2 = trainLoss2 + l2

			trainLoss = trainLoss/len(trainFeed)
			trainLoss2 = trainLoss2/len(trainFeed)
			feedValid = {X1: vx1, X2: vx2, earlyStop1: ves1, earlyStop2: ves2, Y: vy, lamb: lda}
			l, l2 = session.run([supLoss, unsupLoss], feed_dict = feedValid)
			validLoss = l
			validLoss2 = l2
			feedTest = {X1: tx1[:, :len(test)/2, :], X2: tx2[:, :len(test)/2, :], earlyStop1: tes1[:len(test)/2], earlyStop2: tes2[:len(test)/2], Y: ty[:len(test)/2], lamb:lda}
			lt, lt2 = session.run([supLoss, unsupLoss], feed_dict = feedTest)
			testLoss = lt
			testLoss2 = lt2
			feedTest = {X1: tx1[:, len(test)/2:, :], X2: tx2[:, len(test)/2:, :], earlyStop1: tes1[len(test)/2:], earlyStop2: tes2[len(test)/2:], Y: ty[len(test)/2:], lamb:lda}
			lt, lt2 = session.run([supLoss, unsupLoss], feed_dict = feedTest)
			testLoss = (testLoss + lt)/2
			testLoss2 = (testLoss2 + lt2)/2
			currStr = 'Epoch: {}\n'.format(epoch)
			currStr = currStr + 'Train: RMSE= {}\tUnsupRMSE= {}\n'.format(np.sqrt(trainLoss), np.sqrt(trainLoss2))		
			currStr = currStr + 'Valid: RMSE= {}\n'.format(np.sqrt(validLoss))		
			currStr = currStr + 'Test: RMSE= {}\n'.format(np.sqrt(testLoss))		
			outString = outString + currStr
			print currStr

		# saver.restore(session, path + "model_1.ckpt")
		feed = {X1: ttx1, X2: ttx2, earlyStop1: ttes1, earlyStop2: ttes2}
		ttout = session.run(out, feed_dict = feed)
		f = open(resultPath + "test_results.txt", 'w')

		i = 0

		for tup in ap_test:
			strn = "###\n1. " + " ".join(tup[0]) + "\n2. " + " ".join(tup[1]) + "\nscore: " + str(ttout[i]) + "\n\n"
			f.write(strn)
			outString = outString + strn
			i = i + 1

		f.close()
		save_path = saver.save(session, modelPath + '{}_lstm11_lambda_{}.ckpt'.format(testDomain, lda))
		file = open(resultPath + '{}_lstm11_lambda_{}.txt'.format(testDomain, lda), 'w')
		file.write(outString)
		file.close()
		print outString

except:
	file = open(resultPath + '{}_lstm11_lambda_{}.txt'.format(testDomain, lda), 'w')
	file.write(outString)
	file.close()
	print outString
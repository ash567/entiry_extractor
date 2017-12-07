## The Rule Template (Takens from the seed dictionary)
company1 = ['Boeing', 'Atlantic', 'Veal', 'Lamb', 'Golf', 'SunPower', 'Ralph', 'Roberts', 'Comcast', 'SanDisk', 'IBM', 'Intel', 'Dell', 'Buffets', 'Berkshire']
aquire = ['bought', 'buys', 'buying']
company2 = ['Argon', 'Buckeye', 'airwaves', 'Knott', 'ATT', 'Vivisimo', 'Qlogic', 'Clerity', 'Waco']
list_dictionary = [company1, aquire, company2]

avoid = ['Co', 'Corp', 'company', 'Company', 'investors', 'advertising', 'provider', 'software', 'phones', 'rival', 'Inc', 'stake', "'Technologies", 'subscribers', 'technology', 'electronics', 'shares']
avoid = avoid + ['hardware', 'mobile', 'server']

code_path = "/home/biadmin/Documents/ishu_sync/Codes/gensim_codes/"
code_file = "similarity_funtions"

data_path = '/home/biadmin/Documents/ishu_sync/Data/APNewsData/'
data_directory = 'processed_data/'
model_directory = "saved_models/"
result_directory = "model_results/"


f = open(data_path + data_directory +  "final_processed", "r")
data = f.read().split("\n")
data = [line.split() for line in data]
f.close()

train = data[:len(data)/1000]



idx = 4514
similar =[i for (i, score )  in model_para.docvecs.most_similar(positive=[idx], negative = [], topn = 100)]
print "Original Sentence: " + " ".join(train[idx])
print "Similar Sentences Found"

for i in similar:
	print " ".join(train[i])



#----------------------------
# does not work
# import all similarity funtions
# import sys
# sys.path.insert(0, code_path)

# from similarity_funtions import *

#-----------------------------
# Either train the model or import it
#importing the model

model = gensim.models.word2vec.Word2Vec.load(data_path + model_directory + 'model_'+ model_name)

# then copy functions from the similarity
#--------------------------------------

# Model testing code
idx = 0
start = 0
end = len(train)

#changing the definition of train (during the model it was different)
# train = data[start:end]
threshold = 0.010

# f = open(data_path + result_directory + "results_ " + model_name, "a")

for idx, line1 in enumerate(train):
	line = []
	for word in line1:
		if (word in model.vocab and word not in avoid):
			line.append(word)
	if (len(line) == 0):
		continue
	(score, tokens) = dp_similarity(list_dictionary, line)
	if(score >= threshold):
		# f.write("\n$----#\n$$: " + " ".join(line) + "\n##: " + " ".join(tokens))
		# f.flush()
		print "\n$----#\n %d$$: "  %(idx) + " ".join(line) + "\n##: " + " ".join(tokens)

# f.close()
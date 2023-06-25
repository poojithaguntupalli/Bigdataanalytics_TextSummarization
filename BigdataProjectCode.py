#lsa

!apt-get install openjdk-8-jdk-headless -qq > /dev/null

# installing spark
!wget -q https://archive.apache.org/dist/spark/spark-3.0.0/spark-3.0.0-bin-hadoop3.2.tgz

# unzipping the spark file 
!tar xf spark-3.0.0-bin-hadoop3.2.tgz

#setting path
import os
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "/content/spark-3.0.0-bin-hadoop3.2"
!pip install -q findspark
import findspark
findspark.init()
from pyspark.sql import SparkSession
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import numpy as np
spark = SparkSession.builder.master("local[*]").getOrCreate()
sc = spark.sparkContext

!wget https://cs6350-s22.s3.amazonaws.com/Project/AmazonReviewsData.zip
!unzip AmazonReviewsData.zip

DatasetDirectory = 'AmazonReviewsData/'
dFiles = os.listdir(DatasetDirectory)
print(dFiles)

stopWord = stopwords.words('english')

class DataParser: 
  @staticmethod
  def getReviews(x):

    #extracting 6 columns from the dataset
    review_id, review_title, ratings, _,_ ,review = x.split('\t')
    sentncs = review.split('.')
    return [(review_id+'_'+str(e),f) for e,f in enumerate(sentncs)]
  
  @staticmethod
  def getDocuments(x):
    
    #pre processing on the text in the AmazonReviewData file.
    Wordlemmatizer = WordNetLemmatizer()
    #stopWord = stopwords.words('english')

    review_id, review_title, ratings,_, _, review = x.split('\t')
    sentncs  = review.split('.')

    Words_res = []

    for k,l in enumerate(sentncs):
      id = review_id + '_' + str(k)
      wrds = l.split(' ')
      if len(wrds) < 5 : 
        continue 
      
      # Removing stop words and lemantizing
      wrds = re.findall(r'[a-zA-Z]+', l)
      wrds = [wrd.lower() for wrd in wrds]
      wrds = [Wordlemmatizer.lemmatize(wrd) for wrd in wrds if wrd not in stopWord and len(wrd)>=3] 

      Words_res.append((id,[wrd for wrd in wrds]))

    return Words_res

  #computing term frequency 
  #taking text and review_id
  @staticmethod
  def tf(wrds,k):
    
    hm = {}
    for wrd in wrds:
        hm[wrd] = hm.get(wrd, 0) +1 
    return (k,hm)
   
  #Computing the IDF
  @staticmethod
  def idf(n, documentFreq):
    return np.log10(np.reciprocal(documentFreq) * n)
  
class LSA: 
  def __init__(self,spark, path):
    self.path = path 
    self.sc = spark.sparkContext
    
  #Keywords Extraction
  def keywordsExtraction(self,VT, rowheader, k = 5, n = 3):
    concepts = []
    for ind in np.fliplr(VT[:k,:].argsort()[:,-n:]):
        keywords = []
        for i in ind:
            keywords.append(rowheader[i])
        concepts.append(keywords)
    return concepts
  
  #Sentence extraction
  def SentencesExtraction(self,VT, reviews, colHeader, k=5, n=3):
    concepts = []
    for ind in np.fliplr(VT[:k,:].argsort()[:,-n:]):
        keysentncs = []
        for i in ind:
            keysentncs.append(reviews.lookup(colHeader[i]))
        concepts.append(keysentncs)
    return concepts
  
  #Function to find the review summaries
  def ReviewsSummary(self):
    sc,path = self.sc,self.path
    reviews = sc.textFile(path).flatMap(lambda sentenceline:DataParser.getReviews(sentenceline))
    documents = sc.textFile(path).flatMap(lambda sentenceline:DataParser.getDocuments(sentenceline))
    trmFreq = documents.map(lambda z: DataParser.tf(z[1], z[0]))
    vocabularyWordList = trmFreq.map(lambda x: [d for d in x[1]]).reduce(lambda a,b: a + b)
    vocabularyWordList = list(set(vocabularyWordList))
    trmfreqMatrix= lambda trmfreqdict: [trmfreqdict.get(wrd, 0) for wrd in vocabularyWordList]
    docfreqMatrix= lambda trmfreqdict: [ 1.0 if (trmfreqdict.get(wrd, 0) > 0) else 0. for wrd in vocabularyWordList]
    
    #Calculating Doc Frequency vector
    dfvector = trmFreq.map(lambda x: docfreqMatrix(x[1])).reduce(lambda a,b: np.array(a) + np.array(b))

    # Calculating Term Frequency matrix
    tefrmatrxDict = trmFreq.map(lambda a: (a[0], trmfreqMatrix(a[1]))).sortByKey()
    tefrmatrix = tefrmatrxDict.values().collect()
    colheader = tefrmatrxDict.keys().collect()
    rowHeader = vocabularyWordList

    tfmatrix = np.array(np.transpose(tefrmatrix))
    idfVector = DataParser.idf(len(colheader), dfvector).T
    idfVector = np.reshape(idfVector, (-1,1))
    tfidf = np.multiply(tfmatrix , idfVector)
    U, S, VT = np.linalg.svd(tfidf, full_matrices=0)
    concepts = self.SentencesExtraction(VT,reviews,colheader)

    U, S, VT = np.linalg.svd(tfidf.T, full_matrices=0)
    kywrds = self.keywordsExtraction(VT, rowHeader)

    reviewsumres = []
    for j,concept in enumerate(concepts):
        print('Concept '+str(j+1))
        print('Keywords:',kywrds[j])
        for j,sent in enumerate(concept):
            reviewsumres.append(str(sent))
            print ('Sentence '+str(j+1)+':\t'+str(sent))
        print('\n')
    return reviewsumres

# Running with 1 dataset product file  
s= LSA(spark,DatasetDirectory+dFiles[0])
print(s)
textSummariesConcepts = s.ReviewsSummary()

pip install git+https://github.com/vinodnimbalkar/PyTLDR.git

tmp =sc.textFile(DatasetDirectory+dFiles[0])
header = tmp.first()
tmpdata = tmp.filter(lambda row: row != header).flatMap(lambda line:DataParser.getReviews(line)).map(lambda h : h[1])
reviewdata= tmpdata.collect()
reviewdata = "".join(reviewdata)

from pytldr.summarize.lsa import LsaSummarizer
summarizer = LsaSummarizer()  # This is identical to the LsaOzsoy object

rfrncedata = summarizer.summarize(
    reviewdata, topics=15, length=15, binary_matrix=True, topic_sigma_threshold=0.75
)

print(len(rfrncedata))
for s in rfrncedata: 
  print(s)

!pip install rouge-score

textSummariesConcepts = [ s.encode("utf8") for s in textSummariesConcepts]
print(textSummariesConcepts)

from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
precision = []
recall = []
fscore = []

rfScores = []

for e,g in zip(textSummariesConcepts,rfrncedata):
  scores = scorer.score(e,g)
  precision.append(scores["rougeL"][0])
  recall.append(scores["rougeL"][1])
  fscore.append(scores["rougeL"][2])

  rfScores.append([scores["rougeL"][0],scores["rougeL"][1],scores["rougeL"][2]])
print("The calculated precision  = ",sum(precision)/len(precision))
print("The calculated Recall = ",sum(recall)/len(recall))
print("The calculated F Measure = ",sum(fscore)/len(fscore))
rfScores = np.mean(rfScores,axis=0)
print("The calculated rfScores= " + str(rfScores))

s= LSA(spark,DatasetDirectory+dFiles[1])
print(s)
textSummariesConcepts1 = s.ReviewsSummary()

tmp =sc.textFile(DatasetDirectory+dFiles[1])
header = tmp.first()
tmpdata = tmp.filter(lambda row: row != header).flatMap(lambda line:DataParser.getReviews(line)).map(lambda h : h[1])
reviewdata= tmpdata.collect()
reviewdata = "".join(reviewdata)

from pytldr.summarize.lsa import LsaSummarizer
summarizer = LsaSummarizer()  # This is identical to the LsaOzsoy object

rfrncedata1 = summarizer.summarize(
    reviewdata, topics=15, length=15, binary_matrix=True, topic_sigma_threshold=0.75
)

precision = []
recall = []
fscore = []

rfScores = []

for e,g in zip(textSummariesConcepts1,rfrncedata1):
  scores = scorer.score(e,g)
  precision.append(scores["rougeL"][0])
  recall.append(scores["rougeL"][1])
  fscore.append(scores["rougeL"][2])

  rfScores.append([scores["rougeL"][0],scores["rougeL"][1],scores["rougeL"][2]])
print("The calculated precision  = ",sum(precision)/len(precision))
print("The calculated Recall = ",sum(recall)/len(recall))
print("The calculated F Measure = ",sum(fscore)/len(fscore))
rfScores = np.mean(rfScores,axis=0)
print("The calculated rfScores= " + str(rfScores))

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

r = ['Rouge-1', 'Rouge-2', 'Rouge-L']
data = rfScores

x_pos = [i for i, _ in enumerate(r)]

plt.plot(x_pos,data,marker='o')
plt.xlabel("Rouge Metric")
plt.ylabel("Value")
plt.title("Comparing rouge metrics")

plt.xticks(x_pos, r)

plt.show()

lsamean = []

#Calculating for all the files in the directory
for i in range(len(dFiles)):
  filePath = DatasetDirectory + dFiles[i]
  print(filePath)
  s= LSA(spark,filePath)
  summariesdata = s.ReviewsSummary()
  summariesdata = [ s.encode("utf8") for s in summariesdata]

  
  #reference text summaries
  tmp =sc.textFile(filePath).flatMap(lambda line:DataParser.getReviews(line)).map(lambda h : h[1])
  reviewdata= tmp.collect()
  reviewdata = "".join(reviewdata)
  
  summarizer = LsaSummarizer()  
  rfrncedata = summarizer.summarize(
      reviewdata, topics=15, length=15, binary_matrix=True, topic_sigma_threshold=0.75
  )
  scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)
  rougeScr = []
  for e,g in zip(textSummariesConcepts,rfrncedata):
  
    scores = scorer.score(e,g)
    rougeScr.append(scores["rouge2"][2])

  averageScore = sum(rougeScr)/len(rougeScr)
  print("Average score of " + filePath + " is "+ str(averageScore)+"\n")

  lsamean.append(averageScore)

productids=[x.split('.')[0] for x in dFiles]
for l in range(len(productids)):
  print(l+1,productids[l])

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('ggplot')

N = len(dFiles)
index = np.arange(N) 
wid = 0.4      
plt.figure(figsize=(10,10))
plt.bar(index, lsamean, wid,color='blue')
plt.ylabel('Scores')
plt.title('LSA')
plt.xticks(index + wid / 2, range(1,N+1))
plt.show()

#textrank


sc = spark.sparkContext


!wget https://cs6350-s22.s3.amazonaws.com/Project/AmazonReviewsData.zip
!unzip AmazonReviewsData.zip

INPUT_PATH = 'AmazonReviewsData/'
inputfiles = os.listdir(INPUT_PATH)
print(inputfiles)

  class DataParser:
    def parser(self,spark,path):
      sc = spark.sparkContext
      #loading the reviews data file rdd and get each review as a sentence and returning words as value and its corresponding review_id as key
      review_phrases = sc.textFile(path).flatMap(lambda review : self.__class__.createWordKvpair(self,review))
      
      # vertices of graph which contains review_id and sentences as phrases list from review sentences
      vertices = review_phrases.map(lambda line: self.__class__.vertex_create(self,line))
      review_phrases = review_phrases.cache()
      #adjacency list for each vertex
      specific_vertex = vertices.collect()
      adjncy_list = vertices.map(lambda vert: self.prepareAdjacencyList(vert,specific_vertex))
      # Filtering only vertices that have more nodes in its adjacency list
      adjncy_list = adjncy_list.filter(lambda lin: len(lin[1]) > 0).cache()
      return adjncy_list,review_phrases

    #similarity = (count of common words in both the vertices)/ (1 + log2(len(vertex1)) + log2(len(vertex2)))
    def compute_similarity(self, ver1, ver2):
      key1,value1 = ver1[0],ver1[1]
      key2,value2 = ver2[0],ver2[1]
      if key1 != key2:
          wordcount = len(set(value1).intersection(value2))
          loglength = np.log2(len(value1)) + np.log2(len(value2))
          simvalue = wordcount/(loglength + 1)
          if simvalue != 0:
              return (key2, simvalue)

    #adjacency list of one vertex with rest of the others
    def prepareAdjacencyList(self, vertx, vertices):
      k,v = vertx[0],vertx[1]
      maped = {}
      for g in vertices:
          ed = self.compute_similarity(vertx, g)
          if ed is not None:
              maped[ed[0]] = ed[1]
      return (k, maped)
    
    def vertex_create(self,line):
      #Take review id and sentence tuple and create vertex as (review_id, sentence)
      review_id, sentence = line[0], line[1]
      wlematizr = WordNetLemmatizer()
      from nltk.corpus import stopwords
      stopwrds = stopwords.words('english')
      wrds = re.findall(r'[a-zA-Z]+', sentence)
      wrds = [wlematizr.lemmatize(wrd.lower()) for wrd in wrds if wrd.lower() not in stopwrds]
      wrds = [wr for wr in wrds if len(wr) > 3]
      return review_id, wrds

    def createWordKvpair(self,line):
      review = line.split("\t")
      review_id = review[0]
      review_phrases = review[5].split(".")
      out = []
      for idx, sentence in enumerate(review_phrases):
          sentence_id = review_id + '_' + str(idx)
          sntnceLen = len(sentence.split(" "))
          if 10 < sntnceLen < 30:
              out.append((sentence_id, sentence))
      return out

class TextRank:
  def computeSummary(self,spark,adjncy_list,review_phrase,no_of_iter,number_of_summary):
    sc = spark.sparkContext
     # Set the default value to 0.15
    graph_ranked = adjncy_list.map(lambda vertex: (vertex[0], 0.15))

    for i in range(0,no_of_iter):
        # TextRank Formula: TR(Vi) = (1-d) + d* SUM,Vj in In(Vi) [[Wji * PR(Vj)]/ [SUM, Vk in Out(Vj) [Wjk]]]
        contribution = adjncy_list.join(graph_ranked).flatMap(lambda neighb_rank: self.measure_contributions(neighb_rank[1][0], neighb_rank[1][1]))
        graph_ranked = contribution.reduceByKey(lambda a,b: a+b).mapValues(lambda rank: 0.15 + 0.85 * rank)

    res_out = []
    outrank=[]
    # Print the sentences that have higher rank
    resRank = graph_ranked.collect()
    resu = sorted(resRank, key=lambda x: x[1], reverse=True)
    for j in range(0, number_of_summary):
        res_out.append(str(review_phrase.lookup(resu[j][0])))
        outrank.append(str(round(resu[j][1],2)))
        print('Rank= ' + str(round(resu[j][1],2)) + '\tSentence= ' + str(review_phrase.lookup(resu[j][0])))
        
    return outrank,res_out

  def measure_contributions(self,neighb_map, node_rank):
    result = []
    res_weight = sum(neighb_map.values())
    for key,w in neighb_map.items():
        cont = (node_rank*w)/res_weight
        result.append((key, cont))
    return result


if __name__ == "__main__":
  dp_instance = DataParser()
  dataset_score = []
  for path in inputfiles:
    print('Top ranked reviews of file '+INPUT_PATH+path)
    adjncy_list,review_phrase = dp_instance.parser(spark,INPUT_PATH+path)
    textrank_instance = TextRank()
    Outrank,result_summary = textrank_instance.computeSummary(spark,adjncy_list,review_phrase,5,5)
 
    
    plt.bar(result_summary,Outrank, color='green')
    plt.xlabel("Sentence")
    plt.ylabel("Rank")
    plt.title("Rank vs Sentences")
    x=['Sentence1','Sentence2','Sentence3','Sentence4','Sentence5']
    x_summary=np.array(result_summary)
    plt.xticks(x_summary, x)
    plt.show()

   

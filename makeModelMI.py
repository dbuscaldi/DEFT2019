#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, re
import codecs
import math

import pickle
import gzip

import numpy as np
from numpy import linalg
from gensim.utils import tokenize

from whoosh.analysis import * #stemming module

class Model:
	#a model contains the indices, widx and lidx, and the LTM matrix
	#we use the model to serialize it using cPickle
	def __init__(self, wordict, labeldict, M):
		self.wordict=wordict
		self.labeldict=labeldict
		self.M=M

	def setWordFrequencies(self, df, N):
		self.freqs=df
		self.N=N

	def getWordFrequencies(self):
		return self.freqs

	def getMatrix(self):
		return self.M

	def getWordDict(self):
		return self.wordict

	def getLabelDict(self):
		return self.labeldict

class Notice:
	id=""
	text=""
	taggedtext=""
	labels=""

	def __init__(self, id, labels):
		self.id=id
		self.labels=labels

	def __hash__(self):
		return hash((self.id))

	def __eq__(self, other):
		return (self.id) == (other.id)

	def setTags(self, str):
		self.taggedtext=str

	def setText(self, str):
		self.text=str

	def write(self):
		print self.id, self.labels, self.text

collection = {}

idx={} #index document id -> numerical id for matrices
widx={} #word -> numerical word id for matrices
lidx={} #label -> numerical label id for matrices
wdict={} #term-document matrix
ldict={} #label-document matrix
"""
print "loading Google freqs"
gfreq={}
gMf=0 #google maxfreq
gf=codecs.open("unigrams-fr.dat", "r")
for line in gf.xreadlines():
	els=line.strip().split('\t')
	gfreq[els[0]]=int(els[1])
	if int(els[1]) > gMf:
		gMf=int(els[1])
gf.close()
"""
print "parsing data"
#donner le repertoire base où se trouvent les fichiers texte et la ref donnees-t1-ref.csv
counter=0
try:
	root=sys.argv[1] #par default data/TRAIN-T1
except IndexError:
	root="data/TRAIN-T1"


refFile=os.path.join(root, "donnees-t1-ref.csv")
ref=codecs.open(refFile, "r", "utf-8")
for line in ref.readlines():
	els=line.strip().split("\t")
	id1=els[0].replace(".txt", "")
	id2=els[1].replace(".txt", "")
	idx[id1]=counter
	counter+=1
	idx[id2]=counter
	counter+=1
	labels=els[3:]
	n1 = Notice(id1, labels)
	n2 = Notice(id2, labels)
	collection[id1]=n1
	collection[id2]=n2

N=counter+1 #we use this as the size of the collection

for file in os.listdir(root):
    if file.endswith(".txt") and not file.startswith("._"):
        if file.startswith("README"): continue
        id=file.replace(".txt", "")
        ffname=os.path.join(root, file)
        tf=codecs.open(ffname, "r", "latin-1")
        lines=tf.readlines()
        tf.close()

        n=collection[id]
        n.setText(' '.join(lines))

#make term-document matrix and label-document matrix to extract MI
wordid=0
labelid=0

ana = LanguageAnalyzer("fr") #STEMMING

for n in collection.values():
	tokens=list(tokenize(n.text, deacc=True))
	docid=n.id
	id=idx[docid]
	for t in tokens:
		termlist=[token.text for token in ana(t)] #STEMMING
		try:
			word=termlist[0] #STEMMING (use items[0] if not stemming)
		except IndexError:
			continue
        try:
			dlist=wdict[word] #STEMMING
        except KeyError:
			dlist=set([]) #let's use set since we're interested if the word appears or not in the document
			widx[word]=wordid #update word index #STEMMING
			wordid+=1
        dlist.add(id)
        wdict[word]=dlist #STEMMING
	for l in n.labels:
		l=l.strip()
		try:
			dlist=ldict[l]
		except KeyError:
			lidx[l]=labelid #update label index
			labelid+=1
			dlist=[]
		dlist.append(id)
		ldict[l]=dlist

"""
#word similarities
wsim={}
wlist=[]
wlist=wdict.keys()
for i in xrange(len(wlist)):
	w=wlist[i]
	wdocs=wdict[w]
	wf=len(wdocs)
	if i < len(wlist)-1:
		for k in wlist[(i+1):]:
			kdocs=wdict[k]
			kf=len(kdocs)
			isect=set(wdocs) & set(kdocs)
			il=len(isect)
			if il > 0 and wf > 3 and kf > 3:
				mi=math.log1p(N*il)-math.log1p(wf*kf)
				try:
					mv=wsim[w]
				except KeyError:
					mv=[]
				mv.append((k, mi))
				wsim[w]=mv
"""

#init label-term matrix (rows=labels, columns=terms)
LTM=np.zeros((labelid, wordid))

print "calculating MI scores"
#calculate MIs
for l in ldict.keys():
	#l is the label
	lf=len(ldict[l])
	ldl=ldict[l]
	if len(ldl) < 5: continue # set a threshold on the frequency of labels
	for k in wdict.keys(): #word
		#if l== k : print l, k
		kdl=wdict[k]
		kf=len(kdl)
		isect=set(ldl) & set(kdl)
		il=len(isect)
		if il > 3: #set a threshold on the frequency of mutual occurrence
			mi=math.log1p(N*il)-math.log1p(lf*kf)
			#mi=math.log1p(N*il)-math.log1p(kf*kf) #using p(label|word)
			if mi > 0:
				i=lidx[l]
				j=widx[k]
				LTM[i][j]=mi
				#try:
				#	LTM[i][j]=mi*(math.log(gMf)-math.log(gfreq[k])) #smooth MI by IDF in Google (to reduce importance of very frequent words)
				#except KeyError:
				#	pass
				#optionally: extend mi to words that are similar to k but not co-occurring with l
				#(maybe computationally expensive...)
"""
# apply filter to LTM
print "normalizing weight matrix"

nLTM=np.zeros((labelid, wordid))
for l in ldict.keys():
	i=lidx[l]
	labelvec=LTM[i]
	for k in xrange(len(labelvec)):
		if i <> k:
			wvec=LTM[:, k]
			wiset=np.flatnonzero(wvec)
			wnzv=[]
			for j in wiset:
				wnzv.append(wvec[j])

			if sum(wnzv) <> 0:
				nzavg=np.mean(wnzv)
				nzstd=np.std(wnzv)

				if LTM[i,k] >= nzstd+nzavg:
					#use the weight if and only if the difference is statistically significative
					#nLTM[i,k] = LTM[i,k] #binarized version: nLTM[i,k]=1.0
					nLTM[i,k] = 1.0
					#else: leave 0

m=Model(widx, lidx, nLTM) #use LTM for the standard
"""
#SVD and LSA
U, S, Vt = linalg.svd(LTM, full_matrices=False)

print S[:100]

#print linalg.norm(S) 95 for archeo, it seems 100 is a reasonable parameter

Sprime = np.zeros(len(S))

print S.shape, Sprime.shape

#reducing eigenvector dimensionality
i=0
for k in S[:100]:
	Sprime[i]=k
	i+=1

Sk= np.diag(Sprime)

print U.shape, Sk.shape, Vt.shape

rM= np.dot(U, np.dot(Sk, Vt))
#print np.allclose(LTM, rM)

#m=Model(widx, lidx, LTM)
m=Model(widx, lidx, rM) #use LSA matrix
m.setWordFrequencies(wdict, len(collection))

print "Saving model to model.pklz"
f = gzip.open('model.pklz','wb')
pickle.dump(m,f)
f.close()

# to create an inverse index:
#b = {v: k for k, v in a.iteritems()}

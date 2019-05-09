#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, re
import codecs
import math

import pickle
import gzip
import argparse

import numpy as np
from scipy.spatial.distance import cosine
from gensim.utils import tokenize

from whoosh.analysis import * #stemming module

class Model: #needed for the unpickle
	#a model contains the indices, widx and lidx, and the LTM matrix
	#we use the model to serialize it using cPickle
	def __init__(self, wordict, labeldict, M):
		self.wordict=wordict
		self.labeldict=labeldict
		self.M=M

	def setWordFrequencies(self, df):
		self.freqs=df

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
	assignedlabels=""

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

	def setAssignedLabels(self, labels):
		self.assignedlabels=labels

	def write(self):
		print self.id, self.labels, self.taggedtext, self.text

collection = {} #the collection to annotate

widx={} #word -> numerical word id for matrices
lidx={} #label -> numerical label id for matrices

parser = argparse.ArgumentParser(description='Annotate text using a MI-based model.')
parser.add_argument('model', metavar='model',
                   help='the .pklz file containing the model')
parser.add_argument('dir', metavar='dir',
                   help='the directory containing the files to annotate')

args = parser.parse_args()

print "loading model "+args.model

f = gzip.open(args.model,'rb')
m=pickle.load(f)
f.close()

"""
print ilidx[0]
for j in xrange(len(LTM[0])):
	if LTM[0][j] > 0:
		print iwidx[j], LTM[0][j]
"""

widx = m.getWordDict()
lidx = m.getLabelDict()
iwidx = {v: k for k, v in widx.iteritems()} #inverse word index
ilidx = {v: k for k, v in lidx.iteritems()} #inverse label index

LTM=m.getMatrix()

print "reading files in directory", args.dir
counter=0

print "parsing data"
#donner le repertoire base où se trouvent les fichiers texte et la ref donnees-t1-ref.csv
counter=0
root=args.dir
refFile=os.path.join(root, "donnees-t1-ref.csv")
ref=codecs.open(refFile, "r", "utf-8")
for line in ref.readlines():
	els=line.strip().split("\t")
	id1=els[0].replace(".txt", "")
	id2=els[1].replace(".txt", "")
	counter+=1
	counter+=1
	labels=els[3:]
	n1 = Notice(id1, labels)
	n2 = Notice(id2, labels)
	collection[id1]=n1
	collection[id2]=n2

N=counter+1 #we use this as the size of the collection

for file in os.listdir(root):
    if file.endswith(".txt") and not file.startswith("._"):
        if file.startswith("README") or file.startswith("filelist"): continue
        id=file.replace(".txt", "")
        ffname=os.path.join(root, file)
        tf=codecs.open(ffname, "r", "latin-1")
        lines=tf.readlines()
        tf.close()

        n=collection[id]
        n.setText(' '.join(lines))

print "annotating..."
#print LTM.shape, len(lidx), len(widx)

wordfreqs=m.getWordFrequencies() #word frequencies over the training collection

ana = LanguageAnalyzer("fr") #STEMMING

for id in collection.keys():
	found_labels=set([])
	vec=np.zeros(len(widx)) # the vector associated to this document
	notice=collection[id]
	tokens=list(tokenize(notice.text, deacc=True))
	notice_indices=[] #not null vec indices
	for t in tokens:
		termlist=[token.text for token in ana(t)] #STEMMING
		try:
			word=termlist[0] #STEMMING
		except IndexError:
			continue
		try:
		    index=widx[word]
		    vec[index]=1.0
		    notice_indices.append(index)
		except KeyError:
		    continue #move to next token, this one is not in the dictionary
	"""
	#applying idf to all elements of the vector
	for i in xrange(len(vec)):
		el = vec[i]
		word = iwidx[i]
		wf=len(wordfreqs[word])
		if el > 0:
			nel = el * (math.log1p(float(m.N)) - math.log1p(float(wf)))
			vec[i]=nel
			#print codecs.encode(word, "utf-8"), nel
	"""
	ratings=[]
	for i in xrange(len(lidx)):
		labelvec=LTM[i]
		"""for j in xrange(len(vec)):
			el = vec[j]
			word = iwidx[j]
			wf=len(wordfreqs[word])
			if el > 0:
				try:
					print codecs.encode(ilidx[i], "utf-8"), codecs.encode(word, "utf-8"), el, wf, labelvec[j]
				except: pass
		"""
		s=0
		for k in notice_indices: #this is more effective than normalizing the whole matrix
			if LTM[i,k] == 0: continue

			wvec=LTM[:, k]
			wiset=np.flatnonzero(wvec)
			wnzv=[]
			for j in wiset:
				wnzv.append(wvec[j])
			nzavg=np.mean(wnzv)
			nzstd=np.std(wnzv)

			#print ilidx[i], codecs.encode(iwidx[k], "utf-8")
			if LTM[i,k] > (0.5*nzstd)+nzavg:
				#print "->", LTM[i,k], nzavg
				s+=vec[k]*labelvec[k]
				#s+=1 #binarization

		#score=np.dot(vec, labelvec) #cosine, np.dot
		#score=(1-cosine(vec, labelvec)) #with cosine 0 means they are the same, 1 they are completely different
		#if score > 0 or s > 0: print "final score for:", ilidx[i], score, s

		if not math.isnan(s):
			ratings.append((ilidx[i], s))

	ratings.sort(key=lambda x : -x[1])
	for r in ratings[:20]:
		if r[1] > 0:
			found_labels.add(r[0])
	#print ratings[:5]
	notice.setAssignedLabels(found_labels)
	collection[id]=notice


print "evaluating..."
scores={}

for k in collection.keys():
	notice=collection[k]
	labels=set(notice.labels)
	alabels=notice.assignedlabels
	print "ref:", labels
	print "assigned:", alabels
	ll=len(labels)
	all=len(alabels)
	isect= labels & alabels
	il = len(isect)

	if ll > 0:
		recall=float(il)/float(ll)
	else: recall=0.0
	if all > 0:
		precision=float(il)/float(all)
	else: precision= 0.0

	scores[k]=(recall, precision)


sum_prec=0
sum_rec=0
print "id, recall, precision:"
for k in scores.keys():
	print k, scores[k][0], scores[k][1]
	sum_prec+=scores[k][1]
	sum_rec+=scores[k][0]

prec= float(sum_prec)/float(len(scores))
rec= float(sum_rec)/float(len(scores))
print "average precision:" , prec
print "average recall:" , rec
print "F-measure:", 2* prec*rec/(prec+rec)

#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, re
import codecs
import rdflib
import argparse, pickle

from rdflib import OWL, RDFS

from whoosh.index import create_in, open_dir
from whoosh.fields import *
from whoosh.lang import *
from whoosh.analysis import *

from whoosh.qparser import QueryParser

from nltk.util import ngrams

from anytree import Node, RenderTree, PreOrderIter

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

parser = argparse.ArgumentParser(description='Index ontology with labels and annotate text.')
parser.add_argument('--index', dest='idx', help='index to be created')
parser.add_argument('--ann', dest='annf', help='directory containing the files to annotate')
parser.add_argument('--repo', dest='repo', help='index to be used')

args = parser.parse_args()

meshroot=pickle.load(open('mesh_anytree2.pkl', 'rb'))
meshmodel=pickle.load(open('mesh_poincare.pkl', 'rb'))

nodes = [node.name for node in PreOrderIter(meshroot)]

meshFreqs={}
mffile=open("meshFreqs.txt", "r")
for line in mffile.readlines():
	els=line.strip().split('\t')
	try:
		meshFreqs[els[0]]=int(els[1])
	except:
		pass
mffile.close()

ana = LanguageAnalyzer("fr")

if args.idx <> None:
	if not os.path.exists(args.idx): os.mkdir(args.idx)
	schema = Schema(concept=ID(stored=True), label=TEXT(stored=True, analyzer=ana))
	ix = create_in(args.idx, schema)
	writer = ix.writer()

	print "Schema created, indexing labels..."

	for node in nodes:
		writer.add_document(concept=node, label=node)

	writer.commit()
	print "Indexing complete"
	sys.exit(0)

if args.annf <> None and args.repo <> None:
	print "reading files in directory", args.annf
	counter=0

	print "parsing data"
	#donner le repertoire base où se trouvent les fichiers texte et la ref donnees-t1-ref.csv
	counter=0
	root=args.annf
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

	for file in os.listdir(root):
	    if file.endswith(".txt") and not file.startswith("._"):
	        if file.startswith("README") or file.startswith("filelist"): continue
	        id=file.replace(".txt", "")
	        ffname=os.path.join(root, file)
	        tf=codecs.open(ffname, "r", "utf-8")
	        lines=tf.readlines()
	        tf.close()

	        n=collection[id]
	        n.setText(' '.join(lines))

	print "annotating..."
	#RE version
	"""for k in collection.keys():
		notice=collection[k]
		found_labels=set([])
		sentence = notice.text
		for node in nodes:
			label=node.replace("(", "\(")
			label=label.replace(")", "\)") #maybe replace all content within () with nothing
			label=label.replace("+", ".")
			label=label.replace("-", ".")
			label=label.replace("*", ".")
			if re.search(label, sentence, re.IGNORECASE):
				found_labels.add(label.lower())
		notice.setAssignedLabels(found_labels)
		collection[k]=notice

	"""#IR version
	ix = open_dir(args.repo)
	with ix.searcher() as searcher:
		for k in collection.keys():
			found_labels=set([])
			labels_dict={} #stocks parameters for each label: score, position in text, freq_text, freq_collection, freq_google, mesh_embeddings, length etc.
			notice=collection[k]
			sentence = notice.text
			reflabels=set(notice.labels)

			n = 3
			trigrams = ngrams(sentence.split(), n)
			position=0
			for grams in trigrams:
				#print grams
				query = QueryParser("label", schema=ix.schema).parse(' '.join(grams))
				results = searcher.search(query, limit=20)
				if len(results) > 0:
					ass_label=results[0]["label"]
					score=results[0].score
					#print ass_label, score
					freq_text=len(re.findall(ass_label, sentence)) #how many times it appears in text
					rel_pos=float(position)/float(len(sentence.split()))
					coords=meshmodel.kv.word_vec(ass_label)
					try:
						freq_collection=meshFreqs[ass_label.lower().encode("utf-8")]
					except:
						freq_collection=0
					if ass_label.lower().encode("utf-8") in reflabels: cls=1
					else: cls=0
					print ass_label, score, freq_text, rel_pos, coords[0], coords[1], freq_collection, cls
					if score > 10.0:
						found_labels.add(ass_label.lower().encode("utf-8"))
				position+=1
			notice.setAssignedLabels(found_labels)
			collection[k]=notice

	#evaluation:

	N=counter

	print "evaluating..."
	scores={}
	sum_prec=0
	sum_rec=0
	for k in collection.keys():
		notice=collection[k]
		labels=set(notice.labels)
		alabels=notice.assignedlabels
		print "ref:", ';'.join(labels)
		print "assigned:", ';'.join(alabels)
		print "text:"
		print notice.text
		print "--------------------"
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
		sum_prec+=precision
		sum_rec+=recall

	print "id, recall, precision:"
	for k in scores.keys():
		print k, scores[k][0], scores[k][1]

	prec= float(sum_prec)/float(N)
	rec= float(sum_rec)/float(N)
	print "average precision:" , prec
	print "average recall:" , rec
	print "F-measure:", 2* prec*rec/(prec+rec)

else:
	print "Both annotation file and index repository must be specified"

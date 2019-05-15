import re, sys, os
import argparse, pickle
from anytree import Node, RenderTree, PreOrderIter
import codecs

meshroot=pickle.load(open('mesh_anytree2.pkl', 'rb'))
nodes = [node.name for node in PreOrderIter(meshroot)]

freqs={}

bigtext=""
root=sys.argv[1]

for file in os.listdir(root):
    if file.endswith(".txt") and not file.startswith("._"):
        if file.startswith("README") or file.startswith("filelist"): continue
        id=file.replace(".txt", "")
        ffname=os.path.join(root, file)
        tf=codecs.open(ffname, "r", "utf-8")
        lines=tf.readlines()
        tf.close()
        bigtext=bigtext+' '.join(lines)

for node in nodes:
    label=node.replace("(", "\(")
    label=label.replace(")", "\)") #maybe replace all content within () with nothing
    label=label.replace("+", ".")
    label=label.replace("-", ".")
    label=label.replace("*", ".")
    freqs[label.lower()]=len(re.findall(label.lower(), bigtext))

for elem in freqs.keys():
    print(elem+"\t"+str(freqs[elem]))

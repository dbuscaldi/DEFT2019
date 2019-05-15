from gensim.models.poincare import PoincareModel

import pickle

from anytree import Node, RenderTree, PreOrderIter

meshroot=pickle.load(open('mesh_anytree.pkl', 'rb'))

nodes = [node.name for node in PreOrderIter(meshroot)]

relations = []

print("loading relations...")
for node in PreOrderIter(meshroot):
    p=node.parent
    if p!=None:
        rel=(node.name, p.name)
        relations.append(rel)
        #print(rel)

print("training model...")
model = PoincareModel(relations, negative=2)
model.train(epochs=50)

print("saving model...")
with open('mesh_poincare.pkl', 'wb') as output:
    pickle.dump(model, output, protocol=2)

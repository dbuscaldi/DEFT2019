#!/usr/local/bin/python3
from owlready2 import *

onto=get_ontology("./owlapi.xrdf").load()

"""res=onto.search(prefLabel="*osome") #anche altLabel
for c in res:
    print(c.prefLabel)
"""

for c in onto.classes():
    print (c.prefLabel)
    for a in c.ancestors():
        try:
            label=a.prefLabel
        except AttributeError:
            label=str(a)
        print("->", label) #ancestors

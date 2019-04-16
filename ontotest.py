#!/usr/local/bin/python3
from owlready2 import *

onto=get_ontology("./owlapi.xrdf").load()

res=onto.search(prefLabel="*osome") #anche altLabel
for c in res:
    print(c.prefLabel)

ancest=c.ancestors() #ancestors

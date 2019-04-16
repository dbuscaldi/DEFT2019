from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
import whoosh.index as index
from whoosh.analysis import *
from whoosh.query import *


stats={} #count first letter of tags

ix = index.open_dir("indexdir")

ref=open("data/TRAIN-T1/donnees-t1-ref.csv", "r")
qp = QueryParser("name", ix.schema, termclass=Variations)

cnt=0
with ix.searcher() as searcher:
    for line in ref.readlines():
        words=line.strip().split('\t')[3:]
        for w in words:
            cnt+=1
            parsed = qp.parse(w)
            docs = searcher.search(parsed)

            #print("Docs for : "+w)
            if len(docs) > 0:
                d=docs[0]
                tags=d["tags"].split(",")
                for t in tags:
                    try:
                        val=stats[t[0]]
                    except KeyError:
                        val=0
                    stats[t[0]]=(val+1)
                    #print(t[0])
                #print(d["id"], d["name"], d["tags"])
            #for d in docs:
            #    print(d["id"], d["name"], d["tags"])
print(cnt)
for k in stats.keys():
    print(k, stats[k])

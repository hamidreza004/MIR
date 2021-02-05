import json

import networkx as nx

with open('../crawler/data.json') as json_file:
    data = json.load(json_file)

G = nx.DiGraph()

for paper in data:
    paper_id = paper['id']
    paper_refs = paper['references']

    G.add_node(paper_id)
    G.add_edges_from([(paper_id, ref) for ref in paper_refs])


alpha = float(input('enter alpha:\n'))

pr = nx.pagerank(G, alpha)

sorted_pr = dict(sorted(pr.items(), key=lambda item: -item[1]))
print(sorted_pr)

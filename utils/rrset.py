import random
from collections import deque

def generate_rr_set(G, start_node, p):
    rr_set = set([start_node])
    queue = deque([start_node])

    while queue:
        v = queue.popleft()
        # dùng cho directed graph
        # for u in G.predecessors(v):
        # dùng cho undirected graph
        for u in G.neighbors(v):
            if u not in rr_set and random.random() <= p:
                rr_set.add(u)
                queue.append(u)
    return rr_set


def generate_rr_sets(G, nodes, theta, p):
    rr_sets = []
    node_list = list(nodes)

    for _ in range(theta):
        v = random.choice(node_list)
        rr = generate_rr_set(G, v, p)
        rr_sets.append(rr)

    return rr_sets


def coverage(seed_set, rr_sets):
    covered = 0
    S = set(seed_set)
    for rr in rr_sets:
        if S & rr:
            covered += 1
    return covered
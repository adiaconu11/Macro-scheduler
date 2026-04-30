"""
A module for graph algorithms used in the QECC codebase. 
This includes functions for bipartite edge coloring, which is used in the CSS code construction to 
assign checks to time steps while respecting qubit-sharing constraints.
"""
import copy

try:
    import rustworkx as rx
except ImportError:  # pragma: no cover - optional dependency
    rx = None

def bpm(i, match_to, visited, adj):
    """bipartite perfect matching
    """
    for j in adj[i]:
        if not visited[j]:
            visited[j] = True
            if match_to[j] == -1 or bpm(match_to[j], match_to, visited, adj):
                match_to[j] = i
                return True
    return False

def max_regular_bipartite_matching(list_lnode, list_rnode, partial_adj_left):
    rnode_to_idx = {rnode: idx for idx, rnode in enumerate(list_rnode)}
    partial_adj_idx = [[rnode_to_idx[rnode] for rnode in adj] for adj in partial_adj_left]
    match_to = [-1] * len(list_rnode)
    result = 0
    for i in range(len(list_lnode)):
        visited = [False] * len(list_rnode)
        if bpm(i, match_to, visited, partial_adj_idx):
            result += 1
    if result != len(list_lnode):
        raise ValueError('matching not maximum')
    return match_to

def regular_bipartite_edge_coloring(list_lnode, list_rnode, adj_left, edges, max_deg):
    tuple2id = {edge: i for i, edge in enumerate(edges)}
    color = [-1 for _ in range(len(edges))]
    for c in range(max_deg):
        partial_adj_left = []
        for i, adj_list in enumerate(adj_left):
            partial_list = [rnode for rnode in adj_list if color[tuple2id[(list_lnode[i], rnode)]] == -1]
            partial_adj_left.append(partial_list)
        match_to = max_regular_bipartite_matching(list_lnode, list_rnode, partial_adj_left)
        for i, lnode_id in enumerate(match_to):
           color[tuple2id[(list_lnode[lnode_id], list_rnode[i])]] = c
    return color

def bipartite_edge_coloring(num_node, num_node_left, original_adj_left, original_edges):
    if num_node_left != len(original_adj_left):
        raise ValueError('number of left nodes different from its adjacency list')
    if num_node_left < num_node - num_node_left:
        raise ValueError('left nodes are fewer than right nodes')

    if rx is not None and hasattr(rx, 'graph_bipartite_edge_color'):
        graph = rx.PyGraph(
            multigraph=True,
            node_count_hint=num_node,
            edge_count_hint=len(original_edges),
        )
        graph.add_nodes_from([None] * num_node)
        edge_indices = []
        for lnode, rnode in original_edges:
            if lnode < 0 or lnode >= num_node_left:
                raise ValueError(f'{lnode} is not a left node')
            if rnode < num_node_left or rnode >= num_node:
                raise ValueError(f'{rnode} is not a right node')
            edge_indices.append(graph.add_edge(lnode, rnode, None))
        edge_colors = rx.graph_bipartite_edge_color(graph)
        if edge_colors is not None:
            return [edge_colors[edge_idx] for edge_idx in edge_indices]

    edges = copy.deepcopy(original_edges)
    adj_left = copy.deepcopy(original_adj_left)
    num_true_node_left = num_node_left
    deg = [0 for _ in range(num_node)]
    for lnode in range(num_node_left):
        for rnode in adj_left[lnode]:
            if rnode < num_node_left or rnode >= num_node:
                raise ValueError(f'{rnode} is not a right node')
            deg[lnode] += 1
            deg[rnode] += 1
    max_deg = max(deg)
    num_true_edges = len(edges)

    list_lnode = [i for i in range(num_node_left)]
    list_rnode = [i for i in range(num_node_left, num_node)]
    diff = len(list_lnode) - len(list_rnode)
    for i in range(num_node, num_node + diff):
        list_rnode.append(i)
        deg.append(0)
        num_node += 1

    # try to color
    for i in list_lnode:
        while deg[i] < max_deg:
            available = False
            for j in list_rnode:
                if j not in adj_left[i] and deg[j] < max_deg:
                    available = True
                    edges.append((i,j))
                    deg[i] += 1
                    deg[j] += 1
                    adj_left[i].append(j)
                    break
            if not available:
                break
    while min(deg) < max_deg:
        lnode = deg[:num_true_node_left].index(min(deg[:num_true_node_left]))
        rnode = deg[num_true_node_left:].index(min(deg[num_true_node_left:])) + num_true_node_left
        adj_left += [[] for _ in range(max_deg)]
        
        edges.append((num_node, rnode))
        adj_left[num_node_left].append(rnode)
        deg[rnode] += 1
        edges.append((lnode, num_node+max_deg))
        adj_left[lnode].append(num_node+max_deg)
        deg[lnode] += 1
        for k in range(1, max_deg):
            edges.append((num_node+k, num_node+max_deg+k))
            adj_left[num_node_left+k].append(num_node+max_deg+k)
        for k in range(num_node, num_node+max_deg):
            for l in range(num_node+max_deg, num_node+2*max_deg):
                if l - k != max_deg:
                    edges.append((k,l))
                    adj_left[num_node_left+k-num_node].append(l)
        list_lnode += [i for i in range(num_node, num_node + max_deg)]
        list_rnode += [i for i in range(num_node + max_deg, num_node + 2*max_deg)]
        num_node += 2*max_deg
        num_node_left += max_deg

    return regular_bipartite_edge_coloring(list_lnode, list_rnode, adj_left, edges, max_deg)[:num_true_edges]

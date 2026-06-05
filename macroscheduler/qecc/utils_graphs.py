"""
A module for graph algorithms used in the QECC codebase.
This includes functions for bipartite edge coloring, which is used in the CSS code construction to
assign checks to time steps while respecting qubit-sharing constraints.
"""

try:
    import rustworkx as rx
except ImportError:  # pragma: no cover - optional dependency
    rx = None

def bpm(i, match_to, visited, adj, edge_left, edge_right):
    """Find an augmenting path for a left node in a bipartite graph.

    This is the DFS step used by the pure-Python edge-coloring fallback.
    ``adj`` is indexed by left-node position and contains edge IDs. Edges are
    represented by parallel ``edge_left`` and ``edge_right`` endpoint arrays,
    which lets the fallback handle multigraphs without collapsing parallel
    edges.

    Args:
        i: Left-node position to match.
        match_to: Current right-to-edge matching, mutated in place on success.
        visited: Right-node visit marks for the current DFS search.
        adj: Left adjacency lists using edge IDs.
        edge_left: Left endpoint for each edge ID.
        edge_right: Right endpoint for each edge ID.

    Returns:
        ``True`` if ``i`` can be matched by augmenting the current matching,
        otherwise ``False``.
    """
    for edge_id in adj[i]:
        j = edge_right[edge_id]
        if not visited[j]:
            visited[j] = True
            if match_to[j] == -1 or bpm(
                edge_left[match_to[j]],
                match_to,
                visited,
                adj,
                edge_left,
                edge_right,
            ):
                match_to[j] = edge_id
                return True
    return False

def max_regular_bipartite_matching(num_lnode, num_rnode, partial_adj_left, edge_left, edge_right):
    """Compute a perfect matching for a regular bipartite graph.

    The fallback edge-coloring algorithm repeatedly removes one perfect
    matching from a balanced regular bipartite multigraph. ``partial_adj_left``
    is indexed by left node and stores the still-uncolored edge IDs adjacent to
    that node.

    Args:
        num_lnode: Number of left nodes.
        num_rnode: Number of right nodes.
        partial_adj_left: ``partial_adj_left[i]`` lists uncolored edge IDs
            adjacent to left node ``i``.
        edge_left: Left endpoint for each edge ID.
        edge_right: Right endpoint for each edge ID.

    Returns:
        A list indexed by right-node position. Each entry is the position in
        ``edge_left``/``edge_right`` of the edge matched to that right node.

    Raises:
        ValueError: If the graph does not contain a matching covering every
            left node.
    """
    match_to = [-1] * num_rnode
    result = 0
    for i in range(num_lnode):
        visited = [False] * num_rnode
        if bpm(i, match_to, visited, partial_adj_left, edge_left, edge_right):
            result += 1
    if result != num_lnode:
        raise ValueError('matching not maximum')
    return match_to

def regular_bipartite_edge_coloring(num_lnode, num_rnode, edge_left, edge_right, max_deg):
    """Color a balanced regular bipartite multigraph by perfect matchings.

    This is the core pure-Python replacement for rustworkx's bipartite edge
    coloring routine once ``bipartite_edge_coloring`` has padded the graph to
    be balanced and ``max_deg``-regular. By Konig's line-coloring theorem, each
    perfect matching can be assigned one color, and ``max_deg`` matchings color
    all edges.

    Args:
        num_lnode: Number of left nodes.
        num_rnode: Number of right nodes.
        edge_left: Left endpoint for each edge ID.
        edge_right: Right endpoint for each edge ID.
        max_deg: Regular degree of the padded bipartite graph and the number
            of colors to assign.

    Returns:
        A list of color indices aligned with ``edge_left``/``edge_right``.
    """
    if num_lnode != num_rnode:
        raise ValueError('regular bipartite edge coloring requires balanced partitions')

    adj_left = [[] for _ in range(num_lnode)]
    for edge_id, lnode in enumerate(edge_left):
        adj_left[lnode].append(edge_id)

    color = [-1 for _ in range(len(edge_left))]
    for c in range(max_deg):
        partial_adj_left = [
            [edge_id for edge_id in adj_list if color[edge_id] == -1]
            for adj_list in adj_left
        ]
        match_to = max_regular_bipartite_matching(
            num_lnode,
            num_rnode,
            partial_adj_left,
            edge_left,
            edge_right,
        )
        for edge_id in match_to:
            color[edge_id] = c
    return color

def _fallback_bipartite_edge_coloring(num_node_left, num_node_right, edge_left, edge_right):
    """Color a bipartite multigraph with a pure-Python Delta-color fallback."""
    num_true_edges = len(edge_left)
    if num_true_edges == 0:
        return []

    num_balanced = max(num_node_left, num_node_right)
    deg_left = [0 for _ in range(num_balanced)]
    deg_right = [0 for _ in range(num_balanced)]
    for lnode, rnode in zip(edge_left, edge_right):
        deg_left[lnode] += 1
        deg_right[rnode] += 1

    max_deg = max(max(deg_left), max(deg_right))
    if max_deg == 0:
        return []

    padded_edge_left = list(edge_left)
    padded_edge_right = list(edge_right)

    lnode = 0
    rnode = 0
    while lnode < num_balanced and rnode < num_balanced:
        while lnode < num_balanced and deg_left[lnode] == max_deg:
            lnode += 1
        while rnode < num_balanced and deg_right[rnode] == max_deg:
            rnode += 1
        if lnode == num_balanced or rnode == num_balanced:
            break

        add_count = min(max_deg - deg_left[lnode], max_deg - deg_right[rnode])
        padded_edge_left.extend([lnode] * add_count)
        padded_edge_right.extend([rnode] * add_count)
        deg_left[lnode] += add_count
        deg_right[rnode] += add_count

    if any(deg != max_deg for deg in deg_left) or any(deg != max_deg for deg in deg_right):
        raise ValueError('failed to pad bipartite graph to regular form')

    return regular_bipartite_edge_coloring(
        num_balanced,
        num_balanced,
        padded_edge_left,
        padded_edge_right,
        max_deg,
    )[:num_true_edges]

def bipartite_edge_coloring(num_node, num_node_left, original_adj_left, original_edges):
    """Color edges of a bipartite graph for check-measurement scheduling.

    Nodes ``0`` through ``num_node_left - 1`` are the left partition, and nodes
    ``num_node_left`` through ``num_node - 1`` are the right partition. The
    returned color for each edge is a timestep: incident edges receive distinct
    colors, so no qubit/check conflict is scheduled in the same timestep.

    If rustworkx is available and provides ``graph_bipartite_edge_color``, this
    function delegates to it. Otherwise it uses a pure-Python fallback: copy
    the input into edge-ID arrays, add dummy vertices and edges when needed to
    balance and regularize the graph, and decompose that graph into perfect
    matchings. Only colors for the original edges are returned.

    Args:
        num_node: Number of original graph nodes.
        num_node_left: Number of left-partition nodes.
        original_adj_left: ``original_adj_left[i]`` lists right-node labels
            adjacent to left node ``i``.
        original_edges: Edge list in the order that colors should be returned.

    Returns:
        A list of nonnegative color indices aligned with ``original_edges``.

    Raises:
        ValueError: If the partition sizes or edge endpoints are invalid, or if
            the fallback cannot find a required perfect matching.
    """
    if num_node_left < 0 or num_node_left > num_node:
        raise ValueError('invalid number of left nodes')
    if num_node_left != len(original_adj_left):
        raise ValueError('number of left nodes different from its adjacency list')

    num_node_right = num_node - num_node_left
    for adj in original_adj_left:
        for rnode in adj:
            if rnode < num_node_left or rnode >= num_node:
                raise ValueError(f'{rnode} is not a right node')

    edge_left = []
    edge_right = []
    for lnode, rnode in original_edges:
        if lnode < 0 or lnode >= num_node_left:
            raise ValueError(f'{lnode} is not a left node')
        if rnode < num_node_left or rnode >= num_node:
            raise ValueError(f'{rnode} is not a right node')
        edge_left.append(lnode)
        edge_right.append(rnode - num_node_left)

    if rx is not None and hasattr(rx, 'graph_bipartite_edge_color'):
        graph = rx.PyGraph(
            multigraph=True,
            node_count_hint=num_node,
            edge_count_hint=len(original_edges),
        )
        graph.add_nodes_from([None] * num_node)
        edge_indices = []
        for lnode, rnode in original_edges:
            edge_indices.append(graph.add_edge(lnode, rnode, None))
        edge_colors = rx.graph_bipartite_edge_color(graph)
        if edge_colors is not None:
            return [edge_colors[edge_idx] for edge_idx in edge_indices]

    return _fallback_bipartite_edge_coloring(
        num_node_left,
        num_node_right,
        edge_left,
        edge_right,
    )

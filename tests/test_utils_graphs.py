from macroscheduler.qecc.utils_graphs import bipartite_edge_coloring


def _assert_valid_edge_coloring(edges, colors):
    assert len(colors) == len(edges)
    for i, (left_i, right_i) in enumerate(edges):
        for j in range(i + 1, len(edges)):
            left_j, right_j = edges[j]
            if left_i == left_j or right_i == right_j:
                assert colors[i] != colors[j]


def test_bipartite_edge_coloring_handles_parallel_edges():
    edges = [(0, 1), (0, 1), (0, 1)]
    colors = bipartite_edge_coloring(
        num_node=2,
        num_node_left=1,
        original_adj_left=[[1, 1, 1]],
        original_edges=edges,
    )

    _assert_valid_edge_coloring(edges, colors)
    assert len(set(colors)) == 3


def test_bipartite_edge_coloring_allows_left_partition_smaller_than_right():
    edges = [(0, 2), (0, 3), (0, 4), (1, 2)]
    colors = bipartite_edge_coloring(
        num_node=5,
        num_node_left=2,
        original_adj_left=[[2, 3, 4], [2]],
        original_edges=edges,
    )

    _assert_valid_edge_coloring(edges, colors)
    assert max(colors) + 1 == 3


def test_bipartite_edge_coloring_preserves_simple_graph_behavior():
    edges = [(0, 2), (0, 3), (1, 2), (1, 3)]
    colors = bipartite_edge_coloring(
        num_node=4,
        num_node_left=2,
        original_adj_left=[[2, 3], [2, 3]],
        original_edges=edges,
    )

    _assert_valid_edge_coloring(edges, colors)
    assert max(colors) + 1 == 2

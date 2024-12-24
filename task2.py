import sys
import time
import copy
from collections import defaultdict
from pyspark import SparkConf, SparkContext

conf = SparkConf().setAppName("Task 2")
sc = SparkContext(conf=conf).getOrCreate()
sc.setLogLevel("ERROR")

def read_csv(spark, file_path):
    """
    Read CSV file into RDD and parse it into tuples of (user_id, business_id)
    """
    rdd = sc.textFile(file_path)

    # Optionally, skip the header (if applicable)
    header = rdd.first()  # get the header
    rdd_data = rdd.filter(lambda row: row != header)  # remove header

    # Now you can split the rows into columns (e.g., using comma as delimiter)
    data = rdd_data.map(lambda row: row.split(','))

    
    return data


def construct_graph(data, filter_threshold: int):
    """
    Construct the graph by creating edges based on shared businesses.
    """
    # Create user to business map
    user2bus = data.groupByKey().mapValues(set)

    # Create edges RDD (user pairs with their common businesses)
    edges = (
        user2bus.cartesian(user2bus)  # Pairwise cartesian product
        .filter(lambda pair: pair[0][0] != pair[1][0])  # Filter self-pairs
        .map(lambda pair: ((pair[0][0], pair[1][0]), len(pair[0][1] & pair[1][1])))  # Count common businesses
        .filter(lambda data: data[1] >= filter_threshold)  # Apply threshold
        .map(lambda data: (data[0][0], data[0][1]))  # Extract user pairs
        .distinct()  # Remove duplicate edges
    )

    # Create adjacency matrix (user -> set of neighbors)
    adj_mat = (
        edges
        .groupByKey()
        .mapValues(set)
        .cache()
    )

    return adj_mat.collectAsMap()


def bfs(graph, source):
    """
    Perform BFS to calculate shortest paths and number of shortest paths.
    """
    parent = defaultdict(set)
    level = {source: 0}
    num_shortest_paths = defaultdict(float)
    num_shortest_paths[source] = 1
    visited = {source}
    queue = [source]
    path=[]
    
    while queue:
        current_node = queue.pop(0)
        path.append(current_node)
        for neighbor in graph.get(current_node, []):
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                parent[neighbor].add(current_node)
                num_shortest_paths[neighbor] += num_shortest_paths[current_node]
                level[neighbor] = level[current_node] + 1
            elif level[neighbor] == level[current_node] + 1:
                parent[neighbor].add(current_node)
                num_shortest_paths[neighbor] += num_shortest_paths[current_node]
    
    return parent, num_shortest_paths, path


def accumulate_edge_weights(path, parent, num_shortest_paths):
    """
    Calculate edge betweenness based on shortest path contributions.
    """
    node_weights = {node: 1 for node in reversed(path)}
    edge_weights = defaultdict(float)

    for node in reversed(path):
        for parent_node in parent[node]:
            temp_weight = node_weights[node] * (num_shortest_paths[parent_node] / num_shortest_paths[node])
            node_weights[parent_node] += temp_weight
            edge = tuple(sorted([node, parent_node]))
            edge_weights[edge] += temp_weight / 2
    
    return edge_weights


def calculate_betweenness(adj_mat):
    """
    Calculate betweenness centrality for each edge in the graph.
    """
    betweenness = defaultdict(float)
    
    for node in adj_mat:
        parent, num_shortest_paths,path = bfs(adj_mat, node)
        #print(parent)
        #print(num_shortest_paths)
        ac=accumulate_edge_weights(path, parent, num_shortest_paths).items()
        for edge, weight in ac:
            betweenness[edge] += weight
    
    return sorted(betweenness.items(), key=lambda x: (-x[1], x[0]))


def girvan_newman(graph, betweenness):
    """
    Apply Girvan-Newman algorithm to detect communities.
    """
    def dfs(graph, node, visited, component):
        """
        Depth-first search to find a connected component iteratively.
        """
        stack = [node]
        while stack:
            current_node = stack.pop()
            if current_node not in visited:
                visited.add(current_node)
                component.append(current_node)
                for neighbor in graph.get(current_node, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
        return visited,component



    def remove_edges(graph, edges):
        for u, v in edges:
            if u in graph:
                graph[u].remove(v)
            if v in graph:
                graph[v].remove(u)

    def calculate_modularity(graph, communities, original_graph, n_edge):
        modularity = 0.0
        for community in communities:
            for p in community:
                for q in community:
                    A = 1 if q in graph[p] and p in graph[q] else 0
                    k_i_k_j = len(original_graph[p]) * len(original_graph[q])
                    modularity += A - k_i_k_j / (2.0 * n_edge)
        return modularity / (2.0 * n_edge)

    sub_graph = copy.deepcopy(graph)
    n_edge = sum(len(neighbors) for neighbors in sub_graph.values()) / 2  # Total edges
    original_graph = copy.deepcopy(graph)
    max_modularity = -float("inf")
    result = []

    while betweenness:
        candidates = []
        visited_nodes = set()

        # Find connected components in the sub-graph
        for node in sub_graph:
            if node not in visited_nodes:
                component = []
                visited_nodes,component=dfs(sub_graph, node, visited_nodes, component)
                candidates.append(component)

        modularity = calculate_modularity(sub_graph, candidates, original_graph, n_edge)
        if modularity > max_modularity:
            max_modularity = modularity
            result = copy.deepcopy(candidates)

        # Remove the edges with highest betweenness
        highest_betweenness = betweenness[0][1]
        pruned_edges = [edge[0] for edge in betweenness if edge[1] >= highest_betweenness]
        remove_edges(sub_graph, pruned_edges)

        betweenness = calculate_betweenness(sub_graph)

    result = sorted(result, key=lambda x: (len(x), sorted(x)))
    return result


def save_data(path, data, format_fn):
    with open(path, "w") as f:
        for item in data:
            f.write(format_fn(item))


def main(filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path):
    """
    Task 2: Calculate betweenness centrality and detect communities.
    """


    #try:
    start_time = time.time()
    data = read_csv(sc, input_file_path)

    adj_mat = construct_graph(data, filter_threshold)

    betweenness = calculate_betweenness(adj_mat)
    save_data(betweenness_output_file_path, betweenness, lambda x: f"{x[0]},{x[1]:.5f}\n")
    #print(betweenness)
    #sys.exit()
    communities = girvan_newman(adj_mat, betweenness)
    save_data(community_output_file_path, communities, lambda x: "{" + ", ".join(map(str, x)) + "}\n")

    execution_time = time.time() - start_time
    print(f"Duration: {execution_time:.2f} seconds")

    #finally:
    sc.stop()


if __name__ == "__main__":

    filter_threshold = int(sys.argv[1])
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    main(filter_threshold, input_file_path, betweenness_output_file_path, community_output_file_path)

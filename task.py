import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis
import argparse
import random
import sys

def load_data(input_file):
    """Load and preprocess the dataset."""
    data = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            items = list(map(float, line.strip().split(',')))
            data.append(items)
    return np.array(data)

def save_output(output_file, intermediate_results, clustering_results):
    """Save the output to a file."""
    with open(output_file, 'w') as f:
        f.write("The intermediate results\n")
        for line in intermediate_results:
            f.write(line + '\n')
        f.write("\n")
        f.write("The clustering results\n")
        for idx, cluster in clustering_results:
            f.write(f"{idx},{cluster}\n")

def mahalanobis_distance(point, centroid, cov_matrix):
    """Calculate the Mahalanobis distance with regularization to handle singular matrices."""
    try:
        cov_inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        # Add small regularization to the diagonal to make the matrix invertible
        cov_inv = np.linalg.inv(cov_matrix + 1e-6 * np.eye(cov_matrix.shape[0]))
    return mahalanobis(point, centroid, cov_inv)

def update_statistics(cluster_stats, points):
    """Update cluster statistics for DS and CS."""
    cluster_stats['N'] += len(points)
    cluster_stats['SUM'] += np.sum(points, axis=0)
    cluster_stats['SUMSQ'] += np.sum(np.square(points), axis=0)

def run_kmeans(data, n_clusters, init='k-means++'):
    """Run KMeans clustering on the given data."""
    kmeans = KMeans(n_clusters=n_clusters, init=init)
    kmeans.fit(data)
    return kmeans

def initialize_bfr(data, n_clusters):
    """Initialize the BFR clustering algorithm."""
    n_samples = data.shape[0]
    sample_size = int(0.2 * n_samples)
    sampled_data = data[np.random.choice(n_samples, sample_size, replace=False)]

    # Step 2: Run KMeans with large K
    kmeans = run_kmeans(sampled_data, n_clusters * 5)

    # Step 3: Move clusters with only 1 point to RS
    RS_points = []
    for i, label in enumerate(kmeans.labels_):
        if np.sum(kmeans.labels_ == label) == 1:
            RS_points.append(sampled_data[i])
    
    RS_points = np.array(RS_points)
    
    # Step 4: Run KMeans with K = n_clusters
    kmeans = run_kmeans(sampled_data, n_clusters)
    
    return kmeans, RS_points, kmeans.labels_

def process_batch(data, n_clusters, DS_clusters, CS_clusters, RS_points, kmeans_labels):
    """Process each batch of data."""
    discard_points = []
    compression_points = []
    retained_points = []

    # Track points for DS, CS, RS
    for i in range(len(data)):
        point = data[i]
        label = kmeans_labels[i]

        # If the point belongs to a discard cluster
        if label in DS_clusters:
            discard_points.append(point)
        elif label in CS_clusters:
            compression_points.append(point)
        else:
            retained_points.append(point)
    
    # Return stats on points
    return len(discard_points), len(CS_clusters), len(compression_points), len(retained_points)

def merge_clusters(DS_clusters, CS_clusters, d_threshold=2):
    """Merge CS clusters that are close to DS."""
    to_merge = []
    for cs_cluster in CS_clusters:
        for ds_cluster in DS_clusters:
            # Calculate Mahalanobis distance between CS and DS clusters
            distance = mahalanobis_distance(cs_cluster, ds_cluster.mean(), np.linalg.inv(ds_cluster.covariance_))
            if distance < d_threshold:
                to_merge.append((cs_cluster, ds_cluster))
    
    return to_merge

def iterative_bfr(data, n_clusters, DS_clusters, CS_clusters, RS_points):
    """Iterative process to handle batches of data."""
    batch_data = data[np.random.choice(len(data), int(0.2 * len(data)), replace=False)]
    
    # Assign points to nearest DS and CS clusters
    for point in batch_data:
        # Mahalanobis distance assignment logic for DS and CS
        pass  # Implement logic here
    
    return DS_clusters, CS_clusters, RS_points

def bfr_clustering(input_file, n_clusters, output_file):
    data = load_data(input_file)
    total_data_points = len(data)
    dimensions = data.shape[1] - 2  # Exclude index and cluster label columns
    chunk_size = total_data_points // 5  # Load 20% of data each round
    random.shuffle(data)
    
    DS = {}  # Discard Set
    CS = {}  # Compression Set
    RS = []  # Retained Set

    intermediate_results = []
    clustering_results = []

    for round_idx in range(5):  # Load data in chunks
        start_idx = round_idx * chunk_size
        end_idx = (round_idx + 1) * chunk_size if round_idx < 4 else total_data_points
        chunk = data[start_idx:end_idx, 2:]  # Exclude index and cluster labels

        if round_idx == 0:
            # Step 2: Initial K-Means with large K
            kmeans = KMeans(n_clusters=n_clusters * 5, random_state=42).fit(chunk)
            labels = kmeans.labels_
            for i, point in enumerate(chunk):
                if np.sum(labels == labels[i]) == 1:
                    RS.append(point)
                else:
                    cluster_id = labels[i]
                    if cluster_id not in DS:
                        DS[cluster_id] = {'N': 0, 'SUM': np.zeros(dimensions), 'SUMSQ': np.zeros(dimensions)}
                    update_statistics(DS[cluster_id], [point])
        else:
            # Step 8: Assign new points to DS using Mahalanobis distance
            new_DS = []
            for point in chunk:
                assigned = False
                for cluster_id, stats in DS.items():
                    centroid = stats['SUM'] / stats['N']
                    cov_matrix = np.diag(stats['SUMSQ'] / stats['N'] - np.square(centroid))
                    dist = mahalanobis_distance(point, centroid, cov_matrix)
                    if dist < 2 * np.sqrt(dimensions):
                        update_statistics(DS[cluster_id], [point])
                        assigned = True
                        break
                if not assigned:
                    RS.append(point)

            # Step 9-10: Handle points not assigned to DS or CS
            if RS:
                rs_kmeans = KMeans(n_clusters=n_clusters * 5, random_state=42).fit(RS)
                rs_labels = rs_kmeans.labels_
                new_RS = []
                for i, point in enumerate(RS):
                    label = rs_labels[i]
                    if np.sum(rs_labels == label) > 1:
                        if label not in CS:
                            CS[label] = {'N': 0, 'SUM': np.zeros(dimensions), 'SUMSQ': np.zeros(dimensions)}
                        update_statistics(CS[label], [point])
                    else:
                        new_RS.append(point)
                RS = new_RS

        # Merge CS clusters with Mahalanobis distance
        merged_CS = {}
        for cluster_id, stats in CS.items():
            centroid = stats['SUM'] / stats['N']
            for other_id, other_stats in merged_CS.items():
                other_centroid = other_stats['SUM'] / other_stats['N']
                cov_matrix = np.diag(other_stats['SUMSQ'] / other_stats['N'] - np.square(other_centroid))
                dist = mahalanobis_distance(centroid, other_centroid, cov_matrix)
                if dist < 2 * np.sqrt(dimensions):
                    update_statistics(merged_CS[other_id], [centroid])
                    break
            else:
                merged_CS[cluster_id] = stats
        CS = merged_CS

        # Save intermediate results
        intermediate_results.append(f"Round {round_idx + 1}: {sum(ds['N'] for ds in DS.values())}, "
                                    f"{len(CS)}, {sum(cs['N'] for cs in CS.values())}, {len(RS)}")

    # Final clustering result
    for cluster_id, stats in DS.items():
        centroid = stats['SUM'] / stats['N']
        for point in data:
            dist = mahalanobis_distance(point[2:], centroid, np.diag(stats['SUMSQ'] / stats['N'] - np.square(centroid)))
            if dist < 2 * np.sqrt(dimensions):
                clustering_results.append((int(point[0]), cluster_id))
                break
        else:
            clustering_results.append((int(point[0]), -1))  # Outliers

    save_output(output_file, intermediate_results, clustering_results)

def parse_arguments():
    parser = argparse.ArgumentParser(description="BFR Clustering Algorithm")
    parser.add_argument("input_file", help="Path to the input data file")
    parser.add_argument("n_clusters", type=int, help="Number of clusters")
    parser.add_argument("output_file", help="Path to save the output results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    bfr_clustering(args.input_file, args.n_clusters, args.output_file)

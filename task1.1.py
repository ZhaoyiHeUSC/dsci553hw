import sys
from pyspark import SparkContext
from itertools import combinations
from collections import defaultdict

# A-Priori algorithm for generating candidate itemsets
def apriori(chunk, support_threshold, k):
    counts = defaultdict(int)
    
    # For k=1, count singletons
    if k == 1:
        for basket in chunk:
            for item in basket:
                counts[frozenset([item])] += 1
    else:
        # For k>1, generate itemsets and count them
        for basket in chunk:
            for itemset in combinations(basket, k):
                counts[frozenset(itemset)] += 1

    # Return candidates that meet the support threshold
    return [itemset for itemset, count in counts.items() if count >= support_threshold]

# SON algorithm with A-Priori for Phase 1
def son_with_apriori(baskets, support_threshold, k):
    total_baskets = baskets.count()
    partitioned_baskets = baskets.mapPartitions(lambda chunk: apriori(chunk, support_threshold / total_baskets, k))
    return partitioned_baskets.distinct()

# Validate candidates globally (Phase 2)
def validate_candidates(baskets, candidates, support_threshold):
    counts = defaultdict(int)
    
    # Count how often each candidate itemset appears in the full dataset
    for basket in baskets.collect():
        for candidate in candidates:
            if set(candidate).issubset(basket):
                counts[frozenset(candidate)] += 1

    # Return itemsets that meet the global support threshold
    return [itemset for itemset, count in counts.items() if count >= support_threshold]

def main():
    if len(sys.argv) != 5:
        print("Use: task1.py <case number> <support> <input_file_path> <output_file_path>")
        sys.exit(1)

    case_number = int(sys.argv[1])  # Case number: should be 1 (A-Priori in this case)
    support_threshold = int(sys.argv[2])  # Support threshold
    input_file = sys.argv[3]  # Path to input CSV file
    output_file = sys.argv[4]  # Path to output file

    # Initialize SparkContext
    sc = SparkContext("local", "SON Algorithm")

    # Load data
    data = sc.textFile(input_file)

    # Remove header and split lines into (user_id, business_id)
    baskets = data.filter(lambda line: not line.startswith('user_id')) \
                  .map(lambda line: line.split(',')) \
                  .groupByKey() \
                  .mapValues(set)  # Create unique baskets for each user
    
    # Cache the baskets RDD as it will be reused
    baskets.cache()

    # SON Phase 1: A-Priori for k=1 (singletons)
    candidates_k1 = son_with_apriori(baskets, support_threshold, k=1)
    
    # SON Phase 2: Validate globally
    valid_singletons = validate_candidates(baskets, candidates_k1.collect(), support_threshold)

    # SON Phase 1: A-Priori for k=2 (pairs)
    candidates_k2 = son_with_apriori(baskets, support_threshold, k=2)
    
    # SON Phase 2: Validate globally
    valid_pairs = validate_candidates(baskets, candidates_k2.collect(), support_threshold)

    # Write results to the output file
    with open(output_file, 'w') as f:
        f.write("Frequent Singletons:\n")
        for itemset in valid_singletons:
            f.write(f"{set(itemset)}\n")
        
        f.write("\nFrequent Pairs:\n")
        for itemset in valid_pairs:
            f.write(f"{set(itemset)}\n")

    print(f"Results written to {output_file}")

if __name__ == "__main__":
    main()

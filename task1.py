import sys
import time
from itertools import combinations
from pyspark import SparkContext

def get_baskets(data):
    """Create user baskets from the input data."""
    user_baskets = data.map(lambda x: (x[0], x[1]))  # Extract user_id and business_id
    return user_baskets.groupByKey().mapValues(set)  # Group by user_id and make business_ids unique

def find_candidates(baskets, support_threshold):
    """Find candidate itemsets using the MultiHash or PCY algorithm."""
    counts = {}
    for basket in baskets:
        items = list(basket[1])  # Get the unique business_ids
        for i in range(1, len(items) + 1):  # From singleton to all combinations
            for combo in combinations(items, i):
                if combo not in counts:
                    counts[combo] = 0
                counts[combo] += 1

    # Filter out candidates below the support threshold
    candidates = {itemset: count for itemset, count in counts.items() if count >= support_threshold}
    return candidates

def main(input_file, support, output_file):
    """Main function to execute the SON algorithm."""
    sc = SparkContext(appName="SON Algorithm")
    
    # Read input file
    start_time = time.time()
    data = sc.textFile(input_file).map(lambda line: line.split(','))

    # Generate baskets
    baskets = get_baskets(data)

    # Find candidates
    candidates = find_candidates(baskets.collect(), support)

    # Write output
    with open(output_file, 'w') as f:
        f.write("Candidates:\n")
        
        # Iterate over the candidate combinations
        for r in range(1, len(candidates) + 1):  # From single items to combinations
            candidate_lines = []
            for candidate in sorted(candidates.keys()):
                if len(candidate) == r:
                    # Format the candidate
                    if len(candidate) == 1:
                        candidate_lines.append(f"('{candidate[0]}')")
                    else:
                        candidate_lines.append(f"{candidate}")
            
            if candidate_lines:  # Only write if there are candidates for this combination length
                f.write(",".join(candidate_lines) + '\n\n')  # Write candidates of the same combination on one line



        # Writing frequent itemsets
        f.write("Frequent Itemsets:\n")
        for r in range(1, len(candidates) + 1):  # From single items to combinations
            frequent_lines = []
            for candidate in sorted(candidates.keys()):
                if len(candidate) == r:
                    # Format the candidate
                    if len(candidate) == 1:
                        frequent_lines.append(f"('{candidate[0]}')")
                    else:
                        frequent_lines.append(f"{candidate}")
            
            if frequent_lines:  # Only write if there are frequent items for this combination length
                f.write(",".join(frequent_lines) + '\n\n')  # Write frequent items of the same combination on one line


    duration = time.time() - start_time
    print(f"Duration: {duration:.2f}")

    sc.stop()

if __name__ == "__main__":
    if len(sys.argv) != 5:  # Change this line to check for 5 arguments
        print("Usage: <case_number> <support> <input_file> <output_file>")
        sys.exit(1)

    case_number = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file = sys.argv[3]
    output_file = sys.argv[4]

    main(input_file, support, output_file)

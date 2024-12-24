import hashlib
import random
import numpy as np
import csv

def myhashs(user_id):
    """Generate a list of hash values for a given user_id using different hash functions."""
    hash_list = []
    
    # Hash function 1: MD5
    hash_value1 = int(hashlib.md5(user_id.encode('utf-8')).hexdigest(), 16)
    hash_list.append(hash_value1)
    
    # Hash function 2: SHA-1
    hash_value2 = int(hashlib.sha1(user_id.encode('utf-8')).hexdigest(), 16)
    hash_list.append(hash_value2)
    
    # Hash function 3: SHA-256
    hash_value3 = int(hashlib.sha256(user_id.encode('utf-8')).hexdigest(), 16)
    hash_list.append(hash_value3)
    
    return hash_list

class FlajoletMartin:
    def __init__(self, num_hashes=3):
        self.num_hashes = num_hashes
        self.max_zeros = [0] * num_hashes

    def add(self, user_id):
        # Obtain hash values for a given user ID
        hash_values = myhashs(user_id)
        
        # Update the max trailing zeros for each hash function
        for i in range(self.num_hashes):
            trailing_zeros = self._count_trailing_zeros(hash_values[i])
            self.max_zeros[i] = max(self.max_zeros[i], trailing_zeros)

    def estimate(self):
        # Combine the estimates from all hash functions
        estimates = [2 ** zeros for zeros in self.max_zeros]
        return np.mean(estimates)

    @staticmethod
    def _count_trailing_zeros(n):
        """Count the number of trailing zeros in the binary representation of a number."""
        if n == 0:
            return 0
        count = 0
        while (n & 1) == 0:
            count += 1
            n >>= 1
        return count

# Main execution
def main():
    stream_size = 300
    num_of_asks = 30
    results = []

    for batch in range(num_of_asks):
        flajolet_martin = FlajoletMartin()
        user_set = set()

        # Simulate a stream of user IDs
        for _ in range(stream_size):
            user_id = f'user_{random.randint(1, 1000)}'
            flajolet_martin.add(user_id)
            user_set.add(user_id)

        ground_truth = len(user_set)
        estimation = flajolet_martin.estimate()
        results.append((batch, ground_truth, estimation))

    # Save results to CSV
    with open('flajolet_martin_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Time', 'Ground Truth', 'Estimation'])
        writer.writerows(results)

    # Calculate and print the ratio for validation
    total_ground_truth = sum(row[1] for row in results)
    total_estimations = sum(row[2] for row in results)
    ratio = total_estimations / total_ground_truth if total_ground_truth > 0 else 0

    if 0.2 <= ratio <= 5:
        print("Ratio is within acceptable limits:", ratio)
    else:
        print("Ratio is out of acceptable limits:", ratio)

if __name__ == "__main__":
    main()

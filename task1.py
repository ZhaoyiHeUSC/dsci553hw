import binascii
import random
import csv
import time

class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = [0] * size
        self.hash_params = self._generate_hash_params()

    def _generate_hash_params(self):
        # Using a small set of prime numbers for hash function parameters
        primes = [101, 103, 107, 109, 113, 127]
        return [(random.randint(1, 100), random.randint(0, 100), random.choice(primes)) for _ in range(self.num_hashes)]

    def _convert_to_int(self, user_id):
        return int(binascii.hexlify(user_id.encode('utf-8')), 16)

    def myhashs(self, user_id):
        item = self._convert_to_int(user_id)
        hash_values = []
        for a, b, p in self.hash_params:
            hash_values.append(self._hash(item, a, b, p))
        return hash_values

    def _hash(self, item, a, b, p):
        return ((a * item + b) % p) % self.size

    def _hash_and_set(self, item):
        for hash_val in self.myhashs(item):
            self.bit_array[hash_val] = 1

    def add(self, user_id):
        self._hash_and_set(user_id)

    def check(self, user_id):
        return all(self.bit_array[hash_val] for hash_val in self.myhashs(user_id))


def evaluate_false_positive_rate(bloom_filter, test_user_ids):
    false_positives = sum(1 for user_id in test_user_ids if bloom_filter.check(user_id))
    fpr = false_positives / len(test_user_ids)
    return fpr


def simulate_bloom_filter(bit_array_length, num_hash_functions, stream_size, num_of_asks, output_file="bloom_filter_results.csv"):
    bloom_filter = BloomFilter(size=bit_array_length, num_hashes=num_hash_functions)
    previous_user_set = set()
    results = []
    
    for batch_index in range(num_of_asks):
        stream = [f"user_{random.randint(1, 1000)}" for _ in range(stream_size)]
        
        # Evaluate false positives for users not in previous_user_set
        test_user_ids = [f"test_user_{random.randint(1001, 2000)}" for _ in range(stream_size)]
        false_positive_count = sum(1 for user_id in test_user_ids if bloom_filter.check(user_id) and user_id not in previous_user_set)

        for user_id in stream:
            bloom_filter.add(user_id)
            previous_user_set.add(user_id)

        fpr = false_positive_count / stream_size
        elapsed_time = time.time() - start_time
        results.append((elapsed_time, fpr))

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Time", "FPR"])
        writer.writerows(results)

def main():
    # Set parameters for simulation
    bit_array_length = 69997
    num_hash_functions = 5
    stream_size = 100
    num_of_asks = 30

    # Start the simulation
    simulate_bloom_filter(bit_array_length, num_hash_functions, stream_size, num_of_asks)

if __name__ == "__main__":
    start_time = time.time()
    main()
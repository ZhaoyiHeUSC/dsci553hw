import random
import csv

# Reservoir Sampling Function
def reservoir_sampling(stream, reservoir_size=100):
    reservoir = []  # Initialize an empty reservoir
    global_sequence = 0  # Keep track of the sequence number
    
    # Open a CSV file to save the results
    with open("reservoir_sampling_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        
        # Process each user in the stream
        for user in stream:
            global_sequence += 1
            
            if global_sequence <= reservoir_size:
                # Fill the reservoir with the first 100 users
                reservoir.append(user)
            else:
                # Decide whether to replace a user in the reservoir
                probability = reservoir_size / global_sequence
                if random.random() < probability:
                    # Randomly pick an index to replace
                    replace_idx = random.randint(0, reservoir_size - 1)
                    reservoir[replace_idx] = user
            
            # After processing each batch of 100 users, save the reservoir state
            if global_sequence % reservoir_size == 0:
                writer.writerow([
                    global_sequence,
                    reservoir[0],
                    reservoir[20],
                    reservoir[40],
                    reservoir[60],
                    reservoir[80]
                ])
    
    print("Reservoir sampling results saved to reservoir_sampling_results.csv")

# Main function
def main():
    random.seed(553)  # Seed the random generator
    stream_size = 100
    num_of_asks = 30
    num_users = 1000  # Simulated number of unique users
    
    for _ in range(num_of_asks):
        # Generate a stream of 100 users
        stream = [f"user{random.randint(1, num_users)}" for _ in range(stream_size)]
        reservoir_sampling(stream)

if __name__ == "__main__":
    main()

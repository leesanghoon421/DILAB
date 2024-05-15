# Load the sequences
with open('processed_data.txt', 'r') as f:
    raw_sequences = [line.strip() for line in f]

# Remove duplicates
unique_sequences = list(set(raw_sequences))

# Save the unique sequences to a new file
with open('new_processed_data.txt', 'w') as f:
    for sequence in unique_sequences:
        f.write(f'{sequence}\n')
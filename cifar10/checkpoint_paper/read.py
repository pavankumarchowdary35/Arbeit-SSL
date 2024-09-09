import numpy as np

# Load the numpy file
labelled_indexes = np.load('checkpoint_paper/sampled_label_idx_4000.npy')

# Print the labelled indexes
print("Labelled Indexes:")
print(len(labelled_indexes))
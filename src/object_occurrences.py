from glob import glob
import os

labels_paths = glob(os.path.join('..', 'dataset', 'labels', '*.txt'))

occurrences = {
    'car': 0,
    'person': 0,
    'cyclist': 0
}

for file in labels_paths:
    with open(file, 'r') as f:
        for line in f:
            if line.startswith('Car'):
                occurrences['car'] += 1
            elif line.startswith('Cyclist'):
                occurrences['cyclist'] += 1
            elif line.startswith('Person') or line.startswith('Person_sitting'):
                occurrences['person'] += 1

print(occurrences)
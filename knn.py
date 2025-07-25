from collections import Counter
from dis import dis
from math import *

def normalize_data(train, test):
    num_features = len(train[0])
    min_vals = [min(row[i] for row in train) for i in range(num_features)]
    max_vals = [max(row[i] for row in train) for i in range(num_features)]

    def scale(dataset):
        scaled = []
        for row in dataset:
            scaled_row = []
            for i in range(num_features):
                if max_vals[i] == min_vals[i]:
                    scaled_val = 0
                else:
                    scaled_val = (row[i] - min_vals[i]) / (max_vals[i] - min_vals[i])
                scaled_row.append(scaled_val)
            scaled.append(scaled_row)
        return scaled

    return scale(train), scale(test)



def get_distances(point, data):
    distance = []
    ans = 0

    for d in data:
        for i in range(len(point)):
            ans += (d[i] - point[i])**2
        
        ans = sqrt(ans)
        distance.append(ans)

    return distance



def run_knn(train_set, test_set, k):
    train_set, test_set = normalize_data(train_set, test_set)

    predictions = []
    actual_labels = []

    for test_point in test_set:
        test_features = test_point[:-1]
        true_label = test_point[-1]

        distances = get_distances(test_features, [row[:-1] for row in train_set])

        labeled_distances = list(zip(distances, [row[-1] for row in train_set]))

        labeled_distances.sort(key=lambda x: x[0])
        top_k = labeled_distances[:k]

        labels = [label for _, label in top_k]
        pred_label = Counter(labels).most_common(1)[0][0]

        predictions.append(pred_label)
        actual_labels.append(true_label)

    return predictions, actual_labels



def run_CV(data, k=3, folds=5):
    fold_size = len(data) // folds
    total_accuracy = 0

    for i in range(folds):
        start = i * fold_size
        end = start + fold_size
        test_set = data[start:end]
        train_set = data[:start] + data[end:]

        preds, actuals = run_knn(train_set, test_set, k)

        correct = sum(1 for p, a in zip(preds, actuals) if p == a)
        accuracy = correct / len(test_set)
        total_accuracy += accuracy

    avg_accuracy = total_accuracy / folds
    print("Cross-validated accuracy:", avg_accuracy)
    return avg_accuracy


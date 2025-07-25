from cgitb import text
from hmac import new
from os import access
from random import uniform


# Make a prediction with weights
def classify(row, weights):
    sum = 0
    for i in range(0, len(row)-1):
        sum += row[i] * weights[i]

    sum += weights[-1]

    if sum < 0:
        return 0
    else:
        return 1
 
#Estimate Perceptron weights using stochastic gradient descent
def train(train_data, n_epoch, l_rate=1):
    initial_weights = []
    accuracy = 0

    for i in range(len(train_data)):
        initial_weights.append(uniform(-1, 1))

    cnt = 0
    while cnt < n_epoch:
        num_cnt = 0
        total_cnt = 0
        for i in train_data:
            row = i
            error=0
            predicted = classify(row, initial_weights)
            actual = row[-1]
            if not(predicted == actual):
                error = actual - predicted
                for j in range(0, len(row)):
                    initial_weights[j] = initial_weights[j] + l_rate*(error * row[j])
            else:
                error = 0
                num_cnt+=1
            
            total_cnt+=1

        accuracy += (num_cnt/total_cnt)*100
        #print(f"Epoch {cnt} .... {accuracy:.1f}% correct.")
        f_accuracy = accuracy / n_epoch
        cnt+=1

    return round(f_accuracy, 2)


def cross_validation(dataset, n_folds, n_epoch):
    increment = int(len(dataset) / n_folds)
    fold = []

    # creates the slides for the test data
    start = 0
    end = start + increment
    
    folds = 0
    while folds < n_folds:
        new_dataset = dataset
        for i in range(start, end):
            del new_dataset[i]
        start += end

        fold.append(train(new_dataset, n_epoch))
        end += increment

        folds+=1


    print("\nFolds:", fold)

    sum = 0
    for f in fold:
        sum += f
    
    print("Mean Accuracy:", str(sum/len(fold)) + "%")
        

    
        
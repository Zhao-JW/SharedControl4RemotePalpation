import csv
from matplotlib import projections
import matplotlib.pyplot as plt
import numpy as np

filename = 'csvdata\\palpation record'

def loaddata():
    file = open(filename+ '.csv')
    csvreader = csv.reader(file)
    header = next(csvreader)
    # press = np.array([])
    answers = []
    accs = []
    position = []
    answer = []
    rowcount = 0
    for row in csvreader:
        rowcount += 1
        
        if row[1] == "position":    #type
            for i in range(10):
                position.append(int(row[i+2]))
        elif row[1] == "answer":
            for i in range(10):
                answer.append(int(row[i+2]))
        if rowcount%2 == 0:
            accuracy = sum([position[i]-answer[i] == 0 for i in range(10)])/10
            accs.append(accuracy)
            answers.append([int(position[i]-answer[i] == 0) for i in range(10)])
            position = []
            answer = []
    return accs, answers

accs, answers = loaddata()
# print(accs)
shareoff = np.array(accs[:-2:2])
shareon = np.array(accs[1:-2:2])
# print(np.average(shareon), np.average(shareoff), np.std(shareon), np.std(shareoff))
learning = []
for i in range(14):
    if i%4 == 0:
        # attempt = answers[i]+answers[i+1]
        continue
    elif i%4 == 2:
        attempt = answers[i+1]+answers[i]
        # continue
    else:
        continue
    # print(attempt)
    # rolling = [sum(attempt[:4]), sum(attempt[4:8]), sum(attempt[8:12]), sum(attempt[12:16]), sum(attempt[16:])]
    rolling = [sum(attempt[i:i+5])/5 for i in range(15)]
    learning.append(rolling)

print(np.average(learning, axis=0), np.std(learning, axis=0))
for i in np.std(learning, axis=0):
    print(i)

# print(position, answer)
# accuracy = sum([position[i]-answer[i] == 0 for i in range(10)])/10
# print(accuracy)
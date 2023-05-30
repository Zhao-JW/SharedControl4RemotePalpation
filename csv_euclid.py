# import csv
# import csv_process
from csv_process import DataReader
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# leo1 test 5 invalid, jerry1 test 0 invalid, jerry0 test 7 invalid, fan1 test 3,4

names = ["fan", "jieying", "joe", "ycty", "yun","dd","leah","aleix", "tj", "leo"]
dates = ["feb-28", "mar-03", "mar-02", "mar-03", "mar-02", "mar-06", "mar-06", "mar-13", "mar-16", "mar-16"]
order = [0,        3,       2,      4,      1,   6,    5,      7,     8,     9]
invalids = [
    ([],[3,4]),
    ([],[1,5,6,7,8,9]),
    ([1,4],[1]),
    ([],[]),
    ([3,4,5,6,7,8],[0,1,2]),
    ([2,7],[7]),
    ([],[]),
    ([],[]),
    ([],[]),
    ([],[3])
]
# jieying 1-6, yun 0-5,7 1-1 invalid



def get_data():
    # get the distance to the palpation center
    dist0_pos = []
    dist1_pos = []
    dist0_pos_diff = []
    dist1_pos_diff = []

    for i, name in enumerate(names[:]):
        # continue
        participant = DataReader(name, dates[i])
        data = participant.readData(invalids[i])
        
        # dist0_pos.append(participant.euclidean_dist(0, isPos=True))
        # dist1_pos.append(participant.euclidean_dist(1, isPos=True))
        # dist0_orientation.append(participant.euclidean_dist(0, isPos=False))
        # dist1_orientation.append(participant.euclidean_dist(1, isPos=False))

        pos, diff = participant.euclidean_dist(0, isPos=True, forPlot=True)
        dist0_pos.append(pos)
        dist0_pos_diff.append(diff)

        pos,diff = participant.euclidean_dist(1, isPos=True, forPlot=True)
        dist1_pos.append(pos)
        dist1_pos_diff.append(diff)
        # dist0_orientation.append(participant.euclidean_dist(0, isPos=False, forPlot=True))
        # dist1_orientation.append(participant.euclidean_dist(1, isPos=False, forPlot=True))
    return dist0_pos, dist1_pos, dist0_pos_diff, dist1_pos_diff

def get_data2():
    # get the distance to ground truth
    dist0_pos = []
    dist1_pos = []
    dist0_orientation = []
    dist1_orientation = []

    for i, name in enumerate(names[:]):
        # continue
        participant = DataReader(name, dates[i])
        data = participant.readData(invalids[i])
        
        # dist0_pos.append(participant.euclidean_dist(0, isPos=True))
        # dist1_pos.append(participant.euclidean_dist(1, isPos=True))
        # dist0_orientation.append(participant.euclidean_dist(0, isPos=False))
        # dist1_orientation.append(participant.euclidean_dist(1, isPos=False))

        dist0 = participant.euclidean_dist(0, isPos=True, forPlot=False)
        dist0_pos.append(np.array(dist0).flatten())

        dist1 = participant.euclidean_dist(1, isPos=True, forPlot=False)
        dist1_pos.append(np.array(dist1).flatten())
        # dist0_orientation.append(participant.euclidean_dist(0, isPos=False, forPlot=True))
        # dist1_orientation.append(participant.euclidean_dist(1, isPos=False, forPlot=True))
    return dist0_pos, dist1_pos

def plot_scatter():
    # plot the average of all palpation
    fig= plt.figure()
    ax = fig.add_subplot()
    # ground_position = np.array([[-0.416, -0.074], [-0.330, 0.075], [-0.263, -0.152]])
    ground_position = np.array([[-0.423, -0.068], [-0.330, 0.079], [-0.250, -0.130]])
    c1 = plt.Circle((ground_position[0,0], ground_position[0,1]), 0.013, color='r', fill=False)
    c2 = plt.Circle((ground_position[1,0], ground_position[1,1]),0.013, color='g', fill=False)
    c3 = plt.Circle((ground_position[2,0], ground_position[2,1]), 0.013,color='b', fill=False)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)


    for participant in range(10):
        ax.scatter(np.array(dist0_pos)[participant,0,0], np.array(dist0_pos)[participant,0,1], color='r')
        ax.scatter(np.array(dist0_pos)[participant,1,0], np.array(dist0_pos)[participant,1,1], color='g')
        ax.scatter(np.array(dist0_pos)[participant,2,0], np.array(dist0_pos)[participant,2,1], color='b')
        # ave = np.average(dist1_pos_std[participant])
        # std = np.std(dist1_pos_std[participant])
        # ax.add_patch(plt.Circle((np.array(dist0_pos)[participant,0,0], np.array(dist0_pos)[participant,0,1]), ave, color='r', fill=False))

    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_xlim([-0.50, -0.20])
    ax.set_ylim([-0.15, 0.15])
    ax.set_xticks([-0.50, -0.40,-0.30,-0.20])
    ax.set_yticks([-0.15,-0.05,0.05,0.15])
    ax.set_title("Plapation spot position of all participants with full control")
    plt.show()


    fig= plt.figure()
    ax = fig.add_subplot()
    c1 = plt.Circle((ground_position[0,0], ground_position[0,1]), 0.013, color='r', fill=False)
    c2 = plt.Circle((ground_position[1,0], ground_position[1,1]),0.013, color='g', fill=False)
    c3 = plt.Circle((ground_position[2,0], ground_position[2,1]), 0.013,color='b', fill=False)
    ax.add_patch(c1)
    ax.add_patch(c2)
    ax.add_patch(c3)
    for participant in range(10):
        ax.scatter(np.array(dist1_pos)[participant,0,0], np.array(dist0_pos)[participant,0,1], color='r')
        ax.scatter(np.array(dist1_pos)[participant,1,0], np.array(dist0_pos)[participant,1,1], color='g')
        ax.scatter(np.array(dist1_pos)[participant,2,0], np.array(dist0_pos)[participant,2,1], color='b')
        # ave = np.average(dist1_pos_std[participant])
        # std = np.std(dist1_pos_std[participant])
        # ax.add_patch(plt.Circle((np.array(dist0_pos)[participant,0,0], np.array(dist0_pos)[participant,0,1]), ave, color='r', fill=False))

    ax.set_xlabel('X/m')
    ax.set_ylabel('Y/m')
    ax.set_xlim([-0.50, -0.20])
    ax.set_ylim([-0.15, 0.15])
    ax.set_xticks([-0.50, -0.40,-0.30,-0.20])
    ax.set_yticks([-0.15,-0.05,0.05,0.15])
    ax.set_title("Plapation spot position of all participants with shared control")
    plt.show()


def plot_compare(dist0_pos, dist1_pos):

    # take average like this because not matrix shape
    dist0_pos_ave = []
    dist1_pos_ave = []
    for row in dist0_pos:
        dist0_pos_ave.append(np.average(row))
    for row in dist1_pos:
        dist1_pos_ave.append(np.average(row))
    dist_ave = np.vstack((dist0_pos_ave, dist1_pos_ave))
    fig = plt.figure()
    (ax1, ax2) = fig.add_subplot(121)

    # position scatter plot
    for i in range(dist_ave.shape[1]):
        ax1.plot([0,1], dist_ave[:2,i])
        ax1.scatter([0,1], dist_ave[:2,i])
    ax1.set_xticks([-0.3,0,1,1.3], ["","Full\n control", "Shared\n control",""])
    ax1.set_yticks([-0.2,0,0.5,1])
    ax1.set_ylabel('Silhouette coefficient',labelpad=-10)
    ax1.title.set_text('Position accuracy')

    diff = dist_ave[1,:]-dist_ave[0,:]
    ax2.bar(0, np.mean(diff), yerr = np.std(diff), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax2.scatter([0]*10, diff, alpha=0.3)
    # ax11.set_ylabel('Improvement of Silhouette mark by shared control')
    ax2.set_xlim([-1,1])
    ax2.set_ylim([-0.2,1])
    ax2.set_xticks([0], ["Position accuracy\n improvement"])
    # ax11.set_title('P = 0.17')
    # ax.yaxis.grid(True)

    plt.show()

def p_calculate(dist0_pos_diff, dist1_pos_diff):
    order = [0,        3,       2,      4,      1,   6,    5,      7,     8,     9]
    # times
    label = [" time-full", " time-shared"]
    
    set0 = np.array([])
    set1 = np.array([])
    # fill invalid data with nan
    for participant in range(10):
        for point in range(3):
            row0 = dist0_pos_diff[participant][point]
            row1 = dist0_pos_diff[participant][point]
            if np.size(row0) < np.size(row1):
                n = np.size(row0)
            else:
                n = np.size(row1)
            set0 = np.concatenate((set0, row0[:n]))
            set1 = np.concatenate((set1, row1[:n]))


    print("time overall: ", stats.ttest_rel(set0,set1, nan_policy='omit').pvalue)



# dist0_pos, dist1_pos, dist0_pos_diff, dist1_pos_diff = get_data()
# plot_scatter()

# p_calculate(dist0_pos_diff, dist1_pos_diff)

dist0_pos, dist1_pos = get_data2()
plot_compare(dist0_pos, dist1_pos)

# list average over all participants
# print(np.average(np.array(dist0_pos), axis=0), np.average(np.array(dist1_pos), axis=0))

# list all the average and std
# for participant in range(10):
#     pos0 = (np.average(dist0_pos[participant]), np.std(dist0_pos[participant]))
#     pos1 = (np.average(dist1_pos[participant]), np.std(dist1_pos[participant]))
#     ori0 = (np.average(dist0_orientation[participant]), np.std(dist0_orientation[participant]))
#     ori1 = (np.average(dist1_orientation[participant]), np.std(dist1_orientation[participant]))
#     print("participant ", participant, ": ", pos0,pos1,ori0,ori1)

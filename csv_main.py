import csv
import csv_process
from csv_process import DataReader
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats



def loadtumordata():
    filename = 'csvdata\\palpation record'
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
    return [accs[::2],accs[1::2]], [answers[::2],answers[1::2]]

def plot_all(marks, times, tumors):
    fig,([ax1,ax11,_ax11,ax2,ax22], _, [ax3, ax33,_ax33, ax4, ax44]) = plt.subplots(3,5, gridspec_kw={'width_ratios': [2, 1, 0.1,2, 1], 'height_ratios': [1,0.1,1]})

    # turn off borders for space holders
    for _ax in _:
        _ax.axis('off')
    _ax11.axis('off')
    _ax33.axis('off')

    # position scatter plot
    for i in range(marks.shape[1]):
        ax1.plot([0,1], marks[:2,i])
        ax1.scatter([0,1], marks[:2,i])
    ax1.set_xticks([-0.3,0,1,1.3], ["","Full\n control", "Shared\n control",""])
    ax1.set_yticks([-0.2,0,0.5,1])
    ax1.set_ylabel('Silhouette coefficient',labelpad=-10)
    ax1.title.set_text('Position accuracy')

    diff = marks[1,:]-marks[0,:]
    ax11.bar(0, np.mean(diff), yerr = np.std(diff), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax11.scatter([0]*10, diff, alpha=0.3)
    # ax11.set_ylabel('Improvement of Silhouette mark by shared control')
    ax11.set_xlim([-1,1])
    ax11.set_ylim([-0.2,1])
    ax11.set_xticks([0], ["Position accuracy\n improvement"])
    ax11.set_title('P = 0.17')
    # ax.yaxis.grid(True)

    # orientation scatter plot
    for i in range(marks.shape[1]):
        ax2.plot([0,1], marks[2:4,i])
        ax2.scatter([0,1], marks[2:4,i])
    ax2.set_xticks([-0.3,0,1,1.3], ["","Full\n control", "Shared\n control",""])
    ax2.set_yticks([-0.2,0,0.5,1])
    ax2.set_ylabel('Silhouette coefficient',labelpad=-10)
    ax2.title.set_text('Orientation accuracy')
    

    diff = marks[3,:]-marks[2,:]
    ax22.bar(0, np.mean(diff), yerr = np.std(diff), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax22.scatter([0]*10, diff, alpha=0.3)
    # ax22.set_ylabel('Improvement of Silhouette mark by shared control')
    ax22.set_xlim([-1,1])
    ax22.set_ylim([-0.2,1])
    ax22.set_xticks([0], ["Orientation accuracy\n improvement"])
    ax22.set_title('P = 0.00001')


    for i in range(times.shape[1]):
        ax3.plot([0,1], times[:,i,0])
        ax3.scatter([0,1], times[:,i,0])
        # ax3.errorbar([0,1], times[:,i,0],yerr=times[:,i,1])
    ax3.set_xticks([-0.3,0,1,1.3], ["","Full\n control", "Shared\n control",""])
    ax3.set_ylim([50,120])
    ax3.set_ylabel("Time per trail/s")
    ax3.title.set_text('Efficiency')

    diff = times[1,:,0]-times[0,:,0]
    ax33.bar(0, np.mean(diff), yerr = np.std(diff), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax33.scatter([0]*10, diff, alpha=0.3)
    ax33.set_xlim([-1,1])
    # ax33.set_ylim([0,50])
    ax33.set_xticks([0], ["Time per trail\n reduction"])
    ax33.set_title('P = 0.016')

    for i in range(tumors.shape[1]):
        ax4.scatter([0,1], tumors[:,i])
        ax4.plot([0,1], tumors[:,i])
        # ax4.set_ylabel("count")
    ax4.set_xticks([-0.3,0,1,1.3], ["","Full\n control", "Shared\n control",""])
    ax4.set_ylim([0.2, 1.1])
    ax4.set_ylabel("Ratio of correct diagnosis")
    ax4.set_title("Diagnosis accuracy")

    diff = tumors[1,:]-tumors[0,:]
    ax44.bar(0, np.mean(diff), yerr = np.std(diff), align='center', alpha=0.5, ecolor='black', capsize=10)
    ax44.scatter([0]*10, diff, alpha=0.3)
    # ax22.set_ylabel('Improvement of Silhouette mark by shared control')
    ax44.set_xlim([-1,1])
    # ax44.set_ylim([-0.5,0.5])
    ax44.set_xticks([0], ["Daignosis accuracy\n improvement"])
    ax44.set_title('P = 0.34')

    plt.show()


def p_calculate(marks, times_all, tumors, times):
    order = [0,        3,       2,      4,      1,   6,    5,      7,     8,     9]
    # marks 
    label = ["position-full","position-shared","orientation-full","orientation-shared"]
    for i, mark in enumerate(marks):
        print(label[i]+": ",stats.kstest(mark,'norm').pvalue)
    print("position: ", stats.ttest_rel((marks[0,:]+1), (marks[1,:]+1)).pvalue)
    print("orientation: ", stats.ttest_rel((marks[2,:]+1), (marks[3,:]+1)).pvalue)

    # times
    label = [" time-full", " time-shared"]
    for i in range(len(times_all[0])):
        # print(" subject "+ str(i) + label[0] +" kstest: ",stats.kstest(times_all[0][i],'norm').pvalue)
        # print(" subject "+ str(i) + label[1] +" kstest: ",stats.kstest(times_all[1][i],'norm').pvalue)
        print("subject "+ str(i) +" time ttest: ",stats.ttest_ind((times_all[0][order[i]]),(times_all[1][order[i]])).pvalue)
        # print("subject "+ str(i) +" time utest: ",stats.mannwhitneyu(times_all[0][order[i]],times_all[1][order[i]]).pvalue)
    
    # print("time overall: ", stats.ttest_ind(times[0,:,0], times[1,:,0]).pvalue)
    print("time overall: ", stats.ttest_rel((times[0,:,0]), (times[1,:,0])).pvalue)

    # tumors
    label = ["tumor-full","tumor-shared"]
    for i, tumor in enumerate(tumors):
        print(label[i]+": ",stats.kstest(tumor,'norm').pvalue)
    # print("tumor accuracy ttest: ", stats.ttest_ind(tumors[0,:], tumors[1,:]).pvalue)
    # print("tumor accuracy ttest: ", stats.kstest(tumors[0,:]-tumors[1,:], 'norm').pvalue)
    print("tumor accuracy ttest: ", stats.ttest_rel((tumors[0,:]), (tumors[1,:])).pvalue)
    diff_mean = tumors[1,:] - tumors[0,:]
    n = len(tumors[0])
    s = np.sqrt((n-1)*np.var(tumors[0], ddof=1) + (n-1)*np.var(tumors[1], ddof=1) / (2*n - 2))
    d = diff_mean / s
    print("d: ",d, np.average(d))


def plot_learning_curve(times_all, tumors_all):
    # get rolling average

    order = [0,        3,       2,      4,      1,   6,    5,      7,     8,     9]
    rolling = []
    full_first_tumor = []
    shared_first_tumor = []
    full_first_time = []
    shared_first_time = []
    n = 5
    for idx, i in enumerate(order):
        if i%2 == 0:
            attempt_time = times_all[0][idx] + times_all[1][idx]
            rolling_time = [sum(attempt_time[t:t+n])/n for t in range(len(attempt_time)-n)]
            full_first_time.append(rolling_time)
            
        else:
            attempt_time = times_all[1][idx] + times_all[0][idx]
            rolling_time = [sum(attempt_time[t:t+n])/n for t in range(len(attempt_time)-n)]
            shared_first_time.append(rolling_time)
    
    # plot the time learning curve
    fig,(ax1, ax2) = plt.subplots(1, 2)

    full_mean, full_std = [np.mean(full_first_time, axis = 0), np.std(full_first_time, axis = 0)]
    full_mean_plot, = ax1.plot(np.arange(20-n), full_mean,'r')
    # full_plot = ax1.fill_between(np.arange(15), np.array(full_mean-full_std), np.array(full_mean+full_std))
    full_plot = ax1.fill_between(np.arange(20-n), np.array(full_mean-full_std), np.array(full_mean+full_std), facecolor ='red', alpha=0.2)
    shared_mean, shared_std = [np.mean(shared_first_time, axis = 0), np.std(shared_first_time, axis = 0)]
    shared_mean_plot, = ax1.plot(np.arange(20-n), shared_mean,'b')
    shared_plot = ax1.fill_between(np.arange(20-n), np.array(shared_mean-shared_std), shared_mean+shared_std, facecolor  ='blue', alpha=0.2)
    vline = ax1.axvline(x=(20-n)/2, c='g')
    ax1.legend([full_mean_plot, shared_mean_plot, vline], ["Full control first", "Shared control first", "Switch control mode"])
    ax1.set_title("Efficiency learning curve")
    ax1.set_ylabel("Time per trail rolling average of 5 trails(s)")
    ax1.set_xlabel("Medium trail count")
    ax1.set_xticks(np.arange(20-n),np.arange(20-n)+n//2+1)

    n = 5
    for i in range(10):
        if i%2 == 0:
            attempt_tumor = tumors_all[0][i] + tumors_all[1][i]
            rolling_tumor = [sum(attempt_tumor[t:t+n])/n for t in range(len(attempt_tumor)-n)]
            full_first_tumor.append(rolling_tumor)
        else:
            attempt_tumor = tumors_all[1][i] + tumors_all[0][i]
            rolling_tumor = [sum(attempt_tumor[t:t+n])/n for t in range(len(attempt_tumor)-n)]
            shared_first_tumor.append(rolling_tumor)

    full_mean, full_std = [np.mean(full_first_tumor, axis = 0), np.std(full_first_tumor, axis = 0)]
    full_mean_plot, = ax2.plot(np.arange(20-n), full_mean,'r')
    # full_plot = ax1.fill_between(np.arange(15), np.array(full_mean-full_std), np.array(full_mean+full_std))
    full_plot = ax2.fill_between(np.arange(20-n), np.array(full_mean-full_std), np.array(full_mean+full_std), facecolor ='red', alpha=0.2)
    shared_mean, shared_std = [np.mean(shared_first_tumor, axis = 0), np.std(shared_first_tumor, axis = 0)]
    shared_mean_plot, = ax2.plot(np.arange(20-n), shared_mean,'b')
    shared_plot = ax2.fill_between(np.arange(20-n), np.array(shared_mean-shared_std), shared_mean+shared_std, facecolor  ='blue', alpha=0.2)
    vline = ax2.axvline(x=(20-n)/2, c='g')
    ax2.legend([full_mean_plot, shared_mean_plot, vline], ["Full control first", "Shared control first", "Switch control mode"])
    ax2.set_title("Diagnosis accuracy learning curve")
    ax2.set_ylabel("Diagnosis accuracy rolling average of 5 trails")
    ax2.set_xlabel("Medium trail count")
    ax2.set_xticks(np.arange(20-n),np.arange(20-n)+n//2+1)

    plt.show()


def animate_skin_data(skin_data_all):
    for i in range(len(skin_data_all)):
        skin_data = np.array(skin_data_all[i])
        for data in skin_data:
            # plot the data
            fig, ax = plt.subplots(1)

            c = ax.pcolor(skin_data)
            plt.show()


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
mark0 = []
mark1 = []
mark0_angle = []
mark1_angle = []
time0 = []
time1 = []
time0_all = []
time1_all = []
dist0_pos = []
dist1_pos = []
dist0_orientation = []
dist1_orientation = []

for i, name in enumerate(names[:]):
    # continue
    participant = DataReader(name, dates[i])
    data = participant.readData(invalids[i])
    
    # skin_data_all0 = participant.get_skin_data(0)
    # animate_skin_data(skin_data_all0)
    
    # times = participant.gettime(0)
    # time0.append([np.average(times), np.std(times)])
    # time0_all.append(times)
    # times = participant.gettime(1)
    # time1.append([np.average(times), np.std(times)])
    # time1_all.append(times)

    # mark0.append(participant.indi_trajectory(0, '2d'))
    # mark1.append(participant.indi_trajectory(1, '2d'))
    # mark0_angle.append(participant.indi_orientation(0))
    # mark1_angle.append(participant.indi_orientation(1))

    # participant.indi_trajectory(0,option = '3d')
    # participant.indi_trajectory(1,option = '3d')
    participant.indi_orientation(0)
    # participant.indi_orientation(1)

    # dist0_pos.append(participant.euclidean_dist(0, isPos=True))
    # dist1_pos.append(participant.euclidean_dist(1, isPos=True))
    # dist0_orientation.append(participant.euclidean_dist(0, isPos=False))
    # dist1_orientation.append(participant.euclidean_dist(1, isPos=False))

# print(mark0, mark1, mark0_angle, mark1_angle)
# print(time0, time1)

# learning = []
# for i in range(7):
#     if i%2 == 0:
#         attempt = time0[i] + time1[i]
#         # continue
#     else:
#         # attempt = time1[i] + time0[i]
#         continue
#     rolling = rolling = [sum(attempt[i:i+5])/5 for i in range(15)]
#     learning.append(rolling)
# print(learning)

# print(np.average(learning, axis=0), np.std(learning, axis=0))
# for i in np.average(learning, axis=0):
#     print(i)
# for i in np.std(learning, axis=0):
#     print(i)

# in the order of name list
mark0 = np.array([0.3982887626371427, 0.1041852035481876, 0.22940400231743108, 0.6859459525096945, -0.08005022972869132, 0.2914207424937762, 0.5003657173254259,0.9257689921558008, 0.9320390081903475, 0.91648677240812])
mark1 = np.array([0.6221432149287827, 0.4408150713107583, 0.5242475367199013, 0.6616246567070211, 0.12047494263133464, 0.18164730370181445, 0.3887231526280848,0.9398736815108379, 0.910972487855924, 0.907592234402999])
mark0_angle = np.array([0.29743335404968907, 0.03273509531180943, 0.41054025622796325, 0.23009039411252213, 0.1011200755839542, -0.025636918167760425, 0.17512588541697457,0.2933037015667058, 0.6973922042055695, 0.37962114557566556])
mark1_angle = np.array([0.9116579871905461, 0.866342381796012, 0.8926180800484717, 0.9036923915264397, 0.8493479090334235, 0.8536370623294096, 0.8278140794447011,0.8652241981758079, 0.7927376448860052, 0.7756616067610064])
marks = np.array([mark0, mark1, mark0_angle, mark1_angle])

time0 = np.array([[63.895999999980816, 15.96057101732781], [96.26480000000632, 17.3384180870183], [86.08099999999467, 17.17143741798552], [110.25119999999296, 12.71105758620419], [107.19200000000274, 8.483748228228203], [84.78299999999979, 23.817931207390597], [95.86880000000005, 12.728618564478971],[105.41600000000327, 11.14726958497338], [89.35760000002338, 14.116088772754592], [111.1960000000001, 6.190782567656518]])
time1 = np.array([[54.73699999999998, 9.8341953915915], [101.12599999999657, 14.236424691615378], [58.048888888889145, 7.016776284882183], [77.2527999999933, 17.270997358574675], [107.12799999999697, 10.773786600557012], [82.710222222222, 13.438473340688713], [87.77200000000002, 19.171266583092486],[97.57039999997941, 13.456594065369119], [77.68959999998548, 7.219077658550694], [78.08000000000008, 14.284361363237467]])
times =np.array([time0,time1])
# times_all = []
times_all = [time0_all, time1_all]

# in the order of experiment
tumor0 = np.array([0.6, 0.4, 0.6, 0.7, 0.9, 0.9, 0.8, 0.6, 0.6, 0.7])
tumor1 = np.array([0.7, 0.6, 1, 0.8, 1, 0.7, 0.8, 0.5, 0.4, 0.9])
tumors = np.array([tumor0, tumor1])

# plot_all(marks, times, tumors)
# p_calculate(marks, times_all, tumors, times)

# tumors, tumors_all = loadtumordata()
# plot_learning_curve(times_all, tumors_all)

# print(stats.ttest_rel(tumors[0][::2],tumors[1][::2]).pvalue)
# print(stats.ttest_rel(tumors[0][1::2],tumors[1][1::2]).pvalue)
# print(stats.ttest_rel(tumors[0],tumors[1]).pvalue)

# label = ["position-full","position-shared","orientation-full","orientation-shared"]
# for i, mark in enumerate(marks):
#     print(label[i]+": ",stats.kstest(mark,'norm').pvalue)
# print("position: ", stats.ttest_ind(marks[0,:], marks[1,:]).pvalue)
# print("orientation: ", stats.ttest_ind(marks[2,:], marks[3,:]).pvalue)

# for participant in range(10):
#     pos0 = (np.average(dist0_pos[participant]), np.std(dist0_pos[participant]))
#     pos1 = (np.average(dist1_pos[participant]), np.std(dist1_pos[participant]))
#     ori0 = (np.average(dist0_orientation[participant]), np.std(dist0_orientation[participant]))
#     ori1 = (np.average(dist1_orientation[participant]), np.std(dist1_orientation[participant]))
#     print("participant ", participant, ": ", pos0,pos1,ori0,ori1)

# for a in marks:
    # print(np.average(a), np.std(a))
    # for b in a:
    #     print(b)


# print(np.average(time0[0,:]), np.std(time0[0,:]))
# print(np.average(time1[0,:]), np.std(time1[0,:]))

# label = ["time-full", "time-shared"]
# for i, time in enumerate(times):
#     print(label[i]+": ",stats.kstest(time[3:7,0],'norm').pvalue)
# print("time: ", stats.ttest_ind(times[0,:,0], times[1,:,0]).pvalue)



# print("diagnosis0: ",stats.kstest(diagnosis0,stats.norm.cdf).pvalue)
# print("diagnosis1: ",stats.kstest(diagnosis1,stats.norm.cdf).pvalue)
# print("diagnosis: ",stats.ttest_ind(diagnosis0,diagnosis1).pvalue)


# full_first_full = [0.6, 0.6, 0.7, 0.9, 0.6]
# full_first_shared = [0.7, 1, 0.8, 0.7,0.4]
# shared_first_full = [0.4,0.9,0.8,0.6,0.7]
# shared_first_shared = [0.6,0.8,0.8,0.5,0.9]
# print(stats.ttest_rel(full_first_full,full_first_shared).pvalue)
# print(stats.ttest_rel(shared_first_full,shared_first_shared).pvalue)
# print(stats.ttest_rel(full_first_full+shared_first_full, shared_first_shared+full_first_shared))
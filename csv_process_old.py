import csv
from matplotlib import projections
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import cv2

# file = open('data\starting_0_1.csv')
# csvreader = csv.reader(file)
# header = next(csvreader)
# data = []
# rowcount = 0
# for row in csvreader:
#     rowcount += 1
#     if rowcount %1000 == 0:
#         print(rowcount)
#     if rowcount > 10000:
#         break
#     TCPpose = list(map(float, row[28][1:-1].split(',')))
#     TCPposereal = list(map(float, row[25][1:-1].split(',')))
#     data.append(TCPposereal)
# print(header[25])
# data = np.array(data)
# plt.plot(data[:,0], data[:,1])
# plt.show()

# moveDatas = []
# forceDatas = []
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# for i in [0,2]: #data 1 does not go through the same start point?
#     moveData, forceData = np.load("data\moveData\moveDatab"+str(i+1)+".npy")
#     moveDatas.append(moveData)
#     forceDatas.append(forceData)
#     scatter = ax.plot(np.unwrap(moveData[:,3]), np.unwrap(moveData[:,4]), np.unwrap(moveData[:,5]))
#     # ax.plot(forceData[:,2])
#     # scatter = ax.plot(moveData[:,0], moveData[:,1], moveData[:,2])
#     # labels = ['point {0}'.format(i + 1) for i in range(len(moveData))]
#     # tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
# # mpld3.plugins.connect(fig, tooltip)
# # ax = fig.add_subplot(211)



# mpld3.show()
# plt.title("Plapation trajectory on abdominal phantom")
# plt.show()



moveDatas = []
forceDatas = []
fig = plt.figure()
ax = fig.add_subplot()
palptime = [[1879, 2804, 3758, 4450],[],[1558, 2685, 3583, 4591], [1668, 2415, 3592, 4336], [1277, 1791, 2824, 3402]]
# palptime = [[1388, 2642, 4328,5483],[], [1372, 2828, 4636, 6215], [2089, 4282, 5881, 7038], [1600, 3351, 4786, 6131]]
palppos = np.zeros((4,4,3))
for j,i in enumerate([0,2,3,4]): #data 1 does not go through the same start point?
    moveData, forceData = np.load("data\moveData\moveDatab"+str(i+1)+".npy")
    moveDatas.append(moveData)
    forceDatas.append(forceData)
    # scatter = ax.scatter(range(len(forceData)),forceData[:,2])
    scatter = ax.scatter(moveData[:,1], moveData[:,2])
    labels = ['point {0}'.format(i + 1) for i in range(len(moveData))]
    tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
    # print(moveData[np.array(palptime[i]), :3])
    orientation = (moveData[np.array(palptime[i]), 3:])
    for i, row in enumerate(orientation):
        converted = cv2.Rodrigues(row)[0]
        endpose = np.matmul(np.array([0,0,1]),converted)
        palppos[j,i,:] = endpose 
mpld3.plugins.connect(fig, tooltip)
print(np.average(palppos, axis = 0))
avecenter = np.average(palppos, axis = 0)
diff = palppos - np.tile(avecenter,(4,1,1))
dist = np.linalg.norm(diff,axis=2)
print(np.average(dist), np.std(dist))
# mpld3.show()

# start & end point: (timestep 0.01)
# f1: 256-4613, f3: 761-4737, f4: 264-4476, f5: 242-3554 = 43.57s, 39.76s, 42.12s, 33.12s

# palpation point:
# f1: 1879 2804 3758 4450, f3: 1558 2685 3583 4591, f4: 1668 2415 3592 4336, f5: 1277 1791 2824 3402


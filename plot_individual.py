import csv
from logging import exception
from matplotlib import projections
import matplotlib.pyplot as plt
import numpy as np
import mpld3
import cv2
import sklearn.metrics

# filename = 'csvdata\leo1_feb-22-2023__0\starting_0_'
# filename = 'csvdata\jerry1_feb-26-2023__0\starting_0_'
filename = 'csvdata\\fan0_feb-28-2023__0\starting_0_'
# leo1 test 5 invalid, jerry1 test 0 invalid, jerry0 test 7 invalid, fan1 test 3,4
filelist = [0,1,2,3,4,5,6,7,8,9]
# filelist = [0,1,2,5,6,7,8,9]



def loaddata():
    TCPmap = {}
    # print(header[4],header[25])
    for testcount in filelist:
        file = open(filename+ str(testcount)+'.csv')
        # file = open('csvdata\leo1_feb-22-2023__0\starting_0_0.csv')
        csvreader = csv.reader(file)
        header = next(csvreader)
        TCP = []
        press = np.array([])
        rowcount = 0
        for row in csvreader:
            rowcount += 1
            if rowcount %1000 == 0:
                print("reading data: ",rowcount)
            if rowcount > 10000:
                print("row exceed 10000")
                break
            # TCPpose = list(map(float, row[28][1:-1].split(',')))
            TCPposereal = list(map(float, row[25][1:-1].split(',')))
            # keys_pressed = list(map(float, row[5][1:-1].split(',')))
            keys_pressed = row[4]
            TCP.append(TCPposereal)
            if "\'L\'" in keys_pressed:
                press = np.append(press, int(rowcount))
        TCP = np.array(TCP)
        TCPmap[testcount] = (TCP,press)
    return TCPmap, TCP
    #TCPmap: testcount -> (TCP, press) 
    

def findpress(press, TCP, threshold = 1000):
    #divide into three events
    eventcount = 0
    eventmark = np.zeros(3)
    for t in range(len(press)):
        if t == 0:
            eventmark[eventcount] = press[t]
        elif t < len(press) -1 :
            # if within same 6s, count as same event
            if press[t] - press[t-1] > 300:
                eventcount += 1
                eventmark[eventcount] = press[t]
    if eventcount != 2:
        print("wrong event detection: ", eventcount)
        print(press)
        return None
    press = press.astype(int)
    eventmark = eventmark.astype(int)

    # find lowest press point
    pressmin = np.ones((2,3))*10  #[[depth,d,d],[timestep,t,t]]
    border = [-0.375, -0.28]
    for event, timestep in enumerate(eventmark):
        for t in np.arange(max(timestep-threshold,0),min(timestep+threshold,TCP.shape[0])):
            if TCP[t, 2] < pressmin[0, event]:
                pressmin[0, event] = TCP[t, 2]
                pressmin[1, event] = t
    # if pressmin[0,0] == pressmin[0,1] or pressmin[0,1] == pressmin[0,2]:
    if TCP[int(pressmin[1,0]),0] > border[0] or TCP[int(pressmin[1,1]),0] < border[0] or TCP[int(pressmin[1,1]),0] >border[1] or TCP[int(pressmin[1,2]),0] < border[1]:
        if threshold < 5:
            return None
        # print("wrong press event detection, try with ", str(threshold-200))
        pressmin = findpress(press, TCP, threshold=threshold-200)
    return pressmin


def indi_trajectory(TCPmap, option = "3d"):
    # individual trajecto``ry plot
    fig = plt.figure()
    pressTCPs = [[],[],[]]  # pressTCP for each palpation point over the 10 trails
    if option == "3d":
        ax = fig.add_subplot(projection='3d')
        counter = 0
        for item in TCPmap.items():
            print("event: ", counter)
            counter += 1
            TCP, press = item[1]
            ax.plot(TCP[:,0], TCP[:,1], TCP[:,2])

            # mark all press points
            # pressTCP = TCP[[int(step) for step in press]]
            # ax.scatter(pressTCP[:,0], pressTCP[:,1], pressTCP[:,2], color='r')
            
            pressmin = findpress(press, TCP)
            if not isinstance(pressmin, np.ndarray):        # error occured when finding the three events
                print("no valid press proints found for event ", str(counter-1))
                continue
            pressTCP = TCP[pressmin[1,:].astype(int)]
            pressTCPs[0].append(pressTCP[0,:2]+pressmin[1,0])
            pressTCPs[1].append(pressTCP[1,:2]+pressmin[1,1])
            pressTCPs[2].append(pressTCP[2,:2]+pressmin[1,2])
            ax.scatter(pressTCP[:,0], pressTCP[:,1], pressTCP[:,2], color='r')
            
        plt.title("individual result without shared control")
        plt.show()
        # print("average: ", np.average(np.array(pressTCPs), 1))
        # dist = np.linalg.norm(np.array(pressTCPs) - np.stack([np.average(np.array(pressTCPs), 1)]*len(filelist), axis = 1), axis = 2)
        # print("dist sum: ", np.sum(dist, axis=1))
        label = np.array([0]*len(filelist) + [1]*len(filelist) + [2]*len(filelist))
        silhouette = sklearn.metrics.silhouette_score(np.array(pressTCPs).reshape(3*len(filelist),2), label)
        print("silhouetter mark for position: ",round(silhouette,4))
    elif option == "2d":
        ax = fig.add_subplot()
        for item in TCPmap.items():
            TCP, press = item[1]

            pressmin = findpress(press, TCP)
            if not isinstance(pressmin, np.ndarray):        # error occured when finding the three events
                continue
            pressTCP = TCP[pressmin[1,:].astype(int)]
            pressTCPs[0].append(pressTCP[0,:2]+pressmin[1,0])
            pressTCPs[1].append(pressTCP[1,:2]+pressmin[1,1])
            pressTCPs[2].append(pressTCP[2,:2]+pressmin[1,2])
            ax.scatter(pressTCP[0,0], pressTCP[0,1], color='r')
            ax.scatter(pressTCP[1,0], pressTCP[1,1], color='g')
            ax.scatter(pressTCP[2,0], pressTCP[2,1], color='b')
        plt.title("individual result without shared control")
        plt.show()
        label = np.array([0]*len(filelist) + [1]*len(filelist) + [2]*len(filelist))
        silhouette = sklearn.metrics.silhouette_score(np.array(pressTCPs).reshape(3*len(filelist),2), label)
        print("silhouetter mark for position: ",round(silhouette,4))
    
    



def TCP2pose(TCP):
    # convert rotation vector in the TCP to the position of tip in unit vector space
    poses = np.zeros((TCP.shape[0],3))
    for i, row in enumerate(TCP):
        startpos = np.array([[0,0,1]])
        R, _ = cv2.Rodrigues(row[3:])
        endpose = np.matmul(R,startpos.T)
        poses[i] = endpose.reshape(3,)
    return poses

def pose2angle(TCPpose):
    # convert the pose of the end effector to theta, phi
    angles = np.zeros((TCPpose.shape[0],2))
    for i, pose in enumerate(TCPpose):
        theta = np.arctan2(np.linalg.norm(pose[:2]), pose[2])
        phi = np.arctan2(pose[1],pose[0])
        angles[i,:] = [theta, phi]
    return angles

def indi_orientation(TCPmap):
    # individual orientation plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.axes.set_xlim3d(left=-0.7, right=0.7) 
    ax.axes.set_ylim3d(bottom=-0.7, top=0.7) 
    ax.axes.set_zlim3d(bottom=-1, top=0.4) 
    # ax = fig.add_subplot()
    pressTCPs = [[],[],[]]
    counter = 0
    for item in TCPmap.items():
        print("event: ", counter)
        counter += 1
        TCP, press = item[1]
        TCPpose = TCP2pose(TCP)
        angles = pose2angle(TCPpose)
        # ax.plot(TCPpose[:,0], TCPpose[:,1], TCPpose[:,2])

        # mark all press points
        # pressTCP = TCP[[int(step) for step in press]]
        # ax.scatter(pressTCP[:,0], pressTCP[:,1], pressTCP[:,2], color='r')
        
        pressmin = findpress(press, TCP)
        if not isinstance(pressmin, np.ndarray):        # error occured when finding the three events
            continue

        pressTCP = TCPpose[pressmin[1,:].astype(int)]
        ax.scatter(pressTCP[0,0], pressTCP[0,1], pressTCP[0,2], color='r')
        ax.scatter(pressTCP[1,0], pressTCP[1,1], pressTCP[1,2], color='g')
        ax.scatter(pressTCP[2,0], pressTCP[2,1], pressTCP[2,2], color='b')

        pressTCPs[0].append(pressTCP[0,:])
        pressTCPs[1].append(pressTCP[1,:])
        pressTCPs[2].append(pressTCP[2,:])
    
    # xx_grid, yy_grid = np.meshgrid(np.arange(-0.5,0.5,0.01), np.arange(-0.5,0.5,0.01))
    # shell = np.sqrt(1.001-xx_grid**2-yy_grid**2)
    theta = np.arange(0,np.pi/3,0.005)
    phi = np.arange(0,np.pi*2,0.005)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    shell = [np.sin(theta_grid)*np.cos(phi_grid), np.sin(theta_grid)*np.sin(phi_grid), np.cos(theta_grid)]
    ax.plot_surface(shell[0], shell[1], -shell[2], alpha=.3)
    plt.title("individual result with shared control")
    plt.show()

    label = np.array([0]*len(filelist) + [1]*len(filelist) + [2]*len(filelist))
    silhouette = sklearn.metrics.silhouette_score(np.array(pressTCPs).reshape(3*len(filelist),3), label)
    print("silhouetter mark for position: ",round(silhouette,4))

TCPmap, TCP = loaddata()
indi_trajectory(TCPmap,option="3d")
indi_orientation(TCPmap)

# moveDatas = []
# forceDatas = []
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# for i in [0,2]: #data 1 does not go through the same start point?
#     moveData, forceData = np.load("data\moveData\moveDatab"+str(i+1)+".npy")
#     moveDatas.append(moveData)
#     forceDatas.append(forceData)
#     scatter = ax.plot(np.unwrap(moveData[:,3]), np.unwrap(moveData[:,4]), np.unwrap(moveData[:,5]))
#     ax.plot(forceData[:,2])
#     scatter = ax.plot(moveData[:,0], moveData[:,1], moveData[:,2])
#     labels = ['point {0}'.format(i + 1) for i in range(len(moveData))]
#     tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
# mpld3.plugins.connect(fig, tooltip)
# ax = fig.add_subplot(211)



# mpld3.show()
# plt.title("Plapation trajectory on abdominal phantom")
# plt.show()



# moveDatas = []
# forceDatas = []
# fig = plt.figure()
# ax = fig.add_subplot()
# palptime = [[1879, 2804, 3758, 4450],[],[1558, 2685, 3583, 4591], [1668, 2415, 3592, 4336], [1277, 1791, 2824, 3402]]
# # palptime = [[1388, 2642, 4328,5483],[], [1372, 2828, 4636, 6215], [2089, 4282, 5881, 7038], [1600, 3351, 4786, 6131]]
# palppos = np.zeros((4,4,3))
# for j,i in enumerate([0,2,3,4]): #data 1 does not go through the same start point?
#     moveData, forceData = np.load("data\moveData\moveDatab"+str(i+1)+".npy")
#     moveDatas.append(moveData)
#     forceDatas.append(forceData)
#     # scatter = ax.scatter(range(len(forceData)),forceData[:,2])
#     scatter = ax.scatter(moveData[:,1], moveData[:,2])
#     labels = ['point {0}'.format(i + 1) for i in range(len(moveData))]
#     tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
#     # print(moveData[np.array(palptime[i]), :3])
#     orientation = (moveData[np.array(palptime[i]), 3:])
#     for i, row in enumerate(orientation):
#         converted = cv2.Rodrigues(row)[0]
#         endpose = np.matmul(np.array([0,0,1]),converted)
#         palppos[j,i,:] = endpose 
# mpld3.plugins.connect(fig, tooltip)
# print(np.average(palppos, axis = 0))
# avecenter = np.average(palppos, axis = 0)
# diff = palppos - np.tile(avecenter,(4,1,1))
# dist = np.linalg.norm(diff,axis=2)
# print(np.average(dist), np.std(dist))
# mpld3.show()

# start & end point: (timestep 0.01)
# f1: 256-4613, f3: 761-4737, f4: 264-4476, f5: 242-3554 = 43.57s, 39.76s, 42.12s, 33.12s

# palpation point:
# f1: 1879 2804 3758 4450, f3: 1558 2685 3583 4591, f4: 1668 2415 3592 4336, f5: 1277 1791 2824 3402


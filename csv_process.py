import csv
from logging import exception
from matplotlib import projections
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sklearn.metrics
import json
import os

class DataReader:
    def __init__(self,name, date):  #invalid([list], [list]): list of invalid experiement ids
        self.name = name
        self.filename0 = 'csvdata\\' + name + '0_' + date + '-2023__0\starting_0_'
        self.filename1 = 'csvdata\\' + name + '1_' + date + '-2023__0\starting_0_'
    
    def readData(self, invalid):
        self.filelist0 = [i for i in range(10)]
        self.filelist1 = [i for i in range(10)]
        _ = [self.filelist0.remove(i) for i in invalid[0]]
        _ = [self.filelist1.remove(i) for i in invalid[1]]
        
        savename = 'csvdata\\' + self.name
        self.savename = savename

        if os.path.exists(savename+'0.json'):
            with open(savename+'0.json', 'r') as fp:
                self.data0 = json.load(fp)
        else:
            print("load 0 set")
            self.data0 = loaddata(self.filename0, self.filelist0, savename+'0.json')
            self.data0['s_trajectory'] = []
            self.data0['s_oreintation'] = []

        if os.path.exists(savename+'1.json'):
            with open(savename+'1.json', 'r') as fp:
                self.data1 = json.load(fp)        
        else:
            print("load 1 set")
            self.data1 = loaddata(self.filename1, self.filelist1, savename+'1.json')
            self.data1['s_trajectory'] = 0
            self.data1['s_oreintation'] = 0


        
    # isShared: 0 for no, 1 for yes
    def indi_trajectory(self, isShared, option = '3d'):
        if not isShared:
            TCPmap = {}
            for fileid in self.filelist0:
                TCPmap[fileid] = (self.data0['actual_TCP_pose'][int(fileid)], self.data0['keys_pressed'][int(fileid)])
            # for key in self.data0:
            #     TCPmap[key] = (self.data0['actual_TCP_pose'][int(key)], self.data0['keys_pressed'][int(key)])
            s_mark = indi_trajectory(TCPmap, self.filelist0, option, isShared)
            self.data0['s_trajectory'] = s_mark
            with open(self.savename+'0.json', 'r') as fp:
                self.data0 = json.load(fp)    
        else:
            TCPmap = {}
            for fileid in self.filelist1:
                TCPmap[fileid] = (self.data1['actual_TCP_pose'][int(fileid)], self.data1['keys_pressed'][int(fileid)])
            s_mark = indi_trajectory(TCPmap, self.filelist1, option, isShared)
            self.data1['s_trajectory'] = s_mark
            with open(self.savename+'1.json', 'r') as fp:
                self.data1 = json.load(fp)
        return s_mark
        
    
    # isShared: 0 for no, 1 for yes
    def indi_orientation(self, isShared):
        if not isShared:
            TCPmap = {}
            for fileid in self.filelist0:
                TCPmap[fileid] = (np.array(self.data0['actual_TCP_pose'][int(fileid)]), self.data0['keys_pressed'][int(fileid)])
            # for key in self.data0:
            #     TCPmap[key] = (self.data0['actual_TCP_pose'][int(key)], self.data0['keys_pressed'][int(key)])
            s_mark = indi_orientation(TCPmap, self.filelist0, isShared)
            self.data0['s_oreintation'] = s_mark
            with open(self.savename+'0.json', 'r') as fp:
                self.data0 = json.load(fp)    
        else:
            TCPmap = {}
            for fileid in self.filelist1:
                TCPmap[fileid] = (np.array(self.data1['actual_TCP_pose'][int(fileid)]), self.data1['keys_pressed'][int(fileid)])
            s_mark = indi_orientation(TCPmap, self.filelist1, isShared)
            self.data1['s_oreintation'] = s_mark
            with open(self.savename+'1.json', 'r') as fp:
                self.data1 = json.load(fp)    
        return s_mark

    def gettime(self, isShared):
        times = []
        if not isShared:
            # for fileid in self.filelist0:
            for fileid in range(10):
                times.append(float(self.data0['timestamp'][int(fileid)][-1]) - float(self.data0['timestamp'][int(fileid)][0]))
            
        else:
            # for fileid in self.filelist1:
            for fileid in range(10):
                times.append(float(self.data1['timestamp'][int(fileid)][-1]) - float(self.data1['timestamp'][int(fileid)][0]))
        return times

    def get_skin_data(self,isShared):
        skin_data_all = []
        if not isShared:
            for fileid in self.filelist0:
                # skin_data = [list(map(float, self.data0['skin_data'][fileid][i][4:-2].replace(']\n [',' ').split(' '))) for i in range(len(self.data0['skin_data'][fileid]))]
                skin_data = [str2matrix(self.data0['skin_data'][fileid][i]) for i in range(len(self.data0['skin_data'][fileid]))]
                skin_data_all.append(skin_data)
            self.skin_data_all0 = skin_data_all
        else:
            for fileid in self.filelist1:
                # skin_data = [list(map(float, self.data1['skin_data'][fileid][i][4:-2].replace(']\n [',' ').split(' '))) for i in range(len(self.data1['skin_data'][fileid]))]
                skin_data = [str2matrix(self.data1['skin_data'][fileid][i]) for i in range(len(self.data1['skin_data'][fileid]))]
                skin_data_all.append(skin_data)
            self.skin_data_all1 = skin_data_all
        # return skin_data_all

    # return the eclidean distance of the 10 palpation spots with respect to the ground truth
    def euclidean_dist(self, isShared, isPos = True, forPlot=False):
        ground_position = np.array([[-0.416, -0.074], [-0.330, 0.075], [-0.263, -0.152]])
        ground_orientation = TCP2pose(np.array([[0.174, 2.771, -0.111],[0.177, 3.070, 0.126],[0.187, 3.282, -0.21]]))
        
        TCPmap = {}
        if not isShared:
            for fileid in self.filelist0:
                TCPmap[fileid] = (self.data0['actual_TCP_pose'][int(fileid)], self.data0['keys_pressed'][int(fileid)])
            filelist = self.filelist0
        else:
            for fileid in self.filelist1:
                TCPmap[fileid] = (self.data1['actual_TCP_pose'][int(fileid)], self.data1['keys_pressed'][int(fileid)])
            filelist = self.filelist1
            
        if isPos:  # get the position of the 30 palpations and find their distance to ground truth
            mark, pressTCPs = indi_trajectory(TCPmap, filelist, option = "2d", isShared=isShared, isTCP=True)
            dist = []
            dist.append(np.linalg.norm(np.array(pressTCPs)[0,:,:2] - np.tile(ground_position[0,:], (len(filelist),1)), axis=1))
            dist.append(np.linalg.norm(np.array(pressTCPs)[1,:,:2] - np.tile(ground_position[1,:], (len(filelist),1)), axis=1))
            dist.append(np.linalg.norm(np.array(pressTCPs)[2,:,:2] - np.tile(ground_position[2,:], (len(filelist),1)), axis=1))
            
        else:   # get the orientation
            mark, pressTCPs = indi_orientation(TCPmap, filelist, isShared=isShared, isTCP=True)
            dist = []
            angle = []
            for i in range(3):
                diff = np.array(pressTCPs)[i,:,:] - np.tile(ground_orientation[i,:], (len(filelist),1))
                # angle.append(pose2angle(diff))   # this is wrong
                new_angle = []
                for row in pressTCPs[i]:
                    theta = np.arcsin(np.linalg.norm(np.cross(ground_orientation[i,:], row)))
                    new_angle.append(theta)
                angle.append(new_angle)
                dist.append(np.linalg.norm(diff, axis = 1))

            # dist.append(np.linalg.norm(np.array(pressTCPs)[0,:,:]) - np.tile(ground_orientation[0,:], (len(filelist),1), axis=1))
            # dist.append(np.linalg.norm(np.array(pressTCPs)[1,:,:]) - np.tile(ground_orientation[1,:], (len(filelist),1), axis=1))
            # dist.append(np.linalg.norm(np.array(pressTCPs)[2,:,:]) - np.tile(ground_orientation[2,:], (len(filelist),1), axis=1))

        if not forPlot:
            if isPos:  # euclidean distance to ground truth
                return dist
            else:       # angle(dist in radian) to ground truth, and the (theta, phi)
                return dist, angle
        if isPos:
            # average of 3 palpation spot over 10 trails
            pos = np.average(np.array(pressTCPs)[:,:,:2], axis = 1)
            # distance to center
            stacked = np.vstack((np.tile(pos[0,:], (1,len(filelist),1)), np.tile(pos[1,:], (1,len(filelist),1)), np.tile(pos[2,:], (1,len(filelist),1))))
            diff = np.linalg.norm(np.array(pressTCPs)[:,:,:2]- stacked, axis =2)
            return pos, diff
        else:
            pos = np.average(pose2angle(np.array(pressTCPs)[:,:,:]), axis = 1)
            return None
            #np.std(np.array(pressTCPs)[:,:,:3], axis = 1)

    def get_palpation_tactile(self, isShared):
        t = 10   # return 1s(~70) before and after the palpation event
        if isShared:
            filelist = self.filelist0
            data = self.data0
            skin_data_all = self.skin_data_all0
        else:
            filelist = self.filelist1
            data = self.data1
            skin_data_all = self.skin_data_all1
        TCPmap = {}
        for fileid in filelist:
            TCPmap[fileid] = (np.array(data['actual_TCP_pose'][int(fileid)]), data['keys_pressed'][int(fileid)])
        palp_data = [[],[],[]]  # pressTCP for each palpation point over the 10 trails
        # skin_data_all = self.get_skin_data(isShared)
        counter = 0
        for item in TCPmap.items():
            print("event: ", filelist[counter])
            skin_data = skin_data_all[counter]
            counter += 1
            TCP, key_history = item[1]
            TCP = np.array(TCP)
            pressmin = findpress(key_history, TCP)
            if not isinstance(pressmin, np.ndarray):        # error occured when finding the three events
                print("no valid press proints found for event ", filelist[counter-1])
                continue
            
            # record only the tactile at deepest palpation
            # tactile_data = [skin_data[i] for i in pressmin.astype(int)]
            # palp_data[0].append(tactile_data[0])
            # palp_data[1].append(tactile_data[1])
            # palp_data[2].append(tactile_data[2])

            # record the data within -t~t samples with max pta
            for i, point in enumerate(pressmin.astype(int)):
                result = []
                # for tstep in np.arange(max(0,point-t),min(len(skin_data),point+t)):
                #     # result.append(matrix_pta(skin_data[tstep]))   # peak to average value
                #     result.append(matrix_ave(skin_data[tstep]))     # average value
                #     # result.append(matrix_std(skin_data[tstep]))
                # result = np.array(result)[~np.isnan(result)]
                # palp_data[i].append(skin_data[np.arange(point-t,point+t)[np.argmax(result)]])

                # integration of the period
                for tstep in np.arange(max(0,point-t),min(len(skin_data),point+t)):
                    result.append(matrix_ave(skin_data[tstep]))
                result = np.array(result)[~np.isnan(result)]
                palp_data[i].append(np.average(result))

        return palp_data

    def get_control_keys(self, isShared):
        if not isShared:
            filelist = self.filelist0
            data = self.data0
        else:
            filelist = self.filelist1
            data = self.data1

        keys_history_all = []
        for fileid in filelist:
            keys_history = get_keys(data['keys_pressed'][int(fileid)])
            keys_history_all.append(keys_history)
        if not isShared:
            self.keys_history_all0 = keys_history_all
        else:
            self.keys_history_all1 = keys_history_all


def get_keys(keys_pressed):
    keys = np.array(keys_pressed[1:-1]).split(',')
    option = ['+X', '-X', '+Y', '-Y', 'PRESS', 'LIFT', '+Rx', '-Rx', '+Ry', '-Ry', 'L', 'C']
    key_result = [0,0,0,0]
    # [xy, z, Rxy, other], number for degrees pressed at the same time
    for key in keys:
        if key in option[:4]:
            key_result[0] += 1
        elif key in option[4:6]:
            key_result[1] += 1
        elif key in option[6:10]:
            key_result[2] += 1
        else:
            key_result[3] += 1
    return key_result


def matrix_pta(matrix):
    # the peak to average value on both ends
    matrix = np.array(matrix).flatten()
    if np.average(matrix) in [np.nan, np.inf, 0]:
        print(matrix)
        return np.nan
    return max(matrix)/np.average(matrix)

def matrix_ave(matrix):
    # the average value
    matrix = np.array(matrix).flatten()
    if np.average(matrix) in [np.nan, np.inf, 0]:
        print(matrix)
        return np.nan
    return np.average(matrix)

def matrix_std(matrix):
    # the std of the matrix
    matrix = np.array(matrix).flatten()
    if np.average(matrix) in [np.nan, np.inf, 0]:
        print(matrix)
        return np.nan
    return np.std(matrix)

def loaddata(filename, filelist, savename):
    data = {}
    # the id of the headers useful
    headerid = [1,4,8,9,14,25,26,27]

    # record data
    
    for testcount in range(10):
        print("read test ", testcount)
        file = open(filename+ str(testcount)+'.csv')
        csvreader = csv.reader(file)
        header = next(csvreader)
        rowcount = 0
        filedata = {}
        for row in csvreader:
            rowcount += 1
            # store data
            for i, item in enumerate(row):
                if i in headerid:
                    if header[i] not in filedata:
                        filedata[header[i]] = []
                    if i in [25,26,17]:
                        filedata[header[i]].append(list(map(float, item[1:-1].split(','))))
                    elif i in [8,9,14]:
                        filedata[header[i]].append(float(item))
                    else:
                        filedata[header[i]].append(item)
                        # key pressed, tactile map unchanged

            # filedata["actual_TCP_pose"] = list(map(float, filedata["actual_TCP_pose"][-1][1:-1].split(',')))

        # put filedata in to the overall data
        for key in filedata:
            if key not in data:
                data[key] = []
            data[key].append(filedata[key])

        # put the map of each test into the overall data map
        # data[testcount] = filedata

    # save file
    with open(savename, 'w') as fp:
        json.dump(data, fp)
    return data  
    
# press: keys pressed over an evperiement
# TCP: TCP history over an experiement
def findpress(key_history, TCP, threshold = 1000):
    #divide into three events
    eventcount = 0
    eventmark = np.zeros(3)

    # find when L is pressed, recorde the row id in press
    press = np.array([])
    for i, keys in enumerate(key_history):
        if "\'L\'" in keys:
            press = np.append(press, i)

    # divide into events
    for t in range(len(press)):
        if t == 0:
            eventmark[eventcount] = press[t]
        elif t < len(press) -1 :
            # if within same 6s, count as same event
            if press[t] - press[t-1] > 300:
                eventcount += 1
                if eventcount >2:
                    return None
                eventmark[eventcount] = press[t]
    if eventcount != 2:
        print("wrong event detection: ", eventcount)
        print(press)
        return None
    press = press.astype(int)
    eventmark = eventmark.astype(int)

    # find lowest press point
    depth = np.ones((3,))
    pressmin = np.ones((3,))  #[[depth,d,d],[timestep,t,t]]
    border = [-0.375, -0.28]
    for event, timestep in enumerate(eventmark):
        for t in np.arange(max(timestep-threshold,0),min(timestep+threshold,TCP.shape[0])):
            if TCP[t, 2] < depth[event]:
                depth[event] = TCP[t, 2]
                pressmin[event] = t

    # might not be pressed in the same order, sort
    pressminx = np.array([TCP[int(pressmin[0]),0], TCP[int(pressmin[1]),0], TCP[int(pressmin[2]),0]])
    idx = pressminx.argsort()
    pressminx = pressminx[idx]
    pressmin = pressmin[idx]
    if pressminx[0] > border[0] or pressminx[1] < border[0] or pressminx[1] >border[1] or pressminx[2] < border[1]:
    
    # if TCP[int(pressmin[0]),0] > border[0] or TCP[int(pressmin[1]),0] < border[0] or TCP[int(pressmin[1]),0] >border[1] or TCP[int(pressmin[2]),0] < border[1]:
        if threshold < 5:
            return None
        # print("wrong press event detection, try with ", str(threshold-200))
        pressmin = findpress(key_history, TCP, threshold=threshold-200)

    # return the three id of the press event in sorted order
    
    return pressmin


def indi_trajectory(TCPmap, filelist, option = "3d", isShared=True, isTCP=False):
    # individual trajectory plot
    fig = plt.figure()
    pressTCPs = [[],[],[]]  # pressTCP for each palpation point over the 10 trails
    if option == "3d":
        ax = fig.add_subplot(projection='3d')
        counter = 0
        for item in TCPmap.items():
            print("event: ", filelist[counter])
            counter += 1
            TCP, key_history = item[1]
            TCP = np.array(TCP)
            ax.plot(TCP[:,0], TCP[:,1], TCP[:,2])

            # mark all press points
            # pressTCP = TCP[[int(step) for step in press]]
            # ax.scatter(pressTCP[:,0], pressTCP[:,1], pressTCP[:,2], color='r')
            
            pressmin = findpress(key_history, TCP)
            if not isinstance(pressmin, np.ndarray):        # error occured when finding the three events
                print("no valid press proints found for event ", filelist[counter-1])
                continue
            pressTCP = TCP[pressmin.astype(int)]
            pressTCPs[0].append(pressTCP[0,:2])
            pressTCPs[1].append(pressTCP[1,:2])
            pressTCPs[2].append(pressTCP[2,:2])
            # pressTCPs[0].append(pressTCP[0,:2]+pressmin[0])
            # pressTCPs[1].append(pressTCP[1,:2]+pressmin[1])
            # pressTCPs[2].append(pressTCP[2,:2]+pressmin[2])
            # ax.scatter(pressTCP[:,0], pressTCP[:,1], pressTCP[:,2], color='r')
            ax.scatter(pressTCP[0,0], pressTCP[0,1], pressTCP[0,2], color='r')
            ax.scatter(pressTCP[1,0], pressTCP[1,1], pressTCP[1,2], color='g')
            ax.scatter(pressTCP[2,0], pressTCP[2,1], pressTCP[2,2], color='b')
            ax.set_xlabel('X/m',fontsize = 14.0)
            ax.set_ylabel('Y/m',fontsize = 14.0)
            ax.set_zlabel('Z/m',fontsize = 14.0)
            ax.set_xlim([-0.50, -0.20])
            ax.set_ylim([-0.15, 0.15])
            ax.set_zlim([0, 0.20])
            ax.set_xticks([-0.50, -0.40,-0.30,-0.20])
            ax.set_yticks([-0.15,-0.05,0.05,0.15])
            ax.set_zticks([0,0.10,0.20])


        if isShared:
            plt.title("Individual trajectory with shared control")
        else:
            plt.title("Individual trajectory with full control")
        plt.show()
        # print("average: ", np.average(np.array(pressTCPs), 1))
        # dist = np.linalg.norm(np.array(pressTCPs) - np.stack([np.average(np.array(pressTCPs), 1)]*len(filelist), axis = 1), axis = 2)
        # print("dist sum: ", np.sum(dist, axis=1))
        label = np.array([0]*len(filelist) + [1]*len(filelist) + [2]*len(filelist))
        silhouette = sklearn.metrics.silhouette_score(np.array(pressTCPs).reshape(3*len(filelist),2), label)
        print("silhouetter mark for position: ",round(silhouette,4))
    elif option == "2d":
        ax = fig.add_subplot()
        counter = 0
        for item in TCPmap.items():
            print("event: ", filelist[counter])
            counter += 1
            TCP, press = item[1]
            TCP = np.array(TCP)
            pressmin = findpress(press, TCP)
            if not isinstance(pressmin, np.ndarray):        # error occured when finding the three events
                print("no valid press proints found for event ", filelist[counter-1])
                continue
            pressTCP = TCP[pressmin.astype(int)]
            pressTCPs[0].append(pressTCP[0,:])
            pressTCPs[1].append(pressTCP[1,:])
            pressTCPs[2].append(pressTCP[2,:])
            # pressTCPs[0].append(pressTCP[0,:2])
            # pressTCPs[1].append(pressTCP[1,:2])
            # pressTCPs[2].append(pressTCP[2,:2])
            ax.scatter(pressTCP[0,0], pressTCP[0,1], color='r')
            ax.scatter(pressTCP[1,0], pressTCP[1,1], color='g')
            ax.scatter(pressTCP[2,0], pressTCP[2,1], color='b')

            ax.set_xlabel('X/m',fontsize = 14.0)
            ax.set_ylabel('Y/m',fontsize = 14.0)
            ax.set_xlim([-0.50, -0.20])
            ax.set_ylim([-0.15, 0.15])
            ax.set_xticks([-0.50, -0.40,-0.30,-0.20])
            ax.set_yticks([-0.15,-0.05,0.05,0.15])
        if isShared:
            plt.title("Individual palpation spot with shared control")
        else:
            plt.title("Individual palpation spot with full control")
        plt.show()
        label = np.array([0]*len(filelist) + [1]*len(filelist) + [2]*len(filelist))
        silhouette = sklearn.metrics.silhouette_score(np.array(pressTCPs)[:,:,:2].reshape(3*len(filelist),2), label)
        print("silhouetter mark for position: ",round(silhouette,4))
    
    if isTCP:
        return silhouette, pressTCPs
    return silhouette

def TCP2pose(TCP):
    # convert rotation vector in the TCP to the position of tip in unit vector space
    TCP = np.array(TCP)
    poses = np.zeros((TCP.shape[0],3))
    for i, row in enumerate(TCP):
        startpos = np.array([[0,0,1]])
        if len(row) == 6:
            R, _ = cv2.Rodrigues(row[3:])
        elif len(row) == 3: 
            R, _ = cv2.Rodrigues(row[:])
        else:
            print("can't recognize the input format")
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

def indi_orientation(TCPmap, filelist, isShared = True, isTCP = False):
    # individual orientation plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
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
        TCP = np.array(TCP)
        # ax.plot(TCPpose[:,0], TCPpose[:,1], TCPpose[:,2])

        # mark all press points
        # pressTCP = TCP[[int(step) for step in press]]
        # ax.scatter(pressTCP[:,0], pressTCP[:,1], pressTCP[:,2], color='r')
        
        pressmin = findpress(press, TCP)
        if not isinstance(pressmin, np.ndarray):        # error occured when finding the three events
            continue

        pressTCP = TCPpose[pressmin.astype(int)]
        ax.scatter(pressTCP[0,0], pressTCP[0,1], pressTCP[0,2], color='r')
        ax.scatter(pressTCP[1,0], pressTCP[1,1], pressTCP[1,2], color='g')
        ax.scatter(pressTCP[2,0], pressTCP[2,1], pressTCP[2,2], color='b')
        ax.set_xlabel('X',fontsize = 14.0)
        ax.set_ylabel('Y',fontsize = 14.0)
        ax.set_zlabel('Z',fontsize = 14.0)

        pressTCPs[0].append(pressTCP[0,:])
        pressTCPs[1].append(pressTCP[1,:])
        pressTCPs[2].append(pressTCP[2,:])
    
    xx_grid, yy_grid = np.meshgrid(np.arange(-0.5,0.5,0.01), np.arange(-0.5,0.5,0.01))
    shell = np.sqrt(1.001-xx_grid**2-yy_grid**2)
    theta = np.arange(0,np.pi/3,0.005)
    phi = np.arange(0,np.pi*2,0.005)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    shell = [np.sin(theta_grid)*np.cos(phi_grid), np.sin(theta_grid)*np.sin(phi_grid), np.cos(theta_grid)]
    ax.plot_surface(shell[0], shell[1], -shell[2], alpha=.3)

    if isShared:
        plt.title("Individual result with shared control")
    else:
        plt.title("Individual result full control")
    plt.show()

    label = np.array([0]*len(filelist) + [1]*len(filelist) + [2]*len(filelist))
    silhouette = sklearn.metrics.silhouette_score(np.array(pressTCPs).reshape(3*len(filelist),3), label)
    print("silhouetter mark for position: ",round(silhouette,4))
    if isTCP:
        return silhouette, pressTCPs
    return silhouette

def str2matrix(sentence):
    # need all elements in string to be a int with . at the end
    words = sentence.split('.')
    words = [[char for char in word] for word in words]
    matrix = []
    for word in words:
        newint = []
        for char in word:
            if char in '0123456789':
                newint.append(char)
        
        if len(newint) > 0:
            newint = ''.join([i for i in newint])
            matrix.append(int(newint))

        # matrix.append(int(newint.join('')))
    if len(matrix) == 64:
        return np.array(matrix).reshape((8,8))[0::2,0::2]
    else:
        print("error: ", sentence)
        return []

        





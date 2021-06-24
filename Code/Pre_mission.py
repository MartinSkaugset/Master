import os, time, sys, math

import numpy as np

import pyimc

from socket import *

my_path = os.path.dirname(os.path.abspath('main.py'))

# Removing and creating new files in the textfiles folder:
# True temperature for the mission area
if os.path.exists(str(my_path + '/Textfiles/truetemp.txt')):
    os.remove(str(my_path + '/Textfiles/truetemp.txt'))
    f = open(str(my_path + '/Textfiles/truetemp.txt'), 'x')
else:
    f = open(str(my_path + '/Textfiles/truetemp.txt'), 'x')

# Predicted temperature for the mission area
if os.path.exists(str(my_path + '/Textfiles/temppred.txt')):
    os.remove(str(my_path + '/Textfiles/temppred.txt'))
    f = open(str(my_path + '/Textfiles/temppred.txt'), 'x')
else:
    f = open(str(my_path + '/Textfiles/temppred.txt'), 'x')

# Set size of grid
xx = 50
yy = 50
n = xx * yy  # Total number of points in grid
shape = (xx, yy)

# Setup of mesh grid with x and y distances
Lx = 10000  # [m] Length in x-dir (east)
Ly = 10000  # [m] Length in y-dir (north)
dX, dY = Lx / xx, Ly / yy  # Grid partitioning

x_mesh, y_mesh = np.meshgrid(np.arange(0, Lx, dX), np.arange(0, Ly, dY))

# Set up coordinates for the area of interest.
lat = np.zeros(shape)
lon = np.zeros(shape)
first_coord = np.array([63.906453, 8.503611])
first_coord = first_coord * math.pi / 180

for x in range(yy):
    for y in range(xx):
        lat1, lon1 = pyimc.coordinates.WGS84.displace(first_coord[0], first_coord[1], n=y_mesh[x, y],
                                                      e=-x_mesh[x, y])
        lat1, lon1 = (lat1 * (180 / math.pi), lon1 * (180 / math.pi))
        lat[x, y] = lat1
        lon[x, y] = lon1

sim2_lat = lat[49, 0]
sim2_lon = lon[49, 0]
print('Start position for Sim 2 in [0,49]: lat: %.6f , lon: %.6f' % (sim2_lat, sim2_lon))

# Socket setup for retrieving the OceanServer data
HOST = 'localhost'
PORT = 28813
BUFSIZE = 1024
ADDR = (HOST, PORT)
ADDR2 = (HOST, PORT + 1)
s = socket(AF_INET, SOCK_STREAM)
try:
    s.connect(ADDR)
    print('Socket 1 online')

except:
    s.connect(ADDR2)
    print('Socket 2 online')



# Find temperature for the area at given time of mission.
wanted_day = 2  # Day 1: Day 2:
wanted_hour = 2000  # 1000 to 2300
temp_true = np.zeros(shape)
temp_pred = np.zeros(shape)
true_temp = np.zeros(shape)

day = 0
tot = 0
for date in range(20170511, 20170513):
    day += 1
    for hour in range(1000, 2400, 100):
        tot += 1
        for x in range(yy):
            for y in range(xx):
                ocean_temp = []

                # Get Data from net-CDF file ---> The Virtual Ocean
                stval = 'salinity,temperature,u_east,v_north lat={} lon={} date={} time={} depth={}'.format(
                    lat[x,y],
                    lon[x,y],
                    date,
                    hour,
                    0)
                s.sendall(stval.encode('ascii'))
                data = s.recv(BUFSIZE)
                data = data.decode("utf-8")
                Oceandata = data.split(",")  # [Salinity, temperature, u_east, v_north]
                ocean_temp.append(float(Oceandata[1]))
                temp_true[x,y] = float(Oceandata[1])

        temp_pred = temp_pred + temp_true

        if day == wanted_day and hour == wanted_hour:
            true_temp = np.array(temp_true)

temp_pred = temp_pred/tot

# Save to files
np.savetxt(str(my_path + '/Textfiles/truetemp.txt'), true_temp)
np.savetxt(str(my_path + '/Textfiles/temppred.txt'), temp_pred)

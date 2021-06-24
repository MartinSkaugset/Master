from netCDF4 import Dataset
from time import strptime
from datetime import timedelta, datetime
from time import sleep
import numpy as np
import sys, traceback
import datetime
import pyproj
import math


class SinmodData:
    
    def __init__(self, fileName):
        self.nf = Dataset(fileName, 'r')
        # Read time information:
        timev = self.nf.variables['time']
        if len(timev.shape) == 1:
            # Assume "days since" format
            t0 = strptime(timev.units[11:], "%Y-%m-%d %H:%M:%S")
            t0 = datetime.datetime(year=t0.tm_year,month=t0.tm_mon,day=t0.tm_mday, \
                                   hour=t0.tm_hour, minute=t0.tm_mon, second=t0.tm_sec)
            values = timev[:]
            self.times = []
            for v in values:
                self.times.append(t0+timedelta(days=v))

            # Model resolution:
            self.dx = self.nf.horizontal_resolution

            # Check if elevation is stored:
            self.hasElevation = "elevation" in self.nf.variables

            # Find the model area's projection:
            cNP = self.nf.coordinate_north_pole
            self.proj = pyproj.Proj(proj='stere', lat_ts=self.nf.standard_parallel, \
                       lat_0=self.nf.latitude_of_projection_origin, \
                       lon_0=self.nf.straight_vertical_longitude_from_pole, \
                       x_0=cNP[0]*self.dx, y_0=cNP[1]*self.dx, \
                       a=6370000, b=6370000)

            # Read information on vertical layers. Store layer depths and mid layer depths:
            lSizes = self.nf.variables['LayerDepths'][:]
            self.layerDepths = [lSizes[0]]
            self.midLayerDepths = [lSizes[0]/2.0]
            for i in range(1,len(lSizes)):
                lSize = lSizes[i]
                self.layerDepths.append(lSize+self.layerDepths[-1])
                self.midLayerDepths.append(self.midLayerDepths[-1] + \
                   0.5*(lSizes[i-1] + lSize))

    def getVariableNames(self):
        return self.nf.variables.keys()

    def hasVariable(self, varname):
        return varname in self.nf.variables.keys()

    def getModelCoordinates(self, lon, lat):
        x,y = self.proj(lon, lat)
        return x/self.dx, y/self.dx

    def getIntModelCoordinates(self, lon, lat):
        x,y = self.proj(lon, lat)
        return int(math.floor(x/self.dx)), int(math.floor(y/self.dx))

    # Return which layer in the model contains the given depth. 1-based indexing:
    def getLayer(self, depth, useMidDepths=False):
        if useMidDepths:
            dpt = self.midLayerDepths
        else:
            dpt = self.layerDepths

        xlo = 0
        xhi = len(dpt)
        count = 0
        while xlo < xhi and count < 200:
            count = count+1
            #print 'xlo = '+str(xlo)+"; xhi = "+str(xhi)
            newx = int(math.floor((xlo+xhi)/2))
            #print "newx = "+str(newx)
            diff = dpt[newx] - depth
            #print "diff = "+str(diff)
            if diff == 0:
                return newx+1
            elif diff > 0:
                xhi = newx
            else:
                xlo = max(newx, xlo+1)
        return xlo+1

    # Get model depth at the given position:
    def getDepth(self, lon, lat):
        ix,iy = self.getIntModelCoordinates(lon, lat)
        return self.nf.variables['depth'][iy,ix]

    def getInterpolatedValue(self, variable, lon, lat, depth, time):
        x,y = self.getModelCoordinates(lon, lat)
        fx = np.floor(x)
        fy = np.floor(y)

        if fx > self.nf.dimensions['xc'].size:
            fx = self.nf.dimensions['xc'].size

        if fy > self.nf.dimensions['yc'].size:
            fy = self.nf.dimensions['yc'].size

        var = self.nf.variables[variable]
        # TODO: handle elevation if available
        offsets = [[-1, -1], [-1, 0], [0, -1], [0, 0]]
        vals = []
        layer = self.getLayer(depth, useMidDepths=True)
        for i in range(0,4):
            myoff = offsets[i]
            if layer == 1:
                vals.append(var[time, layer-1, int(fy)-1+myoff[1], int(fx)-1+myoff[0]])
            else:
                # Interpolate:
                vAbove = var[time,layer-2,int(fy)-1+myoff[1],int(fx)-1+myoff[0]]
                vBelow = var[time,layer-1,int(fy)-1+myoff[1],int(fx)-1+myoff[0]]
                newVal = (((depth-self.midLayerDepths[layer-2])*vBelow + \
                    (self.midLayerDepths[layer-1]-depth)*vAbove)/ \
                    (self.midLayerDepths[layer-1] - self.midLayerDepths[layer-2]))
                
                                    
                #if newVal == np.nan:
                #    print type(newVal)
                vals.append(newVal)
            # No vert inp: vals.append(var[time,layer-1,int(fy)-1+myoff[1],int(fx)-1+myoff[0]])


        # Get distances to the four nearest points:
        eps = 1e-10
        weights = [1/(math.sqrt((x-(fx-0.5))**2 + (y-(fy-0.5))**2)+eps), \
                    1/(math.sqrt((x-(fx-0.5))**2 + (y-(fy+0.5))**2)+eps), \
                    1/(math.sqrt((x-(fx+0.5))**2 + (y-(fy-0.5))**2)+eps), \
                    1/(math.sqrt((x-(fx+0.5))**2 + (y-(fy+0.5))**2)+eps)]
        
        # Get values for the four nearest points:
        #try:
        #    val = [var[time,layer-1,int(fy)-2,int(fx)-2], \
        #               var[time,layer-1,int(fy)-1,int(fx)-2], \
        #               var[time,layer-1,int(fy)-2,int(fx)-1], \
        #               var[time,layer-1,int(fy)-1,int(fx)-1]]
        #except:
        #    print "ERROR, POSSIBLY TRYING TO INTERPOLATE AT EDGE OF MODEL AREA"
        
        sumWV = 0
        sumW = 0
        for i in range(0,4):
            sumW = sumW+weights[i]
            sumWV = sumWV+vals[i]*weights[i]

        return sumWV/sumW

    def getPhysVar(self, lon, lat, depth, time):
        """
        Get interpolated values of current components (east, north), temperature and salinity at the
        given longitude, latitude, depth and sample number.
        """
        return [self.getInterpolatedValue('u_east', lon, lat, depth, time),
                self.getInterpolatedValue('v_north', lon, lat, depth, time),
                self.getInterpolatedValue('temperature', lon, lat, depth, time),
                self.getInterpolatedValue('salinity', lon, lat, depth, time)]


    # Find the last time step equal to or before the given time. Zero-based indexing:
    def getTimeStep(self, time):
        xlo = 0
        xhi = len(self.times)
        count = 0
        while xlo < xhi and count < 200:
            count = count+1
            newx = int(math.floor((xlo+xhi)/2))
            #print "newx = "+str(newx)
            
            if self.times[newx] == time:
                return newx
            elif self.times[newx] > time:
                xhi = newx
            else:
                xlo = max(newx, xlo+1)
        return xlo
        


#####################################################################################3

# if len(sys.argv) > 1:
#     fileName = sys.argv[1]
# else:
#     fileName = 'samples_NSEW_200501.nc'

# myfile = SinmodData(fileName)

# #print myfile.getTimeStep(datetime.datetime(year=2005,month=1,day=1,hour=0,minute=0,second=5))

# #print myfile.getDepth(-0.8706, 59.8358)
# results = []
# depth = 15
# print myfile.getDepth(-0.86, 59.84)
# for i in range(1,100):
#     lat = 59.84 #+ float(i)/50.
#     depth = float(i)*1.12
#     value = myfile.getInterpolatedValue("temperature", -0.86, lat, depth, 0)
#     if np.isnan(float(value)):
#         value = -100
#     results.append(value)


# print results
# #print myfile.getInterpolatedValue("salinity", -0.8706, 59.8358, 85, 0)
# #x,y = myfile.getModelCoordinates(-0.8706, 59.8358)
# #print x/myfile.dx
# #print y/myfile.dx


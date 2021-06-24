import sys
import datetime
from socket import *

from sinmoddata import SinmodData

def parseCommand(data):
    data = str(data)[2:-1]
    parts = data.split(" ", 1)
    adi = {}
    adi['variable'] = parts[0]
    if len(parts) > 1:
        args = parts[1].split(" ")
        for arg in args:
            p = arg.split("=")
            if len(p) > 1:
                if p[0] == 'date' and len(p[1]) == 8:
                    adi["year"] = int(p[1][0:4])
                    adi["month"] = int(p[1][4:6])
                    adi["day"] = int(p[1][6:8])
                elif p[0] == 'time' and len(p[1]) == 4:
                    adi["hour"] = int(p[1][0:2])
                    adi["minute"] = int(p[1][2:4])
                elif p[0] == 'hour':
                    adi["hour"] = int(p[1])
                elif p[0] == 'minute':
                    adi["minute"] = int(p[1])
                elif p[0] == 'lon':
                    adi["lon"] = float(p[1])
                elif p[0] == 'lat':
                    adi["lat"] = float(p[1])
                elif p[0] == 'depth':
                    adi["depth"] = float(p[1])
                else:
                    adi[p[0]] = p[1]
    return adi


# Open data file:
if len(sys.argv) > 1:
    fileName = sys.argv[1]
else:
    fileName = 'samples_2017.05.11.nc'
myfile = SinmodData(fileName)

# Setup:
HOST = 'localhost'
PORT = 28813
BUFSIZE = 1024
ADDR = (HOST, PORT)
ADDR2 = (HOST, PORT+1)

# Establish socket:
srvSock = socket(AF_INET, SOCK_STREAM)
try:
    srvSock.bind(ADDR)
except:
    srvSock.bind(ADDR2)
srvSock.listen(1)
while True:
    print('Waiting for connection...')
    clientSock, addr = srvSock.accept()
    print('Connected from: ', addr)
    keepOn = True
    while keepOn:
        data = clientSock.recv(1024)
        if len(data) == 0:
            print('Connection closed')
            clientSock.close()
            keepOn = False
            continue

        args = parseCommand(data)
        variables = args['variable'].split(',')
        if variables[0] == 'listvar':
            variables = myfile.getVariableNames()
            result = ""
            for var in variables:
                if len(result)>0:
                    result=result+","
                result=result+var

        else:
            sample = myfile.getTimeStep(datetime.datetime(year=args['year'],month=args['month'],day=args['day'],
                                                          hour=args['hour'],minute=args['minute'],second=0))
            result = ""
            for var in variables:
                if not myfile.hasVariable(var):
                    clientSock.sendall("ERROR: variable '"+args['variable']+"' not found")
                    clientSock.close()
                    continue

                value = myfile.getInterpolatedValue(var, args['lon'], args['lat'], args['depth'], sample)
                if len(result)>0:
                    result=result+","
                result=result+str(value)

        clientSock.sendall(result.encode('ascii'))


        



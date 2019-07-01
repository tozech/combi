
from ecmwfapi import ECMWFDataServer, ECMWFService
    
#server = ECMWFDataServer()
server = ECMWFService("mars")


request = {
    "class"   : "od",
    "stream"  : "enfo",
    "expver"  : "1",
    "date"    : "20170801/to/20170831",
    "time"    : "00",
    "step"    : "0/to/48/by/3",
    "type"    : "pf",
    "levtype" : "sfc",
    "param"   : "ssrd/2t/10u/10v",
   " number"  : "1/to/50",
    "grid":"0.2/0.2",
    "area":"56/7/46/12"}

server.execute(request, 'target.grib')

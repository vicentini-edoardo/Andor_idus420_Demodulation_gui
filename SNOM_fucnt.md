Here are some functions and libraries: 

# libraries
import asyncio
import nest_asyncio
import os
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import nea_tools
import matplotlib.pyplot as plt
import h5py

import ctypes
import sys
import math

# definitions
@dataclass
class ScanResult:
    """Dataclass to combine measured data of mirror scan"""
    o0:np.ndarray
    o1:np.ndarray
    o2:np.ndarray
    o3:np.ndarray
    o4:np.ndarray
    o5:np.ndarray
    coordinates:np.ndarray
    """Measured mirror xyz coordinates in nm"""

    def __iter__(self):
        return iter((self.o0,self.o1,self.o2,self.o3,self.o4,self.o5))

@dataclass
class ScanResult_woO:
    coordinates:np.ndarray

# Moving the sample motor downwards (not sure however if it works when tip is in contact...)

from nea_tools.microscope import motors

with motors.Sample() as sample:
    sample.activate()
    sample.move(vz=-0.001) # moving down
    sample.await_movement()
    sample.move(vz=0.001,dt=0.1) # moving up
    Sample.await_movement()

# otherwise from the sdk documentation here are the functions descritptions I found: 

Task ActiveMotorGoRelativeXyzAsync
Moves all axes of the currently selected motor simultaneously (if it has position sensors) relative
to the current position (asynchronously).
Args:
vector (System.Vector3D): Relative coordinates in nm
Returns: Awaitable task with a boolean as result, that indicates True if the new position was
successfully reached, otherwise False.

Task ActiveMotorGotoXyzAsync
Moves the axes of the currently selected motor simultaneously (if it has position sensors) to the
specified position (asynchronously).
Args:
position (System.Point3D): Absolute coordinates in nm
Returns: Awaitable task with a boolean as result, that indicates True if the new position was
successfully reached, otherwise False.

Task GetActiveMotorAsync
Gets the currently activated motor as nea.Motor(asynchronously) which responds to motor
commands.
Returns: Awaitable task with the motor as result.

Task GetActiveMotorDistanceToReferenceXyz
Gets the motor position relative to the reference marks (asynchronously)
Returns: Awaitable task with System.Point3D as result (all axes in nm).

Task MoveActiveMotorXyzAsync
Simultaneously moves the axes of activated motor (asynchronously).
Args:
velocity (System.Vector3D): Each axis of the three axes can have
a velocity between -1.0 and 1.0.
0.0: No movement , or stopps the motor.
1.0: Maximum velocity of motor.
-1.0: Maximum velocity of motor in
opposite direction.
duration (System.TimeSpan): Period of time until motor movement
Stops. 

ReadOnly ActiveMotor
Returns the currently activated stage as nea.Motor. 

Task ActiveMotorGoRelativeXyzAsync
Moves all axes of the currently selected motor simultaneously (if it has position sensors) relative
to the current position (asynchronously).
Args:
vector (System.Vector3D): Relative coordinates in nm
Returns: Awaitable task with a boolean as result, that indicates True if the new position was
successfully reached, otherwise False.


# init neasnom
path_to_dll = ""
fingerprint = None
host = 'nea-server'
   
# nea_tools.set_output(None) # turn off logging
   
# connecting and creating module neaspec on success
loop = asyncio.get_event_loop()
nest_asyncio.apply(loop)
loop.run_until_complete(nea_tools.connect(host,fingerprint,
                                        path_to_dll))

from neaspec import context
from nea_tools.microscope.motors import Mirror

def close_nea():
      nea_tools.disconnect()
      Print('\nDisconnecting')

def read_nea_oxa(harmonic):     
      read=context.Microscope.Py.OpticalAmplitude(harmonic)
    return read
   
def read_nea_mxa(harmonic):     
      read=context.Microscope.Py.MechanicalAmplitude(harmonic)
    return read

## I give here an example of usage: 
for harmonic in range(6):
    amp[harmonic][iy,ix] = read_nea_oxa(harmonic)

amp_result = ScanResult(amp[0],amp[1],amp[2],amp[3],amp[4],amp[5],coords) 

with h5py.File(os.path.join(DIR,f"{FNAME}_{NOW}.h5"),'w') as file:
    for harmonic, (a) in enumerate(zip(amp)):
        file.create_dataset(f"O{harmonic}",data=a)
    file.create_dataset("coordinates",data=amp.coordinates)


## readout of phase values (OpticalPhase and MechanicalPhase) is a bit different and goes via class Stream: 
# Open stream for live phase data
with stream.Stream() as s:
# Phase from stream
stream_key = f"O{harmonic}P"
Try:
    phase[harmonic][iy, ix] = s.data[stream_key][-1]
except Exception as e:
    phase[harmonic][iy, ix] = np.nan
          print(f"Warning: could not read phase for {stream_key}: {e}")

---

# Real Hand Python API Documentation

## API Overview

This document provides a detailed overview of the Python API for the Real Hand, including functions for controlling the hand's movements, retrieving sensor data, and setting operational parameters.


## Public API

### Set Speed
```python
def set_speed(self,speed=[100,100,100,100,100]) # Set speed. Length is 6 for O6/L6, 7 for L7, 10 for L10, otherwise 5
```
**Description**:  
Sets the movement speed of the hand.  
**Parameters**:  
- `speed`: A list containing speed data. The length is 5 elements corresponding to the speed of each joint. If it is L7, it is 7 elements, corresponding to each motor speed. Range of each element value: 0~255.

---

### Set Finger Torque Limit - Force
```python
def set_torque(self, torque=[180,100,80,99,255]) # Set torque. Length is 6 for O6/L6, 7 for L7, 10 for L10, otherwise 5
```
**Description**:  
Sets the torque limit or force of the fingers, used to control gripping force.  
**Parameters**:  
- `torque`: A list containing force data. The length is 5 elements corresponding to the force value of each finger. If it is L7, it is 7 elements, corresponding to each motor force value. Range of each element value: 0~255.

---

### Set Joint Position
```python
def finger_move(self,pose=[120,120,120,120,120,120,120,120,120,120]) # L10 example
```
**Description**:  
Sets the target positions of the joints, used to control finger movement.  
**Parameters**:  
- `pose`: A float-type list containing target position data. Length is 6 for O6/L6, 7 for L7, 10 for L10, 20 for L20, and 25 for L25. Range of each element value: 0~255.

---

### Set Motor Current
```python
def set_current(self, current=[99, 72, 80, 66, 20]) # L20 example
```
**Description**:  
Sets the current value of the motors.  
**Parameters**:  
- `current`: An int-type list containing target current data. Length is 5 elements. Currently only supports the L20 version. Range of each element value: 0~255.

---

### Get Speed
```python
def get_speed(self)
return [180, 200, 200, 200, 200]
```
**Description**:  
Retrieves the currently set speed values. Note: You need to set the joint positions before you can retrieve speed values.

**Returns**:  
- Returns a list containing the current finger speed settings. Range of each element value: 0~255.

---

### Get Current Joint State
```python
def get_state(self)
return [81, 79, 79, 79, 79, 79, 83, 76, 80, 78]
```
**Description**:  
Retrieves the current joint state information as a float-type list. Note: You need to set the joint positions before you can retrieve state information. Length is 6 for O6/L6, 7 for L7, 10 for L10, 20 for L20, and 25 for L25. Range of each element value: 0~255.

**Returns**:  
- Returns a float-type list containing current joint state data. Range of each element value: 0~255.

---

### Get Normal Pressure, Tangential Pressure, Tangential Direction, Proximity Induction
```python
def get_force(self)
return [[255.0, 0.0, 0.0, 77.0, 192.0], [82.0, 0.0, 0.0, 230.0, 223.0], [107.0, 255.0, 255.0, 31.0, 110.0], [255.0, 0.0, 20.0, 255.0, 255.0]]
```
**Description**:  
Retrieves comprehensive hand sensor data, including normal pressure, tangential pressure, tangential direction, and proximity induction.  
**Returns**:  
- Returns a 2D list, where each sub-list contains pressure data for different categories: `[[Normal Pressure], [Tangential Pressure], [Tangential Direction], [Proximity Induction]]`. For each category, elements correspond to: Thumb, Index, Middle, Ring, Pinky.
Range of each element value: 0~255.

---

### Get Version
```python
def get_version(self)
return [10, 6, 22, 82, 20, 17, 0]
```
**Description**:  
Retrieves the current software or hardware version number.  
**Returns**:  
- Returns a list representing the current version number. List elements represent in order: Degrees of Freedom \ Version Number \ Serial Number \ Left Hand (76) or Right Hand (82) \ Internal Serial Number.

---
--------------------------------------------------------------
### Get Torque
```python
def get_torque(self)
return [200, 200, 200, 200, 200]
```
**Description**:  
Retrieves current finger torque list information. Represents the current motor torque for each finger. Supports L20, L25.

**Returns**:  
- Returns a float-type list. Range of each element value: 0~255.

---

### Get Motor Temperature
```python
def get_temperature(self)
return [41, 71, 45, 40, 50, 47, 58, 50, 63, 70]
```
**Description**:  
Retrieves the motor temperature of the current joints.

**Returns**:  
- Returns a list containing the motor temperature of the current joints.

---

### Get Motor Fault Codes
```python
def get_fault(self)
return [0, 4, 0, 0, 0, 0, 0, 0, 0, 0]
```
**Description**:  
Retrieves current joint motor faults. 0 means normal; 1: Current Overload; 2: Over Temperature; 3: Encoding Error; 4: Over/Under Voltage.

**Returns**:  
- Returns a float-type list containing current joint motor faults.

---

### Clear Motor Fault Codes
```python
def clear_faults(self)
```
**Description**:  
Attempts to clear motor faults. No return value. Only supports L20.
**Returns**:  
None
---

## Example Usage

The following is a complete example code showing how to use the API described above:

```python

from RealHand.real_hand_api import RealHandApi
def main():
    # Initialize API. hand_type: left or right. hand_joint: L7, L10, L20, or L25
    real_hand = RealHandApi(hand_type="left", hand_joint="L10")
    # Set finger speed
    real_hand.set_speed(speed=[120,200,200,200,200])
    # Set hand torque
    real_hand.set_torque(torque=[200,200,200,200,200])
    # Get current hand state
    hand_state = real_hand.get_state()
    # Print state values
    print(hand_state)

```

---

## Notes
- Before using the API, please ensure the hand device is correctly connected and initialized.
- Please refer to the device technical manual for specific ranges and meanings of parameter values (such as speed, force, etc.).

---

## Contact
- If you have any questions or need further support, please contact [MAINTAINER_EMAIL]().

---
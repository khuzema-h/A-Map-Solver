# ENPM661 Project 3 Phase 1: Implementation of the A* Algorithm for a Point Robot

# Project Members: Adil Qureshi (122060975) - Khuzema Habib (121288776) - Sounderya Venugopal (121272423)

---

This project implements a pathfinding algorithm using A* algorithm in a 2D environment with obstacles. The environment is represented as an image where obstacles are marked in black and free space is white.

The action set consists of 5 different moves:
- Turn 0 (Moves the Robot forward)
- Turn 30 (Moves the Robot Up by 30 Degrees)
- Turn 60 (Moves the Robot Up by 30 Degrees)
- Turn N30 (Moves the Robot down by 30 Degrees)
- Turn N60 (Moves the Robot down 60 Degrees )

---
# Map 
---
The Map was defined using Half Planes and Semi-Algebraic Equations. The Map has a width of 600 and height of 250, with a clearance of 5 mm defined around the edges and the obstacles. 

![image](https://github.com/user-attachments/assets/a7f9061d-6b6a-4a61-bdf3-cb459f39c1d7)

# Running the Code
---
Ensure the Following Dependencies are installed on your system:
- Python 
- Opencv
- Matplotlib
- Numpy

Running the main file 

```bash
python3 a_star_adil_khuzema_sounderya.py

## User Inputs:
- Step size, Must be within 1-10
- Start Position for x, y, theta, must be within range
- Goal Position for x, y, theta, must be within range
- Do not input float or character values for any inputs

## Code Output

```
Enter step size (1-10): 10
Enter the orientation(theta)(0-360) of your start point: 30
x start(6-594): 10
y start(6-244): 10
theta (0-360): 30
Enter the orientation(theta)(0-360) of your start point: 30
x goal(6-594): 400
y goal(6-244): 200
Solving...
Goal reached!
Animation saved as 'astar_path.mp4'
```
## Final Path Preview
![image](https://github.com/user-attachments/assets/ba04b0fa-1056-4247-8cf7-e580f2c835da)

## Animation of Path Finding

![image](https://github.com/user-attachments/assets/2c1449ce-5c25-4c33-ace1-319a24bcf316)


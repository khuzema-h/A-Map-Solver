# ENPM661 Project 3 Phase 1: Implementation of the A* Algorithm for a Point Robot
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

## Code Output

```
Enter step size (1-10): 10
Enter start position (x, y, theta):
x (0-600): 5
y (0-250): 5
theta (0-360): 0
Enter goal position (x, y):
x (0-600): 100
y (0-250): 10
Solving...
Goal reached!
Path found with 11 steps!
Animation saved as 'astar_path.mp4'
```
## Final Path Preview
![image](https://github.com/user-attachments/assets/ba04b0fa-1056-4247-8cf7-e580f2c835da)

## Animation of Path Finding



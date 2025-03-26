import numpy as np
import matplotlib.pyplot as plt
import time
import math
import cv2
from collections import deque
import heapq
import os

# Define the new map dimensions
map_height = 250
map_width = 600

# Calculate scaling factors
width_scale = map_width / 180
height_scale = map_height / 50

# Create a grid for visualization
x = np.linspace(0, map_width, 1000)  # Increased resolution for better quality
y = np.linspace(0, map_height, 1000)
X, Y = np.meshgrid(x, y)

# Define the shapes using semi-algebraic models with scaled coordinates
def shape_E(x, y):
    # Original: (10, 10) to (23, 35)
    # Scaled coordinates
    vertical = (10*width_scale <= x) & (x <= 15*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    top_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (30*height_scale <= y) & (y <= 35*height_scale)
    middle_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (20*height_scale <= y) & (y <= 25*height_scale)
    bottom_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (10*height_scale <= y) & (y <= 15*height_scale)
    return vertical | top_horizontal | middle_horizontal | bottom_horizontal

def shape_N(x, y):
    # Original: (28, 10) to (43, 35)
    left_vertical = (28*width_scale <= x) & (x <= 33*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    diagonal = (y >= -2.5*height_scale/width_scale * (x - 28*width_scale) + 35*height_scale) & \
               (y <= -2.5*height_scale/width_scale * (x - 33*width_scale) + 35*height_scale) & \
               (28*width_scale <= x) & (x <= 43*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    right_vertical = (38*width_scale <= x) & (x <= 43*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    return left_vertical | diagonal | right_vertical

def shape_P(x, y):
    # Original: (48, 10) to (59, 35)
    vertical = (48*width_scale <= x) & (x <= 53*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    semi_circle = ((x - 53*width_scale)**2 + (y - 29.75*height_scale)**2 <= (8*width_scale)**2) & (x >= 53*width_scale)
    return vertical | semi_circle

def shape_M(x, y):
    # Original: (64, 10) to (92, 35)
    left_vertical = (64*width_scale <= x) & (x <= 69*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    diagonal = (y >= -2.5*height_scale/width_scale * (x - 64*width_scale) + 35*height_scale) & \
               (y <= -2.5*height_scale/width_scale * (x - 69*width_scale) + 35*height_scale) & \
               (64*width_scale <= x) & (x <= 79*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    bottom_horizontal = (75*width_scale <= x) & (x <= 82*width_scale) & (10*height_scale <= y) & (y <= 15*height_scale)
    diagonal_2 = (y >= 2.5*height_scale/width_scale * (x - 92*width_scale) + 35*height_scale) & \
                 (y <= 2.5*height_scale/width_scale * (x - 87*width_scale) + 35*height_scale) & \
                 (79*width_scale <= x) & (x <= 87*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    right_vertical = (87*width_scale <= x) & (x <= 92*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    return left_vertical | diagonal | bottom_horizontal | diagonal_2 | right_vertical

def shape_6(x, y):
    # Original: (97, 10) to (115, 38)
    center_x, center_y = 111*width_scale, 16.5*height_scale
    outer_circle = ((x - center_x)**2 + (y - center_y)**2 <= (11*width_scale)**2) & \
                   (95*width_scale <= x) & (x <= 122*width_scale) & (8*height_scale <= y) & (y <= 50*height_scale)
    inner_circle = ((x - center_x)**2 + (y - center_y)**2 <= (4*width_scale)**2) & \
                   (95*width_scale <= x) & (x <= 115*width_scale) & (10*height_scale <= y) & (y <= 38*height_scale)
    vertical_line = (100*width_scale <= x) & (x <= 105*width_scale) & (17*height_scale <= y) & (y <= 38*height_scale)
    return outer_circle & ~inner_circle | vertical_line 

def shape_1(x, y):
    # Original: (143, 10) to (148, 38)
    vertical_line = (158*width_scale <= x) & (x <= 163*width_scale) & (10*height_scale <= y) & (y <= 38*height_scale)
    return vertical_line

# Combine all shapes
def all_shapes(x, y):
    return shape_E(x, y) | shape_N(x, y) | shape_P(x, y) | shape_M(x, y) | shape_6(x, y) | shape_6(x-25*width_scale, y) | shape_1(x, y)

# Resolution: mm_to_pixels = pixels per millimeter (lower = faster but coarser)
mm_to_pixels = 2  # Adjust as needed (e.g., 5 for high-res, 1 for low-res)
width_pixels = int(map_width * mm_to_pixels)
height_pixels = int(map_height * mm_to_pixels)

# Generate coordinates (vectorized)
x_mm = np.arange(width_pixels) / mm_to_pixels  # X-axis in mm
y_mm = (height_pixels - 1 - np.arange(height_pixels)) / mm_to_pixels  # Y-axis flipped (origin at bottom-left)
X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

# Create obstacle mask
obstacle_mask = all_shapes(X_mm, Y_mm)

# Initialize image (white background)
image = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255
image[obstacle_mask] = 0  # Black for obstacles

# Add clearance (5mm dilation)
clearance_pixels = int(5 * mm_to_pixels)
kernel = np.ones((clearance_pixels, clearance_pixels), np.uint8)
dilated = cv2.dilate(image[:, :, 0], kernel, iterations=1)
image[dilated == 0] = 0  # Expand obstacles

# Display
cv2.imshow('Obstacle Map (White=Free, Black=Obstacle)', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

while True:
    try:
        step_input = int(input("Enter a length for step input between 1-10: "))
        if 1 <= step_input <= 10:
            print("Valid Input, Proceeding ...")
            break
        else:
            print("Invalid Input, please enter a step input size between 1-10")
    except ValueError:
        print("Please enter a valid integer between 1-10")

# Define Action set
def turn_0(node, step_input):
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad)
    new_y = node[1] + step_input * math.sin(theta_rad)
    return (new_x, new_y, node[2])

def turn_30(node, step_input):
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad + math.pi/6)
    new_y = node[1] + step_input * math.sin(theta_rad + math.pi/6)
    new_theta = (node[2] + 30) % 360
    return (new_x, new_y, new_theta)
    
def turn_60(node, step_input): 
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad + math.pi/3)
    new_y = node[1] + step_input * math.sin(theta_rad + math.pi/3)
    new_theta = (node[2] + 60) % 360
    return (new_x, new_y, new_theta)
    
def turn_N30(node, step_input):
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad - math.pi/6)
    new_y = node[1] + step_input * math.sin(theta_rad - math.pi/6)
    new_theta = (node[2] - 30) % 360
    return (new_x, new_y, new_theta)

def turn_N60(node, step_input):
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad - math.pi/3)
    new_y = node[1] + step_input * math.sin(theta_rad - math.pi/3)
    new_theta = (node[2] - 60) % 360
    return (new_x, new_y, new_theta)

action_set = [turn_0, turn_30, turn_60, turn_N30, turn_N60]

# A* Algorithm Implementation
class Node:
    def __init__(self, x, y, theta, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        self.g = 0  # Cost from start to current node
        self.h = 0  # Heuristic cost from current to end node
        self.f = 0  # Total cost (g + h)
    
    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))

def heuristic(node, goal):
    # Euclidean distance
    return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)

def is_valid(node, obstacle_image, mm_to_pixels):
    # Check if node is within bounds
    if (node.x < 0 or node.x >= map_width or node.y < 0 or node.y >= map_height):
        return False
    
    # Check if node is in obstacle space
    x_pixel = int(node.x * mm_to_pixels)
    y_pixel = height_pixels - 1 - int(node.y * mm_to_pixels)  # Flip y-coordinate
    
    if x_pixel < 0 or x_pixel >= width_pixels or y_pixel < 0 or y_pixel >= height_pixels:
        return False
    
    # Check if pixel is black (obstacle)
    if obstacle_image[y_pixel, x_pixel, 0] == 0:
        return False
    
    return True

def discretize_state(node, step_size=5, theta_step=30):
    # Discretize the state space for visited checks
    x = round(node.x / step_size) * step_size
    y = round(node.y / step_size) * step_size
    theta = round(node.theta / theta_step) * theta_step
    return (x, y, theta)

def a_star(start, goal, obstacle_image, action_set, step_input, mm_to_pixels):
    # Initialize open and closed lists
    open_list = []
    closed_list = set()
    
    # Create start and goal nodes
    start_node = Node(start[0], start[1], start[2])
    goal_node = Node(goal[0], goal[1], goal[2])
    
    # Add start node to open list
    heapq.heappush(open_list, start_node)
    
    # For visualization
    visited_nodes = []
    
    # For video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('astar_path.mp4', fourcc, 120, (width_pixels, height_pixels))
    
    # Convert obstacle image to color for visualization
    vis_image = cv2.cvtColor(obstacle_image, cv2.COLOR_BGR2RGB)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        visited_nodes.append(current_node)
        
        # Write current state to video
        frame = vis_image.copy()
        
        # Draw visited nodes
        for node in visited_nodes:
            x_pixel = int(node.x * mm_to_pixels)
            y_pixel = height_pixels - 1 - int(node.y * mm_to_pixels)
            if 0 <= x_pixel < width_pixels and 0 <= y_pixel < height_pixels:
                cv2.circle(frame, (x_pixel, y_pixel), 1, (255, 0, 0), -1)
        
        # Draw current node
        x_pixel = int(current_node.x * mm_to_pixels)
        y_pixel = height_pixels - 1 - int(current_node.y * mm_to_pixels)
        if 0 <= x_pixel < width_pixels and 0 <= y_pixel < height_pixels:
            cv2.circle(frame, (x_pixel, y_pixel), 3, (0, 255, 0), -1)
        
        video_out.write(frame)
        
        # Check if we've reached the goal
        if math.sqrt((current_node.x - goal_node.x)**2 + (current_node.y - goal_node.y)**2) < 5:  # 5mm tolerance
            print("Goal reached!")
            path = []
            current = current_node
            while current is not None:
                path.append((current.x, current.y, current.theta))
                current = current.parent
            
            # Write the final path to video
            path.reverse()
            for i in range(len(path)-1):
                frame = vis_image.copy()
                
                # Draw visited nodes
                for node in visited_nodes:
                    x_pixel = int(node.x * mm_to_pixels)
                    y_pixel = height_pixels - 1 - int(node.y * mm_to_pixels)
                    if 0 <= x_pixel < width_pixels and 0 <= y_pixel < height_pixels:
                        cv2.circle(frame, (x_pixel, y_pixel), 1, (255, 0, 0), -1)
                
                # Draw path
                for j in range(i+1):
                    x1_pixel = int(path[j][0] * mm_to_pixels)
                    y1_pixel = height_pixels - 1 - int(path[j][1] * mm_to_pixels)
                    x2_pixel = int(path[j+1][0] * mm_to_pixels)
                    y2_pixel = height_pixels - 1 - int(path[j+1][1] * mm_to_pixels)
                    if (0 <= x1_pixel < width_pixels and 0 <= y1_pixel < height_pixels and 
                        0 <= x2_pixel < width_pixels and 0 <= y2_pixel < height_pixels):
                        cv2.line(frame, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 0, 255), 2)
                
                video_out.write(frame)
            
            video_out.release()
            return path
        
        # Generate children
        children = []
        for action in action_set:
            new_x, new_y, new_theta = action((current_node.x, current_node.y, current_node.theta), step_input)
            new_node = Node(new_x, new_y, new_theta, current_node)
            
            # Check if valid
            if is_valid(new_node, obstacle_image, mm_to_pixels):
                children.append(new_node)
        
        for child in children:
            # Check if child is in closed list (using discretized state)
            disc_state = discretize_state(child)
            if disc_state in closed_list:
                continue
            
            # Calculate costs
            child.g = current_node.g + step_input
            child.h = heuristic(child, goal_node)
            child.f = child.g + child.h
            
            # Check if child is already in open list and has higher cost
            in_open = False
            for open_node in open_list:
                if discretize_state(open_node) == disc_state and child.g >= open_node.g:
                    in_open = True
                    break
            
            if not in_open:
                heapq.heappush(open_list, child)
        
        # Add current node to closed list (using discretized state)
        closed_list.add(discretize_state(current_node))
    
    video_out.release()
    print("No path found!")
    return None

# Get start and goal positions from user
print("Enter start position (x, y, theta):")
start_x = float(input("x (0-600): "))
start_y = float(input("y (0-250): "))
start_theta = float(input("theta (0-360): "))

print("Enter goal position (x, y):")
goal_x = float(input("x (0-600): "))
goal_y = float(input("y (0-250): "))

start = (start_x, start_y, start_theta)
goal = (goal_x, goal_y, 0)  # Goal orientation doesn't matter

# Run A* algorithm
path = a_star(start, goal, image, action_set, step_input, mm_to_pixels)
print("Solving....")

if path:
    print("Path found with", len(path), "steps!")
    
    # Visualize final path
    final_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw visited nodes (for visualization)
    # Note: In a real implementation, we'd need to keep track of all visited nodes
    
    # Draw path
    for i in range(len(path)-1):
        x1_pixel = int(path[i][0] * mm_to_pixels)
        y1_pixel = height_pixels - 1 - int(path[i][1] * mm_to_pixels)
        x2_pixel = int(path[i+1][0] * mm_to_pixels)
        y2_pixel = height_pixels - 1 - int(path[i+1][1] * mm_to_pixels)
        cv2.line(final_image, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 0, 255), 2)
    
    # Draw start and goal
    start_pixel = (int(start[0] * mm_to_pixels), height_pixels - 1 - int(start[1] * mm_to_pixels))
    goal_pixel = (int(goal[0] * mm_to_pixels), height_pixels - 1 - int(goal[1] * mm_to_pixels))
    cv2.circle(final_image, start_pixel, 5, (0, 255, 0), -1)
    cv2.circle(final_image, goal_pixel, 5, (0, 0, 255), -1)
    
    cv2.imshow('Final Path', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("Animation saved as 'astar_path.mp4'")
else:
    print("No path found!")
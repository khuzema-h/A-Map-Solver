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
x = np.linspace(0, map_width, 1000)
y = np.linspace(0, map_height, 1000)
X, Y = np.meshgrid(x, y)

# Define obstacle shapes
def shape_E(x, y):
    vertical = (10*width_scale <= x) & (x <= 15*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    top_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (30*height_scale <= y) & (y <= 35*height_scale)
    middle_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (20*height_scale <= y) & (y <= 25*height_scale)
    bottom_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (10*height_scale <= y) & (y <= 15*height_scale)
    return vertical | top_horizontal | middle_horizontal | bottom_horizontal

def shape_N(x, y):
    left_vertical = (28*width_scale <= x) & (x <= 33*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    diagonal = (y >= -2.5*height_scale/width_scale * (x - 28*width_scale) + 35*height_scale) & \
               (y <= -2.5*height_scale/width_scale * (x - 33*width_scale) + 35*height_scale) & \
               (28*width_scale <= x) & (x <= 43*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    right_vertical = (38*width_scale <= x) & (x <= 43*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    return left_vertical | diagonal | right_vertical

def shape_P(x, y):
    vertical = (48*width_scale <= x) & (x <= 53*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    semi_circle = ((x - 53*width_scale)**2 + (y - 29.75*height_scale)**2 <= (8*width_scale)**2) & (x >= 53*width_scale)
    return vertical | semi_circle

def shape_M(x, y):
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
    center_x, center_y = 111*width_scale, 16.5*height_scale
    outer_circle = ((x - center_x)**2 + (y - center_y)**2 <= (11*width_scale)**2) & \
                   (95*width_scale <= x) & (x <= 122*width_scale) & (8*height_scale <= y) & (y <= 50*height_scale)
    inner_circle = ((x - center_x)**2 + (y - center_y)**2 <= (4*width_scale)**2) & \
                   (95*width_scale <= x) & (x <= 115*width_scale) & (10*height_scale <= y) & (y <= 38*height_scale)
    vertical_line = (100*width_scale <= x) & (x <= 105*width_scale) & (17*height_scale <= y) & (y <= 38*height_scale)
    return outer_circle & ~inner_circle | vertical_line 

def shape_1(x, y):
    vertical_line = (158*width_scale <= x) & (x <= 163*width_scale) & (10*height_scale <= y) & (y <= 38*height_scale)
    return vertical_line

# Combine all shapes
def all_shapes(x, y):
    return shape_E(x, y) | shape_N(x, y) | shape_P(x, y) | shape_M(x, y) | shape_6(x, y) | shape_6(x-25*width_scale, y) | shape_1(x, y)

# Create obstacle map
mm_to_pixels = 2
width_pixels = int(map_width * mm_to_pixels)
height_pixels = int(map_height * mm_to_pixels)

x_mm = np.arange(width_pixels) / mm_to_pixels
y_mm = (height_pixels - 1 - np.arange(height_pixels)) / mm_to_pixels
X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

# Create image with 3 channels (for color)
image = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255
# Create obstacle mask (original obstacles)
obstacle_mask = all_shapes(X_mm, Y_mm)

# Create edge clearance mask (5mm from edges)
edge_clearance = 5  # mm
edge_mask = (
    (X_mm <= edge_clearance) | 
    (X_mm >= map_width - edge_clearance) | 
    (Y_mm <= edge_clearance) | 
    (Y_mm >= map_height - edge_clearance)
)

# Combine obstacles and edge clearance
total_obstacle_mask = obstacle_mask | edge_mask

# Create image with 3 channels (for color)
image = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255

# Mark original obstacles in black
image[obstacle_mask] = [0, 0, 0]  # Black for obstacles

# Mark edge clearance in red
image[edge_mask] = [0, 0, 255]  # Red for edge clearance

# Create clearance area around obstacles
clearance_pixels = int(5 * mm_to_pixels)
kernel = np.ones((2*clearance_pixels+1, 2*clearance_pixels+1), np.uint8)
dilated = cv2.dilate((obstacle_mask*255).astype(np.uint8), kernel, iterations=1)
obstacle_clearance_area = (dilated > 0) & ~obstacle_mask

# Mark obstacle clearance in Red
image[obstacle_clearance_area] = [0, 0, 255]  # Red for obstacle clearance

# Combine all obstacle areas for path planning
total_obstacle_area = obstacle_mask | obstacle_clearance_area | edge_mask

# Create a binary version for path planning (0 = obstacle, 1 = free)
planning_image = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255
planning_image[total_obstacle_area] = 0

cv2.imshow('Obstacle Map with 5mm Clearance', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get step input
while True:
    try:
        step_input = int(input("Enter step size (1-10): "))
        if 1 <= step_input <= 10:
            break
        print("Invalid input, try again")
    except ValueError:
        print("Please enter a number")

# Action definitions
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


# A* Implementation
class Node:
    def __init__(self, x, y, theta, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0
    
    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def __lt__(self, other):
        return self.f < other.f
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))

def heuristic(node, goal):
    return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)

def is_valid(node, obstacle_image, mm_to_pixels):
    if (node.x < 0 or node.x >= map_width or node.y < 0 or node.y >= map_height):
        return False
    
    x_pixel = int(node.x * mm_to_pixels)
    y_pixel = height_pixels - 1 - int(node.y * mm_to_pixels)
    
    if x_pixel < 0 or x_pixel >= width_pixels or y_pixel < 0 or y_pixel >= height_pixels:
        return False
    
    # Check if pixel is black (original obstacle) or red (clearance)
    if (obstacle_image[y_pixel, x_pixel, 0] == 0 or 
        np.array_equal(obstacle_image[y_pixel, x_pixel], [0, 0, 255])):
        return False
    
    return True

def discretize_state(node, step_size=5, theta_step=30):
    x = round(node.x / step_size) * step_size
    y = round(node.y / step_size) * step_size
    theta = round(node.theta / theta_step) * theta_step
    return (x, y, theta)

def a_star(start, goal, obstacle_image, action_set, step_input, mm_to_pixels):
    open_list = []
    closed_list = set()
    start_node = Node(start[0], start[1], start[2])
    goal_node = Node(goal[0], goal[1], goal[2])
    heapq.heappush(open_list, start_node)
    
    visited_nodes = []
    explored_actions = []
    action_colors = [
        (0, 255, 0),  # 0°
        (0, 255, 0),  # 30° 
        (0, 255, 0),  # 60°
        (0, 255, 0),  # -30°
        (0, 255, 0)   # -60°
    ]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('astar_path.mp4', fourcc, 60, (width_pixels, height_pixels))
    vis_image = cv2.cvtColor(obstacle_image, cv2.COLOR_BGR2RGB)
    arrow_scale = 0.5 + (step_input / 30)
    
    while open_list:
        current_node = heapq.heappop(open_list)
        visited_nodes.append(current_node)
        
        frame = vis_image.copy()
        
        # Draw explored actions
        for action in explored_actions:
            node, end_x, end_y, color_idx = action
            x_pix = int(node.x * mm_to_pixels)
            y_pix = height_pixels - 1 - int(node.y * mm_to_pixels)
            end_x_pix = int(end_x * mm_to_pixels)
            end_y_pix = height_pixels - 1 - int(end_y * mm_to_pixels)
            
            if (0 <= x_pix < width_pixels and 0 <= y_pix < height_pixels and
                0 <= end_x_pix < width_pixels and 0 <= end_y_pix < height_pixels):
                cv2.arrowedLine(frame, (x_pix, y_pix), (end_x_pix, end_y_pix),
                              action_colors[color_idx], 1, tipLength=0.2)
        
        # Draw visited nodes
        for node in visited_nodes:
            x_pix = int(node.x * mm_to_pixels)
            y_pix = height_pixels - 1 - int(node.y * mm_to_pixels)
            if 0 <= x_pix < width_pixels and 0 <= y_pix < height_pixels:
                overlay = frame.copy()
                cv2.circle(overlay, (x_pix, y_pix), 2, (255, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Draw current node
        x_pix = int(current_node.x * mm_to_pixels)
        y_pix = height_pixels - 1 - int(current_node.y * mm_to_pixels)
        if 0 <= x_pix < width_pixels and 0 <= y_pix < height_pixels:
            cv2.circle(frame, (x_pix, y_pix), 4, (0, 255, 0), -1)
        
        # Explore current actions
        current_actions = []
        for i, action in enumerate(action_set):
            new_x, new_y, new_theta = action((current_node.x, current_node.y, current_node.theta), step_input)
            vis_x = current_node.x + (new_x - current_node.x) * arrow_scale
            vis_y = current_node.y + (new_y - current_node.y) * arrow_scale
            
            new_node = Node(new_x, new_y, new_theta)
            if is_valid(new_node, obstacle_image, mm_to_pixels):
                current_actions.append((current_node, vis_x, vis_y, i))
                end_x_pix = int(vis_x * mm_to_pixels)
                end_y_pix = height_pixels - 1 - int(vis_y * mm_to_pixels)
                
                if (0 <= x_pix < width_pixels and 0 <= y_pix < height_pixels and
                    0 <= end_x_pix < width_pixels and 0 <= end_y_pix < height_pixels):
                    thickness = 2 if i == 0 else 1
                    cv2.arrowedLine(frame, (x_pix, y_pix), (end_x_pix, end_y_pix),
                                  action_colors[i], thickness, tipLength=0.3)
        
        explored_actions.extend(current_actions)
        video_out.write(frame)
        
        # Check for goal
        if math.sqrt((current_node.x - goal_node.x)**2 + (current_node.y - goal_node.y)**2) < 5:
            print("Goal reached!")
            path = []
            current = current_node
            while current is not None:
                path.append((current.x, current.y, current.theta))
                current = current.parent
            
            # Draw final path
            path.reverse()
            for i in range(len(path)-1):
                frame = vis_image.copy()
                
                # Draw exploration
                for action in explored_actions:
                    node, end_x, end_y, color_idx = action
                    x_pix = int(node.x * mm_to_pixels)
                    y_pix = height_pixels - 1 - int(node.y * mm_to_pixels)
                    end_x_pix = int(end_x * mm_to_pixels)
                    end_y_pix = height_pixels - 1 - int(end_y * mm_to_pixels)
                    cv2.arrowedLine(frame, (x_pix, y_pix), (end_x_pix, end_y_pix),
                                  action_colors[color_idx], 1, tipLength=0.2)
                
                for node in visited_nodes:
                    x_pix = int(node.x * mm_to_pixels)
                    y_pix = height_pixels - 1 - int(node.y * mm_to_pixels)
                    overlay = frame.copy()
                    cv2.circle(overlay, (x_pix, y_pix), 2, (255, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Draw path
                for j in range(i+1):
                    x1 = int(path[j][0] * mm_to_pixels)
                    y1 = height_pixels - 1 - int(path[j][1] * mm_to_pixels)
                    x2 = int(path[j+1][0] * mm_to_pixels)
                    y2 = height_pixels - 1 - int(path[j+1][1] * mm_to_pixels)
                    if (0 <= x1 < width_pixels and 0 <= y1 < height_pixels and 
                        0 <= x2 < width_pixels and 0 <= y2 < height_pixels):
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                video_out.write(frame)
            
            video_out.release()
            return path
        
        # Generate children
        children = []
        for action in action_set:
            new_x, new_y, new_theta = action((current_node.x, current_node.y, current_node.theta), step_input)
            new_node = Node(new_x, new_y, new_theta, current_node)
            if is_valid(new_node, obstacle_image, mm_to_pixels):
                children.append(new_node)
        
        for child in children:
            disc_state = discretize_state(child)
            if disc_state in closed_list:
                continue
            
            child.g = current_node.g + step_input
            child.h = heuristic(child, goal_node)
            child.f = child.g + child.h
            
            in_open = False
            for open_node in open_list:
                if discretize_state(open_node) == disc_state and child.g >= open_node.g:
                    in_open = True
                    break
            
            if not in_open:
                heapq.heappush(open_list, child)
        
        closed_list.add(discretize_state(current_node))
    
    video_out.release()
    print("No path found!")
    return None

# Get start and goal positions
print("Enter start position (x, y, theta):")
start_x = float(input("x (0-600): "))
start_y = float(input("y (0-250): "))
start_theta = float(input("theta (0-360): "))

print("Enter goal position (x, y):")
goal_x = float(input("x (0-600): "))
goal_y = float(input("y (0-250): "))

start = (start_x, start_y, start_theta)
goal = (goal_x, goal_y, 0)

# Run A* algorithm
print("Solving...")
path = a_star(start, goal, image, action_set, step_input, mm_to_pixels)

# Visualize final result
if path:
    print(f"Path found with {len(path)} steps!")
    final_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Draw path
    for i in range(len(path)-1):
        x1 = int(path[i][0] * mm_to_pixels)
        y1 = height_pixels - 1 - int(path[i][1] * mm_to_pixels)
        x2 = int(path[i+1][0] * mm_to_pixels)
        y2 = height_pixels - 1 - int(path[i+1][1] * mm_to_pixels)
        cv2.line(final_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Mark start and goal
    start_pt = (int(start[0] * mm_to_pixels), height_pixels - 1 - int(start[1] * mm_to_pixels))
    goal_pt = (int(goal[0] * mm_to_pixels), height_pixels - 1 - int(goal[1] * mm_to_pixels))
    cv2.circle(final_image, start_pt, 5, (0, 255, 0), -1)
    cv2.circle(final_image, goal_pt, 5, (0, 0, 255), -1)
    
    cv2.imshow('Final Path', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Animation saved as 'astar_path.mp4'")
else:
    print("No path found!")
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import cv2
from collections import deque
import heapq
import os

            # define map dimensions
map_height = 250
map_width = 600

                    # calculate scaling factors
width_scale = map_width / 180
height_scale = map_height / 50

                            # creating a grid for visualization
x = np.linspace(0, map_width, 1000)
y = np.linspace(0, map_height, 1000)
X, Y = np.meshgrid(x, y)

                    # defining obstacle shapes
def shape_E(x, y):
    vertical = (10*width_scale <= x) & (x <= 15*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    top_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (30*height_scale <= y) & (y <= 35*height_scale)    # follwoing semi algebric equations
    middle_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (20*height_scale <= y) & (y <= 25*height_scale)
    bottom_horizontal = (10*width_scale <= x) & (x <= 23*width_scale) & (10*height_scale <= y) & (y <= 15*height_scale)
    return vertical | top_horizontal | middle_horizontal | bottom_horizontal    # OR operator for merging

def shape_N(x, y):                          # follwoing semi algebric equations
    left_vertical = (28*width_scale <= x) & (x <= 33*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)

    # Diagnoal line definition in letter N
    diagonal = (y >= -2.5*height_scale/width_scale * (x - 28*width_scale) + 35*height_scale) & \
               (y <= -2.5*height_scale/width_scale * (x - 33*width_scale) + 35*height_scale) & \
               (28*width_scale <= x) & (x <= 43*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    right_vertical = (38*width_scale <= x) & (x <= 43*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    return left_vertical | diagonal | right_vertical    # OR operation for merging

def shape_P(x, y):
    vertical = (48*width_scale <= x) & (x <= 53*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)  # Semi-algebric equations
    semi_circle = ((x - 53*width_scale)**2 + (y - 29.75*height_scale)**2 <= (8*width_scale)**2) & (x >= 53*width_scale) #  equation of a circle
    return vertical | semi_circle   

def shape_M(x, y):  # Semi-algebric equations
    left_vertical = (64*width_scale <= x) & (x <= 69*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    diagonal = (y >= -2.5*height_scale/width_scale * (x - 64*width_scale) + 35*height_scale) & \
               (y <= -2.5*height_scale/width_scale * (x - 69*width_scale) + 35*height_scale) & \
               (64*width_scale <= x) & (x <= 79*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    bottom_horizontal = (75*width_scale <= x) & (x <= 82*width_scale) & (10*height_scale <= y) & (y <= 15*height_scale)
    diagonal_2 = (y >= 2.5*height_scale/width_scale * (x - 92*width_scale) + 35*height_scale) & \
                 (y <= 2.5*height_scale/width_scale * (x - 87*width_scale) + 35*height_scale) & \
                 (79*width_scale <= x) & (x <= 87*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    right_vertical = (87*width_scale <= x) & (x <= 92*width_scale) & (10*height_scale <= y) & (y <= 35*height_scale)
    return left_vertical | diagonal | bottom_horizontal | diagonal_2 | right_vertical   # diagnonal lines and OR operations for final letter

def shape_6(x, y):  # digit 6
    center_x, center_y = 111*width_scale, 16.5*height_scale
    outer_circle = ((x - center_x)**2 + (y - center_y)**2 <= (11*width_scale)**2) & \
                   (95*width_scale <= x) & (x <= 122*width_scale) & (8*height_scale <= y) & (y <= 50*height_scale)
    inner_circle = ((x - center_x)**2 + (y - center_y)**2 <= (4*width_scale)**2) & \
                   (95*width_scale <= x) & (x <= 115*width_scale) & (10*height_scale <= y) & (y <= 38*height_scale)
    vertical_line = (100*width_scale <= x) & (x <= 105*width_scale) & (17*height_scale <= y) & (y <= 38*height_scale)
    return outer_circle & ~inner_circle | vertical_line         # defining using vertical lines and circle

def shape_1(x, y):
    vertical_line = (158*width_scale <= x) & (x <= 163*width_scale) & (10*height_scale <= y) & (y <= 38*height_scale)
    return vertical_line        # digit one just has a verticle line

# combining all shapes
def all_shapes(x, y):
    return shape_E(x, y) | shape_N(x, y) | shape_P(x, y) | shape_M(x-2*width_scale, y) | shape_6(x, y) | shape_6(x-30*width_scale, y) | shape_1(x, y)

#Creating the obstacle map and defining initial parameters 
mm_to_pixels = 2        # arbitrary scle factor used for mm to pixel conversion
width_pixels = int(map_width * mm_to_pixels)
height_pixels = int(map_height * mm_to_pixels)  

x_mm = np.arange(width_pixels) / mm_to_pixels       # defining a grind like structure 
y_mm = (height_pixels - 1 - np.arange(height_pixels)) / mm_to_pixels
X_mm, Y_mm = np.meshgrid(x_mm, y_mm)

# creating image with 3 channels (for color)
image = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255
# creating obstacle mask (original obstacles)
obstacle_mask = all_shapes(X_mm, Y_mm)

# Create edge clearance mask (5mm from edges)
edge_clearance = 5  # setting 5mm
edge_mask = (
    (X_mm <= edge_clearance) | 
    (X_mm >= map_width - edge_clearance) | 
    (Y_mm <= edge_clearance) | 
    (Y_mm >= map_height - edge_clearance)
)

# combine obstacles and edge clearance
total_obstacle_mask = obstacle_mask | edge_mask

# create image with 3 channels - color image
image = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255


image[obstacle_mask] = [0, 0, 0]  # black for obstacles


image[edge_mask] = [0, 0, 255]  #red for edge clearance

# Create clearance area around obstacles
clearance_pixels = int(5 * mm_to_pixels)
kernel = np.ones((2*clearance_pixels+1, 2*clearance_pixels+1), np.uint8)
dilated = cv2.dilate((obstacle_mask*255).astype(np.uint8), kernel, iterations=1)    # dialating by 5 mm
obstacle_clearance_area = (dilated > 0) & ~obstacle_mask


image[obstacle_clearance_area] = [0, 0, 255]  # red for obstacle clearance

# Combining all obstacle areas for algorithm
total_obstacle_area = obstacle_mask | obstacle_clearance_area | edge_mask

# converting it into binary
planning_image = np.ones((height_pixels, width_pixels, 3), dtype=np.uint8) * 255
planning_image[total_obstacle_area] = 0

# cv2.imshow('Obstacle Map with 5mm Clearance', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


while True:
    try:
        step_input = int(input("Enter step size (1-10): ")) # getting step size input
        if 1 <= step_input <= 10:       # value between 1 to 10
            break
        print("Invalid input, try again")
    except ValueError:          # validating input type
        print("Please enter a number")
if step_input == 1:
 step_input +=1

 
# Action set definitions
def turn_0(node, step_input):       # zero degree
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad)
    new_y = node[1] + step_input * math.sin(theta_rad)
    return (new_x, new_y, node[2])      # returning new state

def turn_30(node, step_input):      # 30 degree
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad + math.pi/6)
    new_y = node[1] + step_input * math.sin(theta_rad + math.pi/6)
    new_theta = (node[2] + 30) % 360
    return (new_x, new_y, new_theta)    # returning new state
    
def turn_60(node, step_input):  # 60 degree
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad + math.pi/3)  
    new_y = node[1] + step_input * math.sin(theta_rad + math.pi/3)
    new_theta = (node[2] + 60) % 360
    return (new_x, new_y, new_theta)    # returning new state
    
def turn_N30(node, step_input):     # -30 degrees
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad - math.pi/6)
    new_y = node[1] + step_input * math.sin(theta_rad - math.pi/6)
    new_theta = (node[2] - 30) % 360
    return (new_x, new_y, new_theta)    # returning new state

def turn_N60(node, step_input):     # -60 degrees
    theta_rad = math.radians(node[2])
    new_x = node[0] + step_input * math.cos(theta_rad - math.pi/3)
    new_y = node[1] + step_input * math.sin(theta_rad - math.pi/3)
    new_theta = (node[2] - 60) % 360
    return (new_x, new_y, new_theta)        # returning new state

action_set = [turn_0, turn_30, turn_60, turn_N30, turn_N60]


# A* algorithm Implementation
class Node:
    def __init__(self, x, y, theta, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        self.g = 0
        self.h = 0          # defining a node class
        self.f = 0
    
    def __eq__(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def __lt__(self, other):
        return self.f < other.f         # class atributes
    
    def __hash__(self):
        return hash((self.x, self.y, self.theta))

def heuristic(node, goal):
    return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)       # deinfing heuristic fucntion - Euclidean distance in our case

def is_valid(node, obstacle_image, mm_to_pixels):       # function to check if node is with in an obstical or boundary
    if (node.x < 0 or node.x >= map_width or node.y < 0 or node.y >= map_height):
        return False
    
    x_pixel = int(node.x * mm_to_pixels)        # converting back to pixels
    y_pixel = height_pixels - 1 - int(node.y * mm_to_pixels)
    
    if x_pixel < 0 or x_pixel >= width_pixels or y_pixel < 0 or y_pixel >= height_pixels:       # if out of bounds
        return False
    
    # Check if pixel has the color
    if (obstacle_image[y_pixel, x_pixel, 0] == 0 or             # black if obstcal
        np.array_equal(obstacle_image[y_pixel, x_pixel], [0, 0, 255])): # red if clearance
        return False
    
    return True

def discretize_state(node, step_size=5, theta_step=30):     # discretizing the theta and x y in the grid
    x = round(node.x / step_size) * step_size
    y = round(node.y / step_size) * step_size
    theta = round(node.theta / theta_step) * theta_step
    return (x, y, theta)


visited_matrix = np.zeros((500, 1200, 12), dtype=np.uint8)  # 500x1200x12 matrix for duplicate checking based on the logic in slides

def get_visited_index(node):
                                        ### convert node coordinates to visited matrix indices
    x_idx = int(node.x / 0.5)
    y_idx = int(node.y / 0.5)
    theta_idx = int(node.theta / 30) % 12
    return (x_idx, y_idx, theta_idx)        

def mark_visited(node):         #mark a node as visited in the matrix

    x_idx, y_idx, theta_idx = get_visited_index(node)
    if 0 <= x_idx < 500 and 0 <= y_idx < 1200 and 0 <= theta_idx < 12:
        visited_matrix[x_idx, y_idx, theta_idx] = 1     # changing matrix id from 0 to 1 

def is_visited(node):       #check if a node has been visited
  
    x_idx, y_idx, theta_idx = get_visited_index(node)
    if 0 <= x_idx < 500 and 0 <= y_idx < 1200 and 0 <= theta_idx < 12:
        return visited_matrix[x_idx, y_idx, theta_idx] == 1     # changing matrix id from 0 to 1 
    return False        # returning false if alreadu visted

def a_star(start, goal, obstacle_image, action_set, step_input, mm_to_pixels):      # main A star implementation
    open_list = []  # defining open list
   
    start_node = Node(start[0], start[1], start[2])
    goal_node = Node(goal[0], goal[1], goal[2])
    heapq.heappush(open_list, start_node)   
    
    # Clear visited matrix for new search
    global visited_matrix
    visited_matrix.fill(0)
    mark_visited(start_node)        # closed list
    
    # Visualization data structures
    visited_nodes = []      
    explored_actions = []
    visited_nodes_set = set()  

    action_colors = [
        (0, 255, 0),   # 0° (green)
        (0, 200, 200), # 30° (cyan)
        (0, 0, 255),   # 60° (blue)
        (200, 200, 0), # -30° (yellow)
        (255, 0, 0)    # -60° (red)
    ]
    
    # video setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = cv2.VideoWriter('astar_path.mp4', fourcc, 120, (width_pixels, height_pixels))
    vis_image = cv2.cvtColor(obstacle_image, cv2.COLOR_BGR2RGB)
    
    # visualization parameters for animation
    max_actions_to_draw = 1000 
    max_nodes_to_draw = 2000
    arrow_scale = 0.7  
    
    while open_list:
        current_node = heapq.heappop(open_list)
        
        # skip if already processed with better cost
        disc_state = discretize_state(current_node)
        if disc_state in visited_nodes_set:
            continue
        visited_nodes_set.add(disc_state)
        
     
        visited_nodes.append(current_node)
        
        # Create new frame
        frame = vis_image
        
        # draw explored actions 
        recent_actions = explored_actions[-max_actions_to_draw:] if len(explored_actions) > max_actions_to_draw else explored_actions
        for action in recent_actions:
            node, end_x, end_y, color_idx = action
            x_pix = int(node.x * mm_to_pixels)      # converitng into pixels for plotting
            y_pix = height_pixels - 1 - int(node.y * mm_to_pixels)
            end_x_pix = int(end_x * mm_to_pixels)
            end_y_pix = height_pixels - 1 - int(end_y * mm_to_pixels)
            
            if (0 <= x_pix < width_pixels and 0 <= y_pix < height_pixels and
                0 <= end_x_pix < width_pixels and 0 <= end_y_pix < height_pixels):
                cv2.arrowedLine(frame, (x_pix, y_pix), (end_x_pix, end_y_pix),      # drawing arrows 
                              action_colors[color_idx], 1, tipLength=0.3)
        
        # Draw visited nodes 
        recent_nodes = visited_nodes[-max_nodes_to_draw:] if len(visited_nodes) > max_nodes_to_draw else visited_nodes
        for node in recent_nodes:
            x_pix = int(node.x * mm_to_pixels)
            y_pix = height_pixels - 1 - int(node.y * mm_to_pixels)  # drawing visited nodes as circles
            if 0 <= x_pix < width_pixels and 0 <= y_pix < height_pixels:
                cv2.circle(frame, (x_pix, y_pix), 1, (255, 100, 100), -1)  # Light red
        

        
        # explore and visualize current actions
        current_actions = []
        for i, action in enumerate(action_set):
            new_x, new_y, new_theta = action((current_node.x, current_node.y, current_node.theta), step_input)
            vis_x = current_node.x + (new_x - current_node.x) * arrow_scale
            vis_y = current_node.y + (new_y - current_node.y) * arrow_scale     ## getting pixel coordinates for plotting
            
            new_node = Node(new_x, new_y, new_theta)
            if is_valid(new_node, obstacle_image, mm_to_pixels):
                current_actions.append((current_node, vis_x, vis_y, i))
                end_x_pix = int(vis_x * mm_to_pixels)
                end_y_pix = height_pixels - 1 - int(vis_y * mm_to_pixels)
                
                if (0 <= x_pix < width_pixels and 0 <= y_pix < height_pixels and
                    0 <= end_x_pix < width_pixels and 0 <= end_y_pix < height_pixels):
                    thickness = 2 if i == 0 else 1
                    cv2.arrowedLine(frame, (x_pix, y_pix), (end_x_pix, end_y_pix),  # daring arrows
                                  action_colors[i], thickness, tipLength=0.3)
        
        explored_actions.extend(current_actions)        # appening already explored nodes
        video_out.write(frame)
        
        # Check for goal
        if math.sqrt((current_node.x - goal_node.x)**2 + (current_node.y - goal_node.y)**2) < 1.5 * mm_to_pixels:       ### Threshhold for goal node. 1.5 as given in slides
            print("Goal reached!")
            path = []
            current = current_node
            while current is not None:
                path.append((current.x, current.y, current.theta))
                current = current.parent
            
            # animate final path
            path.reverse()
            for i in range(len(path)-1):
                frame = vis_image
                

                
                # Draw path so far
                for j in range(i+1):
                    x1 = int(path[j][0] * mm_to_pixels)
                    y1 = height_pixels - 1 - int(path[j][1] * mm_to_pixels)
                    x2 = int(path[j+1][0] * mm_to_pixels)
                    y2 = height_pixels - 1 - int(path[j+1][1] * mm_to_pixels)
                    if (0 <= x1 < width_pixels and 0 <= y1 < height_pixels and 
                        0 <= x2 < width_pixels and 0 <= y2 < height_pixels):        # drawing final path
                        cv2.waitKey(1)
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Thick red
                
                video_out.write(frame)
            
            # final frame with complete path
            frame = vis_image
            for i in range(len(path)-1):
                x1 = int(path[i][0] * mm_to_pixels)
                y1 = height_pixels - 1 - int(path[i][1] * mm_to_pixels)
                x2 = int(path[i+1][0] * mm_to_pixels)
                y2 = height_pixels - 1 - int(path[i+1][1] * mm_to_pixels)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Mark start and goal
            start_pt = (int(start[0] * mm_to_pixels), height_pixels - 1 - int(start[1] * mm_to_pixels))
            goal_pt = (int(goal[0] * mm_to_pixels), height_pixels - 1 - int(goal[1] * mm_to_pixels))
            cv2.circle(frame, start_pt, 8, (0, 255, 0), -1)  # Green start
            cv2.circle(frame, goal_pt, 8, (255, 0, 0), -1)   # Blue goal
            
            video_out.write(frame)  # writing out in video
            video_out.release()
            return path
        
        # generate children
        children = []
        for action in action_set:
            new_x, new_y, new_theta = action((current_node.x, current_node.y, current_node.theta), step_input)
            new_node = Node(new_x, new_y, new_theta, current_node)
            if is_valid(new_node, obstacle_image, mm_to_pixels):    # checking if not in obstcal
                children.append(new_node)
        
        for child in children:
            if is_visited(child):   # chekcing if already visted
                continue
                
            child.g = current_node.g + step_input
            child.h = heuristic(child, goal_node)   # calcukating costs
            child.f = child.g + child.h
            
           
            in_open = False
            for open_node in open_list:
                if discretize_state(open_node) == discretize_state(child) and child.g >= open_node.g:
                    in_open = True
                    break
            
            if not in_open:
                heapq.heappush(open_list, child)    # using heap 
                mark_visited(child)     # marking as visited child
    
    video_out.release()
    print("No path found!")
    return None


while True:
    start_x = int(input("\nEnter the x-coordinate(6-594) of your start point: "))
    start_y = int(input("Enter the y-coordinate(6-244) of your start point: "))         # entring input points
    start_theta = int(input("Enter the orientation(theta)(0-360) of your start point: "))

    if not (5 < start_x < 595 and 5 < start_y < 245 and 0 <= start_theta <= 360):
        print("Your values are not within the image boundaries. Please re-enter the start coordinates.")    # validating points
        continue

    # Convert to pixel coordinates and invert y-axis
    x_pixel = int(start_x * mm_to_pixels)
    y_pixel = height_pixels - 1 - int(start_y * mm_to_pixels)

    # Check if the point is valid (not black or red)
    if (x_pixel < 0 or x_pixel >= width_pixels or y_pixel < 0 or y_pixel >= height_pixels):
        print("Invalid coordinates. Please re-enter.")
        continue

    if (np.array_equal(planning_image[y_pixel, x_pixel], [0, 0, 0]) or 
        np.array_equal(planning_image[y_pixel, x_pixel], [0, 0, 255])):
        print("Your start is on an obstacle or clearance area. Please re-enter the start coordinates.")
    else:
        print("Your start coordinates are correct. Search in progress PLEASE WAIT...")
        break
    
while True:
    goal_x = int(input("\nEnter the x-coordinate(6-594) of your goal point: "))
    goal_y = int(input("Enter the y-coordinate(6-244) of your goal point: "))
    goal_theta = int(input("Enter the orientation(theta)(0-360) of your start point: "))

    if not (5 < goal_x < 595 and 5 < goal_y < 245 and 0 <= goal_theta <= 360):
        print("Your values are not within the image boundaries. Please re-enter the goal coordinates.")
        continue

    # Convert to pixel coordinates and invert y-axis
    x_pixel = int(goal_x * mm_to_pixels)
    y_pixel = height_pixels - 1 - int(goal_y * mm_to_pixels)

    # Check if the point is valid (not black or red)
    if (x_pixel < 0 or x_pixel >= width_pixels or y_pixel < 0 or y_pixel >= height_pixels):
        print("Invalid coordinates. Please re-enter.")
        continue

    if (np.array_equal(planning_image[y_pixel, x_pixel], [0, 0, 0]) or 
        np.array_equal(planning_image[y_pixel, x_pixel], [0, 0, 255])):
        print("Your goal is on an obstacle or clearance area. Please re-enter the goal coordinates.")
    else:
        print("Your goal coordinates are correct. Search in progress PLEASE WAIT...")
        break


start = (start_x, start_y, start_theta)
goal = (goal_x, goal_y, goal_theta)

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

import cv2
import numpy as np
import heapq
import math
from generate_map import generate_map

# Heuristic function (Manhattan distance)
def heuristics(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Action set: directions based on 30° increments, corresponding to angles
angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# Constant step size
STEP_SIZE = 5  # You can change this value as needed

# Convert angle to x, y movement direction
def angle_to_direction(angle, step_size):
    # Convert angle to radians
    angle_rad = math.radians(angle)
    dx = int(round(step_size * math.cos(angle_rad)))  # x direction
    dy = int(round(step_size * math.sin(angle_rad)))  # y direction
    return dx, dy

# Function to convert continuous (x, y, theta) to discrete grid coordinates
def continuous_to_discrete(x, y, theta, x_threshold=0.5, y_threshold=0.5, theta_threshold=30):
    # Round x and y to the nearest threshold
    x_discrete = int(round(x / x_threshold))
    y_discrete = int(round(y / y_threshold))
    # Round theta to the nearest threshold (30 degrees intervals)
    theta_discrete = int(round(theta / theta_threshold)) % 12  # There are 12 possible theta values
    print(theta_discrete)
    return x_discrete, y_discrete, theta_discrete

# A* algorithm with action set and constant step size
# A* algorithm with action set and constant step size
def a_star(start, goal, grid, video_writer, V, x_threshold=0.5, y_threshold=0.5, theta_threshold=30):
    rows, cols = len(grid), len(grid[0])
    open_list = []
    heapq.heappush(open_list, (0, start[0], start[1], 0))  # Start node: (f_cost, x, y, theta)
    
    g_cost = {start: 0}
    f_cost = {start: heuristics(start, goal)}
    
    came_from = {}
    goal_radius = 2
    while open_list:
        _, x, y, theta = heapq.heappop(open_list)
        
        # Convert the continuous coordinates (x, y, theta) to discrete grid coordinates
        x_discrete, y_discrete, theta_discrete = continuous_to_discrete(x, y, theta, x_threshold, y_threshold, theta_threshold)
        
        # Check if the current node has been visited in the matrix V
        if V[x_discrete][y_discrete][theta_discrete] == 1:
            continue  # Skip if already visited
        
        # Mark the node as visited
        V[x_discrete][y_discrete][theta_discrete] = 1
        
        # Draw the current node in the search space (to visualize the search process)
        image_copy = image  # Use a copy to avoid modifying the original image
        
        # Color the visited node blue
        cv2.rectangle(image_copy, (y * cell_size, x * cell_size), ((y + 1) * cell_size, (x + 1) * cell_size), (255, 0, 0), -1)
        
        # Draw the vector in the search space (the direction of the move)
        vector_start = (y * cell_size + cell_size // 2, x * cell_size + cell_size // 2)  # Center of the current cell
        dx, dy = angle_to_direction(theta, STEP_SIZE)  # Get the direction based on angle
        vector_end = (vector_start[0] + dx * cell_size, vector_start[1] + dy * cell_size)  # Calculate the end of the vector
        
        # Draw the arrow to represent the movement direction
        # cv2.arrowedLine(image_copy, vector_start, vector_end, (0, 255, 255), 2, tipLength=0.05)  # Yellow arrow
        
        # Display the updated image with the vector
        cv2.imshow("Map with Search Vectors", image_copy)
        cv2.waitKey(1)
        video_writer.write(image_copy)  # Save the frame to video
        
        if  heuristics((x, y), goal) <= goal_radius:
            path = []
            while (x, y) in came_from:
                path.append((x, y))
                x, y = came_from[(x, y)]
            path.reverse()
            # Draw the final path in green with arrows
            for i in range(len(path) - 1):
                start_point = path[i]
                end_point = path[i + 1]
                start_pixel = (start_point[1] * cell_size + cell_size // 2, start_point[0] * cell_size + cell_size // 2)
                end_pixel = (end_point[1] * cell_size + cell_size // 2, end_point[0] * cell_size + cell_size // 2)
                cv2.arrowedLine(image, start_pixel, end_pixel, (0, 255, 0), 2, tipLength=0.2)

            cv2.imshow("Final Path", image)
            video_writer.write(image)  # Save the final frame to video
            return path
        
        # Continue the search for the goal
        for angle in angles:  # Loop through action set of angles
            dx, dy = angle_to_direction(angle, STEP_SIZE)  # Use constant step size
            new_x, new_y = x + dx, y + dy
            
            if 0 <= new_x < rows and 0 <= new_y < cols and grid[new_x][new_y] == 0:  # Check if valid move
                new_position = (new_x, new_y, angle)
                tentative_g_cost = g_cost[(x, y)] + STEP_SIZE  # Add step size to cost
                if (new_x, new_y) not in g_cost or tentative_g_cost < g_cost[(new_x, new_y)]:
                    came_from[(new_x, new_y)] = (x, y)
                    g_cost[(new_x, new_y)] = tentative_g_cost
                    f_cost[(new_x, new_y)] = tentative_g_cost + heuristics((new_x, new_y), goal)
                    heapq.heappush(open_list, (f_cost[(new_x, new_y)], new_x, new_y, angle))
    
    return None

# Load and preprocess the map
image = generate_map()

# Get image dimensions
height, width = image.shape[:2]

# Cell size in pixels (e.g., 3x3 pixels)
cell_size = 2

# Calculate the number of rows and columns
rows = height // cell_size
cols = width // cell_size

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold to find obstacles (black and red pixels)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)

# Convert the image to a grid (using the image size and cell size)
grid = np.zeros((rows, cols))  # Rows x Columns grid

# Map the obstacles (black or red) to the grid
for i in range(rows):
    for j in range(cols):
        # Each grid cell is 3x3 pixels
        cell = thresh[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
        if np.sum(cell) > 0:  # Check if there are obstacles in this cell
            grid[i][j] = 1  # Blocked cell

# Initialize the visited matrix V of size (500 x 1200 x 12)
V = np.zeros((500, 1200, 12), dtype=int)

# User input for start and end (using mouse click)
def click_event(event, x, y, flags, param):
    global start, end
    
    if event == cv2.EVENT_LBUTTONDOWN:
        row, col = y // cell_size, x // cell_size
        if start is None:
            start = (row, col)
            print(f"Start set to: {start}")
        elif end is None:
            end = (row, col)
            print(f"End set to: {end}")
    
    if start and end:
        # Set up video writer to save animation as a video file
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Video codec (XVID)
        video_writer = cv2.VideoWriter('search_space.avi', fourcc, 120, (width, height))  # Create a video file
        
        if not video_writer.isOpened():
            print("Error: Unable to open video writer.")
        
        path = a_star(start, end, grid, video_writer, V)
        
        # Release video writer after the animation is complete
        video_writer.release()

# Initialize start and end
start, end = None, None

# Show the image and wait for clicks
cv2.imshow("Map", image)
cv2.setMouseCallback("Map", click_event)

cv2.waitKey(0)
cv2.destroyAllWindows()

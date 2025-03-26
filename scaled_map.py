import numpy as np
import matplotlib.pyplot as plt

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

# Evaluate the shapes on the grid
Z = all_shapes(X, Y)

# Plot the shapes
plt.figure(figsize=(18, 8))  # Adjusted figure size for new dimensions
plt.imshow(Z, extent=(0, map_width, 0, map_height), origin='lower', cmap='binary', alpha=0.8)
plt.title("Scaled Map with Obstacles")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
plt.show()
import numpy as np
import cv2

def generate_map():
        
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
        return shape_E(x, y) | shape_N(x, y) | shape_P(x, y) | shape_M(x-2*width_scale, y) | shape_6(x, y) | shape_6(x-30*width_scale, y) | shape_1(x, y)

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
    image[edge_mask] = [0, 0, 0]  # Red for edge clearance

    # Create clearance area around obstacles
    clearance_pixels = int(5 * mm_to_pixels)
    kernel = np.ones((2*clearance_pixels+1, 2*clearance_pixels+1), np.uint8)
    dilated = cv2.dilate((obstacle_mask*255).astype(np.uint8), kernel, iterations=1)
    obstacle_clearance_area = (dilated > 0) & ~obstacle_mask

    # Mark obstacle clearance in Red
    image[obstacle_clearance_area] = [0, 0, 0]  # Red for obstacle clearance

    return image
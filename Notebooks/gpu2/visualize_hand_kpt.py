import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# Function to create a more realistic synthetic high-resolution hand image
def create_realistic_hand_image(image_size=(300, 300)):
    image = np.ones((image_size[0], image_size[1], 3), dtype=np.uint8) * 255

    # Draw palm
    palm_center = (150, 200)
    palm_radius = 60
    cv2.circle(image, palm_center, palm_radius, (200, 150, 100), -1)  # Palm

    # Draw fingers
    finger_length = 50
    finger_width = 15
    finger_offsets = [(-30, -50), (-15, -50), (0, -50), (15, -50), (30, -50)]

    for offset in finger_offsets:
        finger_start = (palm_center[0] + offset[0], palm_center[1] - palm_radius)
        finger_end = (finger_start[0], finger_start[1] - finger_length)
        cv2.rectangle(image, finger_start, (finger_end[0] + finger_width, finger_end[1]), (200, 100, 100), -1)

    return image

# Function to generate synthetic keypoints for hands
def generate_hand_keypoints():
    return np.array([
        [120, 150],  # Finger 1 tip
        [135, 150],  # Finger 2 tip
        [150, 150],  # Finger 3 tip
        [165, 150],  # Finger 4 tip
        [180, 150],  # Finger 5 tip
        [150, 200],  # Palm center
    ])

# Directory to save images
output_dir = "hand_images"
os.makedirs(output_dir, exist_ok=True)

# Generate and save synthetic high-resolution hand images and keypoints
for i in range(4):  # Generate 4 high-resolution images
    hand_image = create_realistic_hand_image(image_size=(300, 300))
    keypoints = generate_hand_keypoints()

    # Save image with keypoints
    plt.figure()
    plt.imshow(hand_image)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], color='red', s=10)  # Plot keypoints in red
    plt.title(f"Realistic Hand Image {i + 1}")
    plt.axis('off')
    
    # Save the figure
    image_path = os.path.join(output_dir, f"realistic_hand_image_{i + 1}.png")
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
    plt.close()  # Close the figure to free memory

print(f"Images saved in directory: {output_dir}")

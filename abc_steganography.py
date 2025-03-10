import numpy as np
import cv2
import random
import skimage
import matplotlib.pyplot as plt

selected_pixels = []

# Define the end flag to mark the end of the message
END_FLAG = "###"

# Convert message to binary
def message_to_binary(message):
    # message += END_FLAG  
    return ''.join(format(ord(char), '08b') for char in message)

# Convert binary to text
def binary_to_text(binary_message):
    chars = [chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8)]
    return ''.join(chars).split(END_FLAG)[0]

# Artificial Bee Colony (ABC) algorithm for pixel selection
def abc_pixel_selection(image, num_bees=20, max_iterations=50):
    height, width, _ = image.shape
    best_positions = []
    
    bees = [(random.randint(0, height-1), random.randint(0, width-1), random.randint(0, 2)) for _ in range(num_bees)]

    for _ in range(max_iterations):
        scores = []
        for bee in bees:
            y, x, channel = bee
            pixel_value = image[y, x, channel]
            score = abs(128 - pixel_value)  # Mid-tone pixels are better for hiding
            scores.append((score, bee))

        scores.sort()
        best_positions = [bee for _, bee in scores[:num_bees // 2]]

        for i in range(len(best_positions)):
            y, x, channel = best_positions[i]
            new_y = min(max(y + random.randint(-1, 1), 0), height - 1)
            new_x = min(max(x + random.randint(-1, 1), 0), width - 1)
            best_positions[i] = (new_y, new_x, channel)

        for i in range(len(best_positions), num_bees):
            best_positions.append((random.randint(0, height-1), random.randint(0, width-1), random.randint(0, 2)))

    return best_positions

# Embed the message in the image
def embed_message(image, message):
    binary_message = message_to_binary(message)
    data_index = 0
    stego_image = image.copy()

    global selected_pixels
    selected_pixels = abc_pixel_selection(image, num_bees=len(binary_message))
    
    # print(selected_pixels)
    
    for y, x, channel in selected_pixels:
        if data_index < len(binary_message):
            bit = int(binary_message[data_index])
            stego_image[y, x, channel] = (stego_image[y, x, channel] & ~1) | bit  # Modify LSB
            data_index += 1
        else:
            break

    return stego_image

# Extract the hidden message
def extract_message(stego_image):
    binary_message = ""
    # selected_pixels = abc_pixel_selection(stego_image, num_bees=500)
    # selected_pixels = puski
    # print(selected_pixels)
    for y, x, channel in selected_pixels:
        binary_message += str(stego_image[y, x, channel] & 1)

        if len(binary_message) >= 8 * len(END_FLAG):
            extracted_text = binary_to_text(binary_message)
            # print(extracted_text)
            # if END_FLAG in extracted_text:
            #     print("nee bondha")
            #     return extracted_text

    return extracted_text

# Compute the number of pixel differences
def pixel_difference(image1, image2):
    diff = np.sum(image1 != image2)
    return diff

# Compute Structural Similarity Index (SSIM)
def compute_ssim(image1, image2):
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return skimage.metrics.structural_similarity(image1_gray, image2_gray)

# Compare histograms of original and stego images
def compare_histograms(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# Load the image
image_path = "image.jpg"
image = cv2.imread(image_path)
if image is None:
    print(f" Error: Image '{image_path}' not found! Please check the filename and path.")
    exit()

# Get the secret message from the user
message = input("Enter the secret message to hide: ")

# Embed the message
stego_image = embed_message(image, message)

# Save the stego image
stego_image_path = "stego_image.png"
cv2.imwrite(stego_image_path, stego_image)
print(f" Stego image saved as '{stego_image_path}'")

# Extract and print the message
extracted_text = extract_message(stego_image)
print(f" Extracted Message: {extracted_text}")

# Compare images
pixel_diff = pixel_difference(image, stego_image)
ssim_score = compute_ssim(image, stego_image)
histogram_similarity = compare_histograms(image, stego_image)

print(f" Pixel Difference: {pixel_diff}")
print(f" SSIM Score: {ssim_score:.4f} (1 = identical, 0 = completely different)")
print(f" Histogram Similarity: {histogram_similarity:.4f} (1 = identical, 0 = different)")

# Display both images side by side
original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
stego_rgb = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(stego_rgb)
plt.title("Stego Image")

plt.show()

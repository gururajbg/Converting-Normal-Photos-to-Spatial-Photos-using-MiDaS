import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to create a mask simulating missing parts in the image
def create_inpainting_mask(image, mask_size=(50, 50)):
    mask = np.ones(image.shape[:2], np.uint8) * 255  # OpenCV uses 255 for visible regions
    h, w = image.shape[:2]
    mask[h // 2 - mask_size[0] // 2:h // 2 + mask_size[0] // 2,
    w // 2 - mask_size[1] // 2:w // 2 + mask_size[1] // 2] = 0  # Mark center area as missing
    return mask


# Function to perform inpainting with OpenCV
def inpaint_image_opencv(image_path):
    img = cv2.imread(image_path)
    mask = create_inpainting_mask(img)

    # Perform inpainting using OpenCV's INPAINT_TELEA or INPAINT_NS
    inpainted_img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Display results
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(inpainted_img, cv2.COLOR_BGR2RGB))
    plt.title('Inpainted Image')
    plt.show()

    return inpainted_img


# Run inpainting on an example image
inpainted_image = inpaint_image_opencv('../data/test.jpg')


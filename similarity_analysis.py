import cv2
from sma import CompareImages

image_1 = cv2.imread('path/to/image_1')
image_2 = cv2.imread('path/to/image_2')

sim_analyzer = CompareImages(image_1, image_2)
image_difference = sim_analyzer.compare_image()

print(image_difference)
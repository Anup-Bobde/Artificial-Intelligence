# Reading,writing & displaying image using imread function(cv2)
# Press Q to exit from output window
import cv2 

input = cv2.imread(r'C:\Users\Anup\VS Code_DataAnalytics\Haarcascade_Projects/elephant.jpg')

cv2.imshow('Test Elephant Image', input)  # Test Elephant Image will be title of output window
cv2.waitKey()
cv2.destroyAllWindows()

# Import numpy
import numpy as np

print(input.shape)

# Let's print each dimension of the image

print('Height of Image:', int(input.shape[0]), 'pixels')
print('Width of Image: ', int(input.shape[1]), 'pixels')

# Simply use 'imwrite' specificing the file name and the image to be saved
cv2.imwrite('output.jpg', input)
cv2.imwrite('output.png', input)
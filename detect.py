import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow   // Since i am doiing the code in google colab

quantization_factor = 16  # quantization value for dct
t_sim = 5          # threshold distance based on similarity 
t_distance = 10    # threshold distance between pixels 
vector_limit = 10  # shift vector elimination limit
block_counter = 0  # to track the count of blocks
block_size = 8     # choosing 8 as ideal size
image = cv2.imread('forged1.jpg')  #reading the image

/// After reding the image covert the image into grayscale image
    Question arises why?
    There are mainly three benefits of converting the original image to the grayscale image-
    1. Dimension reduction (3D image to 2D image ).
    2. It reduces the model complexity.
    3. There are some specific algorithms which works only on grayscale image in opencv.
 ///
 
gray_scale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)     # converting image to it's gray scale image
arr1 = []
arr = np.array(gray_scale)                               # storing in array
predicted_mask = np.zeros((arr.shape[0], arr.shape[1]))  # generating predicted mask
col = arr.shape[1] - block_size                       # saving column values
row = arr.shape[0] - block_size                          # saving row values
dct = np.empty((((col+1)*(row+1)), quantization_factor+2))

///  Here we apply dct and Zigzag traversal
Apply DCT on each block and the apply quantization.     
The feature vector is rearranged into row vector using zigzag scanning. Zig-zag scanning converts 2D matrix into a
1D array (row vectors).
///    
   
print("Zig zag scanning & dct starting...")

for i in range(0, row):           # traversing in row
    for j in range(0, col):       # traversing in col  

        block_val = arr[i:i+block_size, j:j+block_size]
        val = np.float32(block_val) / 255.0  # float conversion/scale
        dcts = cv2.dct(val)  # the dct 
        block_val = np.uint8(np.float32(dcts) * 255.0 ) # converting back
        # zigzag scanning
        result = [[] for k in range(block_size + block_size - 1)]
        for k in range(block_size):                  # traversing block by block
            for l in range(block_size):
                sum_val = k + l
                if (sum_val % 2 == 0):
                    # adding at beginning
                    result[sum_val].insert(0, block_val[k][l])
                else:
                    # adding at end of the list
                    result[sum_val].append(block_val[k][l])

        for item in range(0,(block_size*2-1)):
            arr1 += result[item]

        arr1 = np.asarray(arr1, dtype=np.float)          # converting the input to an array
        arr1 = np.array(arr1[:16])
        arr1 = np.floor(arr1/quantization_factor)        # return the floor of input element_wise
        arr1 = np.append(arr1, [i, j])                   # append values to end of the array

        np.copyto(dct[block_counter], arr1)

        block_counter += 1
        arr1 = []

print("scanning & dct over!")  

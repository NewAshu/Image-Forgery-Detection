# Image-Forgery-Detection

Requirements:
Python3 (3.7.4 Recommended)
OpenCV (pip install opencv-python)

Deviding the whole project in following modules :

1. Reading and converting to grayscale image
2. Partitioning image and Apply DCT to each bloack
3. Rearrange the DCT coefficient and apply quantization
4. Apply Eucledian operation and generate similarity array
5. Perform elimination 
6. Performing the localization (i.e. generating the prediction mask)
7. Showing the prediction mask
8. Calculation of accuracy for this method





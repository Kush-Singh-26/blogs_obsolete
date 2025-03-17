---
layout: post
title: "1. OpenCV Images"
date: 2025-03-15
tags: [OpenCV]
---
# Image Handling

- importing libraries

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
```

## Display Images directly
- `Image(filename = "<image path>")`

## Reading Images
- `retval = cv2.imread( filename[, flags] )`
    - `filename` is a mandatory argument
    - `Flags` is optional
        - used to read image in a particular format (colour, grayscale, with alpha channel)
        - **DEFAULT** : `1` = color image = `cv2.IMREAD_COLOR`
        - `0` = grayscale mode = `cv2.IMREAD_GRAYSCALE`
        - `-1` = loads image as such with alpha channel = `cv2.IMREAD_UNCHANGED`

    - `print(retval)` will print the pixel values of the image 

## Image Attributes
```python
# print the size  of image
print("Image size (H, W) is:", retval.shape)

# print data-type of image
print("Data type of image is:", retval.dtype)
```

## Using Matplotlib to display images
- `plt.imshow(retval, cmap = "gray")`

## Reversing colour channel
- OpenCV stores images in `BGR` format.
- But, matplotlib expects the images in `RGB` format.
- Therfore reversing the channels of image.

```python
ogimg = cv2.imread("image.png", 1)
img-channel-rev = ogimg[:, :, ::-1]  # array[start:stop:step]
```
## Splitting and Merging colour channels
- `cv2.split()` Divides a multi-channel array into several single-channel arrays.

- `cv2.merge()` Merges several arrays to make a single multi-channel array. All the input matrices must have the same size.

```python
img = cv2.imread("image.jpg", 1)

b, g, r = cv2.split(img_nz)

plt.subplot(141);plt.imshow(r, cmap="gray");plt.title("Red Channel")
plt.subplot(142);plt.imshow(g, cmap="gray");plt.title("Green Channel")
plt.subplot(143);plt.imshow(b, cmap="gray");plt.title("Blue Channel")

imgMerged = cv2.merge((b, g, r))
```
- `plt.subplot(141)` -> `plt.subplot(rows, cols, index)`
    - rows : no. of rows
    - cols : no. of columns
    - index : postiton of the subplot

## Converting to different colour space

- `dst = cv2.cvtColor(src, code)`
    - 2 required arguments
        - `src` : input image
        - `code` : color space conversion code
            - `cv2.COLOR_BGR2RGB` : convert to RGB format 
            - `cv2.COLOR_BGR2HSV` : convert to HSV format
    
- `img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
    - images in OpenCV are stored as **BRG** format.
    - `cv2.COLOR_BRG2RGB` code is used to convert it to **RGB**

> Individual channels can be modified by diff. operations line +, -, ..

## Saving Image
- `cv2.imwrite(filename, img[, params])`
    - 2 arguments
        - `img` : image or images to be saved
        - `filname` : path
- saves the image to the specified file

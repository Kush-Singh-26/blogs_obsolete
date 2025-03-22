---
layout: post
title: "2. OpenCV: Basic Image Manipulation"
date: 2025-03-15
tags: [OpenCV]
---
# Basic Image Manipulation

```python
img = cv2.readim("image.png",0)
```

## Accessing Individual Pixels
- `print(img[0][0])` = `print(img[0, 0])`
- `print(img[:, 6])` = `print(img[:][6])`

## Modifying Image Pixels
```python
img_cp = img.copy()
img_cp[2, 3] = 200
img_cp[3, 3] = 500
```

## Cropping Images
- selecting a specific (pixel) region of the image
``` python
img_cropped = img[200:400, 300:600]
# or
img_cropped = img[200:400, 300:600, :]
```
- selects rows from 200 to 399 and columns from 300 to 599.

## Resizing images
- resizes the image `src` down to or up to the specified size. The size and type are derived from the `src`, `dsize`, `fx`, and `fy`.
- `dst = resize( src, dsize[, dst[, fx[, fy[, interpolation]]]] )`
    - `dst` : ouput image
    - 2 required arguments
        - `src` : imput image
        - `dsize` : output image size
    - optional arguments
        - `fx` : scales along the horizontal axis
        - `fy` : scales along the vertical axis

### Methods to `resize` :
1. **Specifying Scaling Factor using fx and fy**
    - `resized_img-2x = cv2.resize(img, None, fx=2, fy=2)`

2. **Specifying exact size of the output image**
```python
desired_width = 100
desired_height = 200
dim = (desired_width, desired_height)

img_resized = cv2.resize(img, dsize=dim, interpolation=cv2.INTER_AREA)
```
- `interpolation=cv2.INTER_AREA` : used to downsize / shrink an image

## Flipping Images
- `dst = cv.flip( src, flipCode )`
    - `dst` : output array of the same size and type as `src`
    - `src` : input image
    - `filpCode` : a flag to specify how to flip the array
        -  `0` means flipping around the x-axis 
        -  `positive value` (for example, 1) means flipping around y-axis
        - `Negative value` (for example, -1) means flipping around both axes.

- `img_flip = cv2.flip(img, 1)`

> Collab Notebook [here](https://github.com/Kush-Singh-26/Learning-Opencv/blob/main/OpenCV_2_Basic_image_manipulation.ipynb) 
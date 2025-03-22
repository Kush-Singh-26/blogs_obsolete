---
layout: post
title: "4. OpenCV: Image Enhancement"
date: 2025-03-18
tags: [OpenCV]
---

# Image Enhancement

Performing mathematical operations can be used to perform image processing.

## Addition and Subtraction

- Adddition of each pixel of image with some number results in the image being brigther.

```python
matrix = np.ones(img.shape, dtype="uint8") * 50

img_brighter = cv2.add(img, matrix)
```

- Subtraction of each pixel of image with some number results in the image being darker.

```python
matrix = np.ones(img.shape, dtype="unit8") * 50

img_darker = cv2.subtract(img, matrix)
```

## Multiplication (Contrast)

- multiplication can be used to improve the contrast of the image.
- **Contrast** : difference in the intensity values of the pixels of an image.
- Multiplying the intensity values with a constant can make the difference larger or smaller ( if multiplying factor is < 1 ).

```python
matrix1 = np.ones(img.shape) * 0.8
matrix2 = np.ones(img.shape) * 1.2

img_darker   = np.uint8(cv2.multiply(np.float64(img), matrix1))
img_brighter = np.uint8(cv2.multiply(np.float64(img), matrix2))
```
- After multiplying, the values which are already high, become greater than 255
    - This cause and overflow issue.

- Therefore, use `np.clip`
    - It will clip the pixel values between 0 and 255.
```python
img_higher = np.uint8(np.clip(cv2.multiply(np.float64(img), matrix2), 0, 255))
```

## Image Threshold 

- **Image Masks** : allows to process on specific parts of an image keeping the other parts intact. 
- _Image Thresholding_ is used to create *Binary Images* from grayscale images.
- _Binary Images_ are used to create _masks_. 

### Global Threshold

```python
retval, dst = cv2.threshold( src, thresh, maxval, type[, dst] )
```
- `src` : input array
- `dst` : output array
- `thresh` : threshold value
- `maxval` : maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
- `type` : thresholding type

### AdaptiveThreshold

```python
dst = cv.adaptiveThreshold( src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst] )
```

- `maxValue` : Non-zero value assigned to the pixels for which the condition is satisfied

- `adaptiveMethod` : Adaptive thresholding algorithm to use. 

- `blockSize` : Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and so on.

- `C`: Constant subtracted from the mean or weighted mean (see the details below). Normally, it is positive but may be zero or negative as well.

## Bitwise Operations

- Given 2 images `img1` and `img2`

```python
dst = cv2.bitwise_and( src1, src2[, dst[, mask]] )
```

- Operations available :
    - `cv2.bitwise_and()`
    - `cv2.bitwise_or()`
    - `cv2.bitwise_xor()` 
    - `cv2.bitwise_not()`


> Collab Notebook [here](https://github.com/Kush-Singh-26/Learning-Opencv/blob/main/OpenCV_4_Image_Enhancement.ipynb) 
    
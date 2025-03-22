---
layout: post
title: "3. OpenCV: Image Annotation"
date: 2025-03-18
tags: [OpenCV]
---

# Annotations

## Types of Annotation

1. Draw lines
2. Draw Circle
3. Draw rectangles
4. Add text

### Draw Lines

- `imgout = cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])`
    - `img` : input image on which annotations have to be performed
    - The line segment to be drawn will have 2 vertexes : `(x1,y1)` & `(x2,y2)`
    - `pt1` : first point `(x1,y1)`
    - `pt2` : second point `(x2,y2)`
    - `color` : colour of the line segment being drawn
    - `thickness` : Integer specifying the line thickness.
        -  Default value is 1.
    - `lineType` : Type of the line. 
        - Default value is 8 which stands for an 8-connected line. 
        - Usually, `cv2.LINE_AA` (antialiased or smooth line) is used for the lineType. 
            - other types are : `cv2.LINE_8` , `cv2.LINE_4`    
### Draw Circles

- `img = cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])`
    - `thickness` : if negative value is given, it will result ina filled circle.

### Drawing Rectangle

- `img = cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])`
    - `pt1` : top left vetex
    - `pt2` : bottom right-vertex 

### Adding text

- `img = cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])`
    - `org` : Bottom-left corner of the text string in the image.
    - `fontFace`: Font type
    - `fontScale`: Font scale factor that is multiplied by the font-specific base size.
        - if negative then upside down text
    
> Collab Notebook [here](https://github.com/Kush-Singh-26/Learning-Opencv/blob/main/OpenCV_3_Image_Annotation.ipynb) 
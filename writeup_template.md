# **Finding Lane Lines on the Road** 

## Term 1 Project for Udacity Self-Driving Car Nanodegree

### Be able to find the lane is an extremely important task for any self-driving car. This project explores 2 possible way for the car to detect the lane using jsut a camera and a lightweight video processing.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[imageGray]: ./examples/grayscale.jpg "Grayscale"
[imageGrayBlur]: ./examples/grayscale.jpg "Grayscale"
[imageCanny]: ./examples/grayscale.jpg "Grayscale"
[imageRegionCanny]: ./test_images_output/region5.png "Grayscale"
[imageLinesCanny]: ./examples/grayscale.jpg "Grayscale"
[imageFinalCanny]: ./examples/grayscale.jpg "Grayscale"
[imageWhiteMask]: ./test_images_output/white5.png "Grayscale"
[imageYellowMask]: ./test_images_output/yellowHSV5.png  "Yellow HSV Mask"
[imageColorMask]: ./test_images_output/mixed5.png "Grayscale"
[imageLinesColorMask]: ./test_images_output/mixed5.png "Grayscale"
[imageRegionColorMask]: ./test_images_output/region5.png "Grayscale"
[imageRegion]: ./test_images_output/regionFull5.png "Grayscale"
[imageFinalColorMask]: ./test_images_output/final5.png "Grayscale"

---

### Reflection

### 1. Pipelines

The project has the implementation of 2 distinct pipelines. One using canny edge detection from a blurred gray image and another that uses a white color mask mixed with a yellow color mask. 

**1.1 Canny Edge Detection Pipeline**

The canny edge method was implemented in the following way:
1. Convert the image to grayscale
2. Pass the grayscale image in a gaussian blur filter
3. Extract the edges using canny function with lower and upper parameters as 200 and 255 respectively.
4. Find the lines using HoughLinesP function with rho = 1, theta = pi/180, threshold = 40, minimum line lenght = 15 and max gap = 15
5. Find the equation of the lines and extend those lines to almost fill the lane
6. Take the mean of the points of the extendeds lines
7. Draw one line with the mean of the points

This method was good with the two videos, but not good ebough for the challenge video, where a bright part make it loose information and darker parts gives too much edges. 

So the method using color masks was implemented and tested, as explained below.

**1.2 Color Masks Detection Pipeline**

This pipeline consist in extract the white and yellow information from the image restrict the area of interest and find the lines that lays within its points.

The first step is to use the inRange method to filter only a certain range of colors within the image. The chosen range is from  [200, 200, 200] to [255, 255, 255] as seen in the code below.

```python
def white_mask(original):
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])

    mask = cv2.inRange(original, lower_white, upper_white)
    return mask
```
The result is an image similar to the image below.

![alt text][imageWhiteMask]

The second step is to extract yellow information from the image, this was a little trickier than the white mask, as it was not givan a good result using RGB/BGR image. The solution was to convert the image from BGR to HSV (Hue, Saturation Value) and then use the range from [10, 100, 100] to [40, 255, 255]) on the hsv image. The code below and the image shows the result obtained by this method.

```python
def yellow_mask(original):
    # Taken from http://aishack.in/tutorials/tracking-colored-objects-opencv/
    HSV = cv2.cvtColor(original, cv2.COLOR_RGB2HSV)

    lowerHSV = np.array([10, 100, 100])
    upperHSV = np.array([40, 255, 255])

    maskHSV = cv2.inRange(HSV, lowerHSV, upperHSV)
    return maskHSV
```
![alt text][imageYellowMask]

Combining these 2 images we got and color mask as seen in this image below.

![alt text][imageColorMask]

The next step was to let only the region on the image that really had important information for the lane detection, using the vertices shown in the code snippet bellow, we were able to select the perfect region of the image.
```python
 vertices = np.array([[
        (0, imshape[0]),
        (imshape[1]*5/100, imshape[0]),
        (imshape[1]*48/100, imshape[0]*57/100),
        (imshape[1]*52/100, imshape[0]*57/100),
        (imshape[1]*95/100, imshape[0]),
        (imshape[1], imshape[0]),
    ]
    ], dtype=np.int32)
```
The image below shows the mask on the original image e the later one on the colormask image.
![alt text][imageRegion]
![alt text][imageRegionColorMask]

With this image beeing the input of the HoughLinesP function of opencv we could extract the lines. Inside the function 'processLines', the lines were separated as beeing a positive slope (left side) and a negative slope (right side). It does seen awkward to have positive slope as left side, but remember that the image left top corner is point (0,0) and not the left bottom point as we normally plot the line. 

One method we are using to make it a little more robust is to take the mean of the extend lines points and draw only 1 line in the image as seen in he snippet below. 


```python
if len(extendedRightLines) > 0:
        rightLine[0] = 0
        rightLine[1] = 0
        rightLine[2] = 0
        for lines in extendedRightLines:
            rightLine[0] += lines[1]
            rightLine[1] += lines[2]
            rightLine[2] += lines[3]

        rightLine[0] /= len(extendedRightLines)
        rightLine[1] /= len(extendedRightLines)
        rightLine[2] /= len(extendedRightLines)

    if len(extendedLeftLines) > 0:
        leftLine[0] = 0
        leftLine[1] = 0
        leftLine[2] = 0
        leftLine[3] = 0
        for lines in extendedLeftLines:
            leftLine[0] += lines[0]
            leftLine[1] += lines[1]
            leftLine[2] += lines[2]
            leftLine[3] += lines[3]
        leftLine[0] /= len(extendedLeftLines)
        leftLine[1] /= len(extendedLeftLines)
        leftLine[2] /= len(extendedLeftLines)
        leftLine[3] /= len(extendedLeftLines)

    # Draw Right Line
    cv2.line(line_img, (0, int(rightLine[0])), (int(rightLine[1]),
                                                int(rightLine[2])), (0, 0, 255), 8, cv2.LINE_AA)
    cv2.line(line_img, (int(leftLine[0]), int(leftLine[1])),
             (int(leftLine[2]), int(leftLine[3])), (0, 0, 255), 8, cv2.LINE_AA)
```

Last we blend thiresulting image to the original image as shown in the picture bleow.

![alt text][imageFinalColorMask]

this method, at least on the videos tested, presented as a much more robust lane detection method.

### 2. Potential shortcomings with the current pipeline

1. The main potential problem it would be not finding the colors, because our eyes can see color does no exists, because our adapt to expand what we see as a whole, but computer vision uses pixels. 

2. Another perk would be a lot of lines no being in the lane, making the mean of the points to deviate from the lane.

### 3. Suggest possible improvements to your pipeline

1. A good way to prevent would maybe try a more adaptive algorithm for finding the lanes, based on some changes in color value and saturation. 

2. Instead of finding the mean, we could find some dominance on the lines, trying to prevent a simple mean error on several lines.

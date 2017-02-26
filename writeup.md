#**Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./report/gray.jpg "Grayscale"
[image2]: ./examples/GaussianBlur.jpg "GaussianBlur"
[image3]: ./examples/canny.jpg "Canny Edge"
[image4]: ./examples/Boundingbox.png "Bounding Box"
[image5]: ./examples/raw.jpg "Raw Hough Transform"
[image6]: ./examples/cont.jpg "Continuous Line"



---

### Reflection

###1. Pipeline Description

My pipeline consisted of 6 steps. 
1. I converted the images to grayscale
![alt text][image1]
2. I applied the Gaussian Blur, with kernel size = 11, to the image to decrease the noise
![alt text][image2]
3. I used the Canny Edge Algorithm to find the edges in the image. The low and high thresholds are set to 60 and 180 respectively.
![alt text][image3]
4. Constrained a bounding box in the image. 
![alt text][image4]
5. Applied Hough Transform to find line segments.
![alt text][image5]
6. Modified the draw_lines() function in order to draw a single line on the left and right lanes. 
    I first separated left and right lane by their slopes. For left and right groups , I did the following respectively:
    * Averaged all the slopes within the group, let's call it "Avg1"
    * Filtered out unwanted slopes by: iterated all the slopes again, but only kept track of slope within the window of Avg1 plus minus a threshold, which I set to 0.35 here.
    * Then searched for the longest line segment, extrapolated to the top and bottom using the average slope.
![alt text][image6]




###2. Potential shortcomings with my current pipeline
1. My pipeline has trouble dealing with more complicated situations. It cannot identify lines for the challenge video.
2. The bounding box is set mannually. So it won't adapt automatically if the camera angle is not pointing to the center.


###3. Suggest possible improvements to your pipeline

1. Create a more robust algorithm that is able to adapt to different scenarios.
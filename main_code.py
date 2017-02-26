import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#%matplotlib inline

#Helper Functions

def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    #print stats and show image
    #print('This image is:', type(image), 'with dimensions:', image.shape)
    #plt.imshow(image)

    # Convert to grayscale
    gray = grayscale(image)
    #plt.imshow(gray, cmap='gray')

    # Apply GaussianBlur
    kernel_size = 11
    blur_gray = gaussian_blur(gray, kernel_size)
    #plt.imshow(blur_gray, cmap='gray')

    # Canny Edge
    edges = canny(blur_gray, 50,150)
    #plt.imshow(edges, cmap='gray')

    # Constrain Area of Interest
    imshape = image.shape
    vertices = np.array([[(0,imshape[0]),(490,320),(510,320),(imshape[1],imshape[0])]])
    mask = region_of_interest(edges, vertices)
    #plt.imshow(mask)

    # Hough Transform
    rho = 2
    theta = np.pi/180
    threshold = 15
    min_line_len = 5
    max_line_gap = 5
    line_image=hough_lines(mask, rho, theta, threshold, min_line_len, max_line_gap)

    color_edges = np.dstack((edges,edges,edges))
    line_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    #line_edges = cv2.cvtColor(line_edges, cv2.COLOR_RGB2BGR)
    return line_edges

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=20):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    '''
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    '''

    s_plus1 = s_minus1 = s_plus2 = s_minus2 =0
    i = j = 0
    temp1 = temp2 = 0
    xedge_minus = yedge_minus =0


    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            #print(x1,y1,x2,y2,slope)
            if (slope >= 0):  #opposite because of opposite y-axis
                s_plus1 += slope
                i += 1
                if (x2-x1 > temp1):
                    xedge_plus = x1
                    yedge_plus = y1
                    temp1 = x2 -x1
            else:
                s_minus1 += slope
                j += 1
                if (x2-x1 > temp2):
                    xedge_minus = x1
                    yedge_minus = y1
                    temp2 = x2 -x1

    s_plus1 = s_plus1/i
    s_minus1 = s_minus1/j
    #print('s_plus = , s_minus = ', s_plus1, s_minus1)

    # Filter
    th = 0.35
    i = j = 0
    for line in lines:
        for x1,y1,x2,y2 in line:
            slope = ((y2-y1)/(x2-x1))
            if (slope >= 0):
                if (slope >= s_plus1 - th and slope <= s_plus1 + th):
                    s_plus2 += slope
                    i += 1
            if (slope <= 0):
                if (slope >= s_minus1 - th and slope <= s_minus1 + th):
                    s_minus2 += slope
                    j += 1
    s_plus = s_plus2/i
    s_minus = s_minus2/j
    #print('s_plus = , s_minus = ', s_plus, s_minus)
    #print(xedge_plus,yedge_plus,xedge_minus,yedge_minus)
    #print('xmin1, xmin2' , x_min1,y_min1,x_min2,y_min2)

    xi_plus = int(xedge_plus + (540-yedge_plus) / s_plus)
    xi_minus = int(xedge_minus - (yedge_minus-540) / s_minus)
    #print('xi_plus, xi_minus' , xi_plus,xi_minus)
    ye = 350
    xe_plus = int(xi_plus - (540-ye)/s_plus)
    xe_minus = int(xi_minus - (540-ye) / s_minus)
    cv2.line(img, (xi_plus, 540), (xe_plus, ye), color, thickness)
    cv2.line(img, (xi_minus, 540), (xe_minus, ye), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)


# Main Code

files = os.listdir("test_images/")
jpgs=[]
jpgs += [each for each in files if each.endswith('jpg')]

print (jpgs)
for jpg in jpgs:
    #reading image
    file_name='test_images/'+jpg
    image = mpimg.imread(file_name)

    #print stats and show image
    print('This image is:', type(image), 'with dimensions:', image.shape)

    line_edges = process_image(image)

    line_edges = cv2.cvtColor(line_edges, cv2.COLOR_RGB2BGR)
    #plt.imshow(line_edges)
    #cv2.imwrite('test_images/solidWhiteRight_lane.jpg',line_edges)
    filename = 'test_images/results/' + jpg
    cv2.imwrite(filename,line_edges)


#white_output = 'white.mp4'
#clip1 = VideoFileClip("solidWhiteRight.mp4")
#white_output = 'yellow.mp4'
#clip1 = VideoFileClip("solidYellowLeft.mp4")
white_output = 'extra.mp4'
clip1 = VideoFileClip("challenge.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)
HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

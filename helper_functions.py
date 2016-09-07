import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def PLOT_IMAGE(image, figure_num=1, show=False):
    '''
    Parameters
    ------------
    image : Numpy array
        Image to be shown
    figure_num : (Optional) int
        Figure number
    show : (Optional) bool
        If true shows O/P immediately, else you need to type plt.show()

    Returns
    ------------
    None
    '''
    plt.figure(figure_num)
    plt.imshow(image)
    plt.axis("off") # Don't show coordinate axis
    if(show is True): # Show image if asked
        plt.show()
    return None


def GINPUT_ROUTINE(image, num_pts=-1):
    '''
    Get coordinates of points in image, click middle click to end
    If num_pts == -1, then choose points indefinately until middle click of mouse

    Parameters
    ------------
    image : Numpy array
        Image where points are to be selected

    Returns
    ------------
    coordinates : Numpy array
        Numpy array containing the coordinates of the points (row,col) format
    '''
    PLOT_IMAGE(image, show=False)
    print("Please select" + str(num_pts) + "points")
    # timeout : time(sec) to wait until termination, if input not given
    coordinates = plt.ginput(n=num_pts, timeout=0)
    coordinates = np.array(coordinates)
    coordinates[:, 0], coordinates[:, 1] = coordinates[:, 1], coordinates[:, 0].copy() # Exchange col1 and col2 to get (row_coordinate, col_coordinate)

    return coordinates.astype(np.uint8) # Return +ve integers only


def RESIZE_IMAGE(image1, fx1, fy1):
    '''
    Function to resize the image

    Parameters:
    ------------
    image1 : Numpy array
        Image to be resized
    fx1 : float
        Horizontal stretch (>0.1)
    fy1 : float
        Vertical stretch (>0.1)

    Returns
    ------------
    Resized image
    '''
    if(image1 != None):
        return cv2.resize(image1, (0, 0), fx=fx1, fy=fy1)
    else:
        print("Image incorrect/Image is NONE")


def PLOT_IMAGE_CV(img,  wait_time=0, window_name="name"):
    '''
    Show's image on OpenCV format

    Parameters
    ------------
    img : Numpy array
        Image to show
    window_name : (Optional) String
        Window name
    wait_time : (Optional) int
        Wait time before closng the window automatically

    Returns
    ------------
    None
    '''
    cv2.imshow(window_name, img)
    cv2.waitKey(wait_time) and 0xFF


def PLOT_COLOR_HISTOGRAM(img, show = True, color = ('b','g','r')):
    '''
    Plot color histogram of image

    Parameters
    ------------
    img : Numpy array
        Image whose histogram is to be calculated
    show : (Optional) bool
        If true shows O/P immediately, else you need to type plt.show()
    color : (Optional) Tuple of strings
        Colors to be used

    Returns
    ------------
    None
    '''
    color = ('blue','green','red')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col, label = str(color[i]) )
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        max_pixel_intensity = max(np.max(img[:,:,0]),np.max(img[:,:,1]), np.max(img[:,:,2]))
        min_pixel_intensity = max(np.min(img[:,:,0]), np.min(img[:,:,1]), np.min(img[:,:,2]))
        plt.xlim([min_pixel_intensity, max_pixel_intensity])
        plt.legend()
    if (show is True):
        plt.show()


def PLOT_GRAY_HISTOGRAM(img, show = True):
    '''
    Plot color histogram of image

    Parameters
    ------------
    img : Numpy array
        Image whose histogram is to be calculated
    show : (Optional) bool
        If true shows O/P immediately, else you need to type plt.show()
    color : (Optional) Tuple of strings
        Colors to be used

    Returns
    ------------
    None
    '''
    color = ('k')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col, label='Frequency count')
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Frequency")
        max_pixel_intensity = max(0,np.max(img[:,:]))
        min_pixel_intensity = min(0,np.min(img[:,:]))
        plt.xlim([min_pixel_intensity, max_pixel_intensity])
        plt.legend()
    if (show is True):
        plt.show()

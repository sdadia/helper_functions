''' This file is a package contains helper functions for opencv and imgs '''


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


def PLOT_img_MAT(img, figure_num=1, show=False):
    '''
    Show img matplotlib style

    Parameters
    ------------
    img : Numpy array
        img to be shown
    figure_num : (Optional) int
        Figure number
    show : (Optional) bool
        If true shows O/P immediately, else you need to type plt.show()

    Returns
    ------------
    None
    '''
    plt.figure(figure_num)
    plt.imshow(img)
    plt.axis("off") # Don't show coordinate axis

    if(show is True): # Show img if asked
        plt.show()
    return None


def GINPUT_ROUTINE(img, num_pts=-1):
    '''
    Get coordinates of points in img, click middle click to end
    If num_pts == -1, then choose points indefinately until middle click of mouse

    Parameters
    ------------
    img : Numpy array
        img where points are to be selected

    Returns
    ------------
    coordinates : Numpy array
        Numpy array containing the coordinates of the points (row,col) format
    '''
    PLOT_img(img, show=False)
    print("Please select" + str(num_pts) + "points")
    coordinates = plt.ginput(n=num_pts, timeout=0) # timeout : time(sec) to wait until termination, if input not given
    coordinates = np.array(coordinates)
    coordinates[:, 0], coordinates[:, 1] = coordinates[:, 1], coordinates[:, 0].copy() # Exchange col1 and col2 to get (row_coordinate, col_coordinate)

    return coordinates.astype(np.uint8) # Return +ve integers only


def RESIZE_img(img, fx1, fy1):
    '''
    Function to resize img

    Parameters
    ------------
    img1 : Numpy array
        img to be resized
    fx1 : float
        Horizontal stretch (>0.1)
    fy1 : float
        Vertical stretch (>0.1)

    Returns
    ------------
    Resized img
    '''
    if(img != None):
        return cv2.resize(img, (0, 0), fx=fx1, fy=fy1)
    else:
        print("img incorrect/img is NONE")


def PLOT_img_CV(img,  wait_time=0, window_name="name"):
    '''
    Show's img on OpenCV format

    Parameters
    ------------
    img : Numpy array
        img to show
    window_name : (Optional) String
        Window name
    wait_time : (Optional) int
        Wait time before closng the window automatically. If zero, then waits indefinately for user input

    Returns
    ------------
    None
    '''
    cv2.imshow(window_name, img)
    cv2.waitKey(wait_time) and 0xFF


def PLOT_COLOR_HISTOGRAM(img, show = True, color = ('b','g','r')):
    '''
    Plot color histogram of img

    Parameters
    ------------
    img : Numpy array
        img whose histogram is to be calculated
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
    Plot color histogram of img

    Parameters
    ------------
    img : Numpy array
        img whose histogram is to be calculated
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

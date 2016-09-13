''' This file is a package contains helper functions for opencv and imgs '''


import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import time


def PLOT_IMG_MAT(img, figure_num=1, show=True):
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
    # PLOT_IMG_MAT(img, show=True)
    plt.figure()
    plt.imshow(img)
    print("Please select" + str(num_pts) + "points")
    coordinates = plt.ginput(n=num_pts, timeout=0) # timeout : time(sec) to wait until termination, if input not given
    coordinates = np.array(coordinates)
    # Exchange col1 and col2 to get in the form (row_coordinate, col_coordinate)
    # Advance slicing
    coordinates[:,[0, 1]] = coordinates[:,[1, 0]]
    coordinates = np.floor(coordinates) # Floor to make them integers

    return coordinates.astype(int) # Return coordinates as integers


def RESIZE_IMG(img, fx1, fy1):
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


def PLOT_IMG_CV(img,  wait_time=0, window_name="name"):
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


def SOBEL(gimg):
    '''
    Return `Sobel` edge derivative image

    Parameters
    ------------
    gimg : 8 bit image
        8 bit gray scale image or 8 bit single channel image

    Returns
    ------------
    GMag : Numpy Array
        Numpy array of Magnitude of resultant X and Y gradient
    GMag_Norm : Numpy array
        Numpy array of Normalized Magnitude of resultant X and Y gradient, for viewing purpose.
     '''

    scale = 1
    delta = 0
    ddepth = cv2.CV_32F
    # Computing the X- and Y-Gradients, using the Sobel kernel
    grad_x = cv2.Sobel(
        gimg,
        ddepth,
        1,
        0,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(
        gimg,
        ddepth,
        0,
        1,
        ksize=3,
        scale=scale,
        delta=delta,
        borderType=cv2.BORDER_DEFAULT)

    # Absolute Gradient for Display purposes --- Remove in future, not needed
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Gradient magnitude computation  --- Magnitude of the field --- Also for display
    g1 = grad_x * grad_x
    g2 = grad_y * grad_y
    GMag = np.sqrt(g1 + g2)  # Actual magnitude of the gradient

    # Normalized gradient 0-255 scale, and 0-1 scale --- For Display
    GMag_Norm = np.uint8(GMag * 255.0 / Gmag.max())  # Magnitude, for display

    return GMag, GMag_Norm


class TIMERS:
    '''
    Timer class to time functions. Interface same as matlab (TIC and TOC)

    Attributes
    ------------
    start_time : Starting time
    end_time : Ending time

    Useage
    ------------
    helper_functions.TIMERS t1;

    t1.TIC()
    .
    ... Your CODE HERE....
    .
    t1.TOC()

    '''

    def __init__(self):
        self.start_time = None
        self.end_time = None
        pass


    def TIC(self):
        '''
        Starts the timer

        Parameters
        ------------
        None

        Returns
        ------------
        None
        '''
        self.start_time = time.time()
        self.end_time = None


    def TOC(self, show_time = True):
        '''
        Stops the timer and optionally prints the time between TIC and TOC

        Parameters
        ------------
        show_time : bool
            If true, prints the time. Else you need to explicitly call self.PRINT_TIME()

        Returns
        ------------
        None
        '''
        if (self.start_time is None):
            print("\n--- Timer not started. Use self.TIC() to start the timer ---\n")
            # break
        else:
            self.end_time = time.time()
            if (show_time):
                if (self.end_time-self.start_time < 0.001): # if time is less than 0.001 Sec, then show time in milllsec
                    print("Time : " + str((self.end_time-self.start_time)*1000) + "  Milli-seconds")
                else:
                    print("Time : " + str(self.end_time-self.start_time) + "  seconds")



    def PRINT_TIME(self):
        '''
        Prints the time calculated between TIC and TOC

        Parameters
        ------------
        None

        Returns
        ------------
        None
        '''
        if (self.start_time is None):
            print("Timer not started. Use self.TIC() to start the timer")
        elif(self.end_time is None):
            self.end_time = time.time()
        else:
            if (self.end_time-self.start_time < 0.001):
                print("Time : " + str((self.end_time-self.start_time)*1000) + "  Milli-seconds")
            else:
                print("Time : " + str(self.end_time-self.start_time) + "  seconds")

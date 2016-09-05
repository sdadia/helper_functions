import cv2
import numpy as np
import matplotlib.pyplot as plt


def PLOT_IMAGE(image, figure_num=1, show=False):
    '''
    Parameters:
        image = any image
        figure_num *= figure_num number

    Returns:
        None
    '''
    plt.figure(figure_num)
    plt.imshow(image)
    plt.axis("off") # Don't show coordinate axis
    if(show is True): # Show image if asked
        plt.show()
    return None


def GINPUT_ROUTINE(image, num_pts=2):
    '''
    Get coordinates of points in image

    Parameters:
    image : Numpy array
        Image where points are to be selected

    Returns:
    coordinates : Numpy array
        Numpy array containing the coordinates of the points (row,col) format
    '''
    PLOT_IMAGE(image, show=False)
    print("Please select" + str(num_pts) + "points")    # timeout : time(sec) to wait until termination, if input not given
    coordinates = np.array(plt.ginput(n=num_pts, timeout=10))
    coordinates[:, 0], coordinates[:, 1] = coordinates[:, 1], coordinates[:, 0].copy() # Exchange col1 and col2 to get (row_coordinate, col_coordinate)

    return coordinates.astype(np.uint8) # Return +ve integers only

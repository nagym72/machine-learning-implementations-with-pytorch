import numpy
import numpy as np
import sys

#numpy.set_printoptions(threshold=sys.maxsize)

def ex4(image_array:np.array, offset:tuple, spacing:tuple) -> np.array:
    """
    image_array will take an array of shape (M, N, 3), np.dtype = float64 were M = rows, N = cols
    and 3 will be RGB color channel.
    offset is a tuple containing 2 int values that specify the movement of the grid in x and y respectively
    spacing is a tuple containing 2 int values that specify the spacing between 2 points in x and y respectively
    return : a tuple consisting of (input_array, known_array, target_array)
    shape of return: input_array (3, M, N) 3dim
                     known_array (3, M, N) 3dim
                     target_array (M,) 1dim (len(removed_values) over all 3 channels)
    """

    #check if the provided image_array is suitable:
    if not type(image_array) is np.ndarray:
        raise TypeError(f"Input array is not of type np.ndarray. Instead it is of type: {type(image_array)}")

    #check if 3D array
    if not len(image_array.shape) == 3:
        raise NotImplementedError(f"Expected array shape: (M, N, 3) \nObtained input shape: {image_array.shape}")

    #check if 3rd dimension is of size 3 (RGB) colorchannel
    if not image_array.shape[2] == 3:
        raise NotImplementedError(f"Expected dimension for RGB channel: 3\nObtained input: {image_array.shape[2]}")

    #check offset dtype convertability
    try:
        n, m  = int(offset[0]), int(offset[1])
    except ValueError:
        raise ValueError(f"Cant convert input{offset} to integer\nPlease provide an integer input")

    #check offset
    if offset[0] < 0 or offset[1] < 0 or offset[0] >32 or offset[1] > 32:
        raise ValueError("Offset too small or too large\nAllowed range: [0 , 32]")

    #check spacing
    if spacing[0] < 2 or spacing[1] < 2 or spacing[0] > 8 or spacing[1] > 8:
        raise ValueError("Spacings too small or too large\nAllowed range: [2 , 8]")

    x_off, y_off = offset
    space_x, space_y = spacing
    #transposing the input array from shape (M, N , 3) to (3, M, N)

    image_array = np.transpose(image_array, (2,0,1))
    known_array = image_array.copy()      #storing as copy in order to not mess with original input array

    #mfilling the array with 0s
    known_array = np.zeros_like(known_array)

    #first dim = RGB (0:3), second dim = N and third dim = M
    # generates a grid where all values that are not filtered out are set to 1 , rest stays 0
    #y_off::space_y
    known_array[0:3,y_off::space_y, x_off::space_x] = 1

    #storing the RGB original values which are set to 0 after filtering:
    #generates a 1d flattened array where first the R then G then B values are stored.

    target_array = image_array[known_array < 1]     #shape = (3,6,6)

    check_array = known_array.flatten()  #generates 1D array consisting of all RGB values total : M*N*3

    #check if after filtering we still have more than 144 pixels.   (total pixels - filtered pixels)
    if len(check_array) - len(target_array) < 144:
        return ValueError("Not enough pixels left after filtering")

    input_array = image_array*known_array      #shape = (3,6,6) . Multiplying all values with  0 will result in 0 and the ones multiplied by 1 will stay the original values.

    #returns tuple of shape (3,6,6) , (3,6,6), (x,) where x is the amount of filtered pixels
    return (input_array, known_array, target_array)


testarray = np.random.rand(20,20,3)
spacing = (2,2)
offset = (0,0)

print(ex4(testarray, offset, spacing))


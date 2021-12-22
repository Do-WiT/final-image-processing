import matplotlib.pyplot as plt
import numpy as np
from skimage import transform
from skimage.color import rgb2gray

# Define prototypes
prototypes = np.zeros((10, 9, 7))
prototypes[0] = [[0, 1, 1, 1, 1, 1, 0],
           [1, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 1],
           [1, 0, 0, 0, 0, 0, 1],
           [0, 1, 1, 1, 1, 1, 0]]
prototypes[1] = [[0, 0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 1],
           [0, 0, 0, 1, 1, 1, 1]]
prototypes[2] = [[0, 0, 0, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 1, 1, 1, 1],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1]]
prototypes[3] = [[0, 0, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 1],
           [0, 0, 1, 1, 1, 1, 1]]
prototypes[4] = [[0, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 1, 1],
           [0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 1]]
prototypes[5] = [[0, 0, 1, 1, 1, 1, 1],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 1, 1, 1, 1, 1], ]
prototypes[6] = [[0, 1, 1, 1, 1, 0, 0],
           [0, 1, 0, 0, 1, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0],
           [0, 1, 1, 1, 1, 1, 1],
           [0, 1, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 1],
           [0, 1, 1, 1, 1, 1, 1]]
prototypes[7] = [[0, 0, 1, 1, 1, 1, 1],
           [0, 0, 1, 0, 0, 0, 1],
           [0, 0, 1, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 0, 0]]
prototypes[8] = [[0, 1, 1, 1, 1, 1, 0],
           [0, 1, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 1, 0],
           [0, 1, 0, 0, 0, 1, 0],
           [0, 1, 1, 1, 1, 1, 0],
           [1, 1, 0, 0, 0, 1, 1],
           [1, 1, 0, 0, 0, 1, 1],
           [1, 1, 0, 0, 0, 1, 1],
           [1, 1, 1, 1, 1, 1, 1]]
prototypes[9] = [[0, 1, 1, 1, 1, 1, 1],
           [0, 1, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 1],
           [0, 1, 0, 0, 0, 0, 1],
           [0, 1, 1, 1, 1, 1, 1],
           [0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 0, 1, 1]]


def alignment(img, src, dst):
    tform = transform.ProjectiveTransform()
    tform.estimate(src, dst)
    warped = transform.warp(img, tform, output_shape=(100, 900))
    return warped


def rgb_to_binary(img, binary_threshold):
    grayscale = rgb2gray(img) * 255
    grayscale = grayscale.astype(int)
    binary = grayscale < binary_threshold
    return binary


def prototype_matching(scale, prot):

    prob_of_prot = []

    for w in range(scale.shape[1]):

        if w + 7 >= scale.shape[1]:
            break

        for p in range(prot.shape[0]):

            prototype = prot[p]

            sub_img = scale[0:9, w:w + 7]

            score = 0

            for row in range(prototype.shape[0]):
                for col in range(prototype.shape[1]):

                    if prototype[row, col] == sub_img[row, col]:
                        score += 1

            prob = score / (prototype.shape[0] * prototype.shape[1])

            #if prob > 0.95:
            prob_of_prot.append([prob, p])

    return prob_of_prot


if __name__ == '__main__':

    img = plt.imread("img/endcode.bmp")


    # crop src image to dst image
    src = np.array([[0, 0], [0, 100], [900, 100], [900, 0]])
    dst = np.array([[218, 339], [249, 385], [600, 174], [571, 143]])
    crop = alignment(img, src, dst)

    #
    binary_threshold = 200
    binary = rgb_to_binary(crop, binary_threshold)

    #
    new_shape = (9, 82)
    scale = transform.resize(binary, new_shape)


    # prototype matching
    prots = prototype_matching(scale, prototypes)


    # print number if prototype matching more than 0.95 percent
    probability = 0.95

    print("Decode from image is : ", end=" ")
    for prot in prots:
        if prot[0] > probability:
            print(prot[1], end=" ")










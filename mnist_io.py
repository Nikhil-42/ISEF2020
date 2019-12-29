import io
import struct
import numpy as np

#temp_lim = 10

def images_from_file(filepath: str) -> np.array:

    file = open(filepath, 'rb')
    
    magic_number = struct.unpack(">i", file.read(4))[0]
    image_count = struct.unpack(">i", file.read(4))[0]

    print("Preparing to load ", image_count, " images from: ", filepath)

    # For debugging speed, TODO Delete later
    #image_count = temp_lim


    image_height = struct.unpack(">i", file.read(4))[0]
    image_width = struct.unpack(">i", file.read(4))[0]

    images = np.zeros((image_count, image_height, image_width), dtype=int)

    for i in range(image_count):
        get_next_image(file, image_height, image_width, images[i])

    print("Loaded ", len(images), " images")
    return images

def get_next_image(file, image_width: int, image_height: int, image: list) -> list:
    for y in range(image_height):
        for x in range(image_width):
            image[y, x] = struct.unpack(">B", file.read(1))[0]

def labels_from_file(filepath: str):
    file = open(filepath, 'rb')

    magic_number = struct.unpack(">i", file.read(4))[0]
    label_count = struct.unpack(">i", file.read(4))[0]


    # For debugging speed, TODO Delete later
    #label_count = temp_lim


    labels = np.zeros((label_count, 10), dtype=int)

    for i in range (0, label_count-1):
        value = struct.unpack(">B", file.read(1))[0]
        labels[i, value] = 1

    return labels
import io
import struct
import numpy as np

def images_from_file(filepath: str, image_count=None) -> np.array:
    with open(filepath, 'rb') as file:
        magic_number = struct.unpack(">i", file.read(4))[0]
        if image_count == None:
            image_count = struct.unpack(">i", file.read(4))[0]
        else:
            file.read(4)

        image_height = struct.unpack(">i", file.read(4))[0]
        image_width = struct.unpack(">i", file.read(4))[0]

        print("Preparing to load ", image_count, " images from: ", filepath)

        # For debugging speed, TODO Delete later
        #image_count = temp_lim

        PIXEL_SIZE = 1
        images_bytes = bytearray(PIXEL_SIZE*image_height*image_width*image_count)
        file.readinto(images_bytes)

        images = np.frombuffer(images_bytes, dtype=np.uint8)
        images = images.reshape((image_count, image_height, image_width))

        print("Loaded ", len(images), " images")
        return images
    return None

def labels_from_file(filepath: str, label_count=0):
    with open(filepath, 'rb') as file:
        magic_number = struct.unpack(">i", file.read(4))[0]
        if label_count == 0:
            label_count = struct.unpack(">i", file.read(4))[0]
        else:
            file.read(4)

        LABEL_SIZE = 1
        labels_bytes = bytearray(label_count*LABEL_SIZE)
        file.readinto(labels_bytes)

        labels = np.frombuffer(labels_bytes, dtype=np.uint8)
        return labels
    return None

if __name__ == '__main__':
    new = images_from_file('datasets\\train-images-idx3-ubyte\\train-images.idx3-ubyte')
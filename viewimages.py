import tkinter as tk
import io
import struct

class ViewImages:

    def __init__(self, images_filename: str, images=[], image_height=0, image_width=0, image_count=0):
        self.images_file = open(images_filename, 'rb')
            
        self.magic_number = struct.unpack(">i", self.images_file.read(4))[0]
        self.image_count = struct.unpack(">i", self.images_file.read(4))[0]
        self.image_height = struct.unpack(">i", self.images_file.read(4))[0]
        self.image_width = struct.unpack(">i", self.images_file.read(4))[0]

        self.current_image = 0
        self.images = []

        top = tk.Tk()
        top.geometry('400x400')

        self.canvas = tk.Canvas(top, width=self.image_width*10, height=self.image_height*10)
        self.canvas.pack(side='top', expand='no')

        previous_button = tk.Button(top, text='Previous', command=self.previous_button_command)
        previous_button.pack(side='bottom', fill='x', expand='yes')

        next_button = tk.Button(top, text='Next', command=self.next_button_command)
        next_button.pack(side='bottom', fill='x', expand='yes')

        self.images.append(self.get_data())
        print(self.images)
        self.draw_data()

        top.mainloop()

    def get_data(self):
        values = []
        for num in range(0, self.image_width*self.image_height):
            values.append(struct.unpack(">B", self.images_file.read(1))[0])
        return values

    def draw_data(self):
        for i, value in enumerate(self.images[self.current_image]):
            x = i % self.image_width
            y = (int) (i / self.image_height)
            self.canvas.create_rectangle(x*10, y*10, (x+1)*10, (y+1)*10, fill="#%02x%02x%02x" % (value, value, value))

    def next_button_command(self):
        self.current_image = self.current_image + 1
        if self.current_image == len(self.images):
            if not len(self.images) == self.image_count:
                self.images.append(self.get_data())
            else:
                self.current_image = len(self.images) -1
        self.draw_data()

    def previous_button_command(self):
        self.current_image = self.current_image - 1
        if self.current_image < 0:
            self.current_image = 0
        self.draw_data()
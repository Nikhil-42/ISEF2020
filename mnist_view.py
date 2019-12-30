import tkinter as tk
import io
import struct

class ViewData:

    def __init__(self, images, labels):

        self.images = images
        self.labels = labels

        self.image_count = images.shape[0]
        self.image_height = images.shape[1]
        self.image_width = images.shape[2]

        self.current = 0

        top = tk.Tk()
        top.geometry('400x400')
        top.title("Visualizer")

        self.canvas = tk.Canvas(top, width=self.image_width*10, height=self.image_height*10)
        self.canvas.pack(side='top', expand='no')

        self.label = tk.Label(top, text="Error")
        self.label.pack(side='top')

        previous_button = tk.Button(top, text='Previous', command=self.previous_button_command)
        previous_button.pack(side='bottom', fill='x', expand='yes')

        next_button = tk.Button(top, text='Next', command=self.next_button_command)
        next_button.pack(side='bottom', fill='x', expand='yes')

        self.draw_data()

        top.mainloop()

    def draw_data(self):
        self.label.configure(text="%s" % self.labels[self.current])
        for y, row in enumerate(self.images[self.current]):
            for x, value in enumerate(row):
                value = int(round(value*255))
                self.canvas.create_rectangle(x*10, y*10, (x+1)*10, (y+1)*10, fill="#%02x%02x%02x" % (value, value, value))

    def next_button_command(self):
        self.current = self.current + 1
        if self.current >= len(self.images):
            self.current = len(self.images) -1
        self.draw_data()

    def previous_button_command(self):
        self.current = self.current - 1
        if self.current < 0:
            self.current = 0
        self.draw_data()
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

from keras.preprocessing import image

#LEFT = 0
#UPPER = 0
#RIGHT = 112
#LOWER = 112


def cover_img():
    Path = os.getcwd()
    data_path = Path + '/Data_case_control_prior1'
    data_dir_list = os.listdir(data_path)
    for dataset in data_dir_list:
        img_list = os.listdir(data_path +'/' +dataset)
        print('loaded the images from dataset-'+'{}\n'.format(dataset))
        for img in img_list:
            imageName = img
            img_path = data_path + '/' + dataset + '/' + imageName
            img = image.load_img(img_path)
            #image.save('croppedImage.png')
            #image = Image.open("croppedImage.png")
            #img.show()
            width, height = img.size
            box_range_width = range (LEFT,RIGHT)
            box_range_height = range(UPPER,LOWER)
    
            for x in range (0,width):
                for y in range (0, height):
                    if(not(x in box_range_width and y in box_range_height)):
                    
                        img.putpixel((x,y),(0,0,0))
            imageName = imageName.split(".")
            img.save("./My_Dataset/croppedImages/"+imageName[0]+".png")

def change_data(my_img_data,LEFT,RIGHT,UPPER,LOWER):
    new_list = []
    n,w,h,d = my_img_data.shape

    for i in range( 0,n):
        img = my_img_data[i]
        width, height,p = img.shape
        box_range_width = range (LEFT,RIGHT)
        box_range_height = range(UPPER,LOWER)
        new_img = np.zeros(shape=(width,height,p))
        for x in range (0,width):
            for y in range (0, height):
                if(not(x in box_range_width and y in box_range_height)):
                    new_img[x,y,0] = 0
                    new_img[x,y,1] = 0
                    new_img[x,y,2] = 0
                else:
                    new_img[x,y,0] = img[x,y,0]
                    new_img[x,y,1] = img[x,y,1]
                    new_img[x,y,2] = img[x,y,2]
#        imageToShow = Image.fromarray(img, 'RGB')
#        imageToShow.show()
#        if(i == 1):
#            plt.figure()
#            plt.imshow(new_img)
#            plt.figure()
#            plt.imshow(img)
        new_list.append(new_img)
    return new_list
#
#import sys
#from PyQt4 import QtGui
#
#def window():
#   app = QtGui.QApplication(sys.argv)
#   w = QtGui.QWidget()
#   b = QtGui.QLabel(w)
#   b.setText("Hello World!")
#   w.setGeometry(100,100,200,50)
#   b.move(50,20)
#   w.setWindowTitle("PyQt")
#   w.show()
#   sys.exit(app.exec_())
#	
#if __name__ == '__main__':
#   window()
   
   
## Simple enough, just import everything from tkinter.
#from tkinter import *
#
#
##download and install pillow:
## http://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow
#from PIL import Image, ImageTk
#
#
## Here, we are creating our class, Window, and inheriting from the Frame
## class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
#class Window(Frame):
#
#    # Define settings upon initialization. Here you can specify
#    def __init__(self, master=None):
#        
#        # parameters that you want to send through the Frame class. 
#        Frame.__init__(self, master)   
#
#        #reference to the master widget, which is the tk window                 
#        self.master = master
#
#        #with that, we want to then run init_window, which doesn't yet exist
#        self.init_window()
#
#    #Creation of init_window
#    def init_window(self):
#
#        # changing the title of our master widget      
#        self.master.title("GUI")
#
#        # allowing the widget to take the full space of the root window
#        self.pack(fill=BOTH, expand=1)
#
#        # creating a menu instance
#        menu = Menu(self.master)
#        self.master.config(menu=menu)
#
#        # create the file object)
#        file = Menu(menu)
#
#        # adds a command to the menu option, calling it exit, and the
#        # command it runs on event is client_exit
#        file.add_command(label="Exit", command=self.client_exit)
#
#        #added "file" to our menu
#        menu.add_cascade(label="File", menu=file)
#
#
#        # create the file object)
#        edit = Menu(menu)
#
#        # adds a command to the menu option, calling it exit, and the
#        # command it runs on event is client_exit
#        edit.add_command(label="Show Img", command=self.showImg)
#        edit.add_command(label="Show Text", command=self.showText)
#
#        #added "file" to our menu
#        menu.add_cascade(label="Edit", menu=edit)
#                # creating a button instance
#        quitButton = Button(self, text="Exit",command=self.client_exit)
#
#        # placing the button on my window
#        quitButton.place(x=0, y=0)
#
#       
#
#    def client_exit(self):
#        exit()
#
#    def showImg(self):
#        load = Image.open("airplane.jpeg")
#        render = ImageTk.PhotoImage(load)
#
#        # labels can be text or images
#        img = Label(self, image=render)
#        img.image = render
#        img.place(x=0, y=0)
#
#
#    def showText(self):
#        text = Label(self, text="Hey there good lookin!")
#        text.pack()
#        
#
#    def client_exit(self):
#        exit()
#
#
## root window created. Here, that would be the only window, but
## you can later have windows within windows.
#root = Tk()
#
#root.geometry("400x300")
#
##creation of an instance
#app = Window(root)
#
#
##mainloop 
#root.mainloop() 

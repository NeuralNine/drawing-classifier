'''
NEURALNINE (c) 2019
Drawing Classifier ML Alpha v0.1

This is the very first prototype and the code is not clean at all
Also there may be a couple of bugs
A lot of exceptions are not handled
'''

'''
IMPORTS
'''
import PIL
import pickle
import os.path
import cv2 as cv
import numpy as np
import tkinter.messagebox

from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from PIL import Image, ImageDraw
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

'''
Project Initialization
If a project is already existing, it loads the saved file
Otherwise it creates a new one
Creates a directory structure with project name and the individual class names
'''
msg = Tk()
msg.withdraw()

proj_name = simpledialog.askstring("Project Name", "Please enter your project name down below!", parent=msg)
if os.path.exists(proj_name):
    with open('{}/{}_data.pickle'.format(proj_name, proj_name), 'rb') as f:
        data = pickle.load(f)
    class1 = data['c1']
    class2 = data['c2']
    class3 = data['c3']
    class1_counter = data['c1c']
    class2_counter = data['c2c']
    class3_counter = data['c3c']
    clf = data['clf']
    proj_name = data['pname']
else:
    class1 = simpledialog.askstring("Project Name", "What is the first class called?", parent=msg)
    class2 = simpledialog.askstring("Project Name", "What is the second class called?", parent=msg)
    class3 = simpledialog.askstring("Project Name", "What is the third class called?", parent=msg)

    class1_counter = 1
    class2_counter = 1
    class3_counter = 1

    clf = LinearSVC()

    os.mkdir(proj_name)
    os.chdir(proj_name)
    os.mkdir(class1)
    os.mkdir(class2)
    os.mkdir(class3)
    os.chdir('..')

'''
FUCTIONS FOR THE MODEL AND THE MACHINE LEARNING
'''

'''
Function for training the model
Reads in all the different examples and fits the classifier
'''
def train_model():
    global clf

    img_list = np.array([])
    class_list = np.array([])

    for x in range(1, class1_counter):
        img = cv.imread('{}/{}/{}.png'.format(proj_name, class1, x))[:, :, 0]
        img = img.reshape(2500)
        img_list = np.append(img_list, [img])
        class_list = np.append(class_list, 1)

    for x in range(1, class2_counter):
        img = cv.imread('{}/{}/{}.png'.format(proj_name, class2, x))[:, :, 0]
        img = img.reshape(2500)
        img_list = np.append(img_list, [img])
        class_list = np.append(class_list, 2)

    for x in range(1, class3_counter):
        img = cv.imread('{}/{}/{}.png'.format(proj_name, class3, x))[:, :, 0]
        img = img.reshape(2500)
        img_list = np.append(img_list, [img])
        class_list = np.append(class_list, 3)

    img_list = img_list.reshape(class1_counter-1 + class2_counter-1 + class3_counter-1, 2500)

    clf.fit(img_list, class_list)
    print("Model successfully trained!")
    tkinter.messagebox.showinfo("NeuralNine", "Model successfully trained!", parent=root)

'''
Function that predicts the class of the current drawing
First it scales the image from the canvas down to 50x50
Then it predicts it using the classifier
'''
def predict():
    global clf

    filename = "temp.png"
    image1.save(filename)
    im = PIL.Image.open(filename)
    im.thumbnail((50, 50), PIL.Image.ANTIALIAS)
    im.save('predictshape.png', 'PNG')

    img = cv.imread('predictshape.png')[:, :, 0]
    img = img.reshape(2500)
    prediction = clf.predict([img])
    if prediction[0] == 1:
        print("The drawing is probably a {}!".format(class1))
        tkinter.messagebox.showinfo("NeuralNine", "The drawing is probably a {}!".format(class1), parent=root)
    elif prediction[0] == 2:
        print("The drawing is probably a {}!".format(class2))
        tkinter.messagebox.showinfo("NeuralNine", "The drawing is probably a {}!".format(class2), parent=root)
    elif prediction[0] == 3:
        print("The drawing is probably a {}!".format(class3))
        tkinter.messagebox.showinfo("NeuralNine", "The drawing is probably a {}!".format(class3), parent=root)

'''
Function that serializes the model into a file
'''
def save_model():
    global clf
    file_path = filedialog.asksaveasfilename(defaultextension="pickle")
    with open(file_path, 'wb') as f:
        pickle.dump(clf, f)
    tkinter.messagebox.showinfo("NeuralNine", "Model successfully saved!", parent=root)

'''
Function that loads and deserializes a model from a file
'''
def load_model():
    global clf
    file_path = filedialog.askopenfilename()
    with open(file_path, 'rb') as f:
        clf = pickle.load(f)
    tkinter.messagebox.showinfo("NeuralNine", "Model successfully loaded!", parent=root)

'''
Function that saves all the important and relevant objects into a dictionary
Saves all the class names, the counters, the project name and the actual model or classifier
Serializes all into a file that gets loaded if you use the same project_name
'''
def save_everything():
    data = {'c1': class1, 'c2': class2, 'c3': class3, 'c1c': class1_counter, 'c2c': class2_counter, 'c3c': class3_counter,
            'pname': proj_name, 'clf': clf}
    with open('{}/{}_data.pickle'.format(proj_name, proj_name), 'wb') as f:
        pickle.dump(data, f)
    tkinter.messagebox.showinfo("NeuralNine", "Project Saved!", parent=root)

'''
Function that changes the used algorithm
The user may choose between various different classifiers
'''
def change_model():
    global clf
    if type(clf) == type(LinearSVC()):
        clf = KNeighborsClassifier()
        print("Now using K-Nearest-Neighbors!")
    elif type(clf) == type(KNeighborsClassifier()):
        clf = LogisticRegression()
        print("Now using Logistic Regression!")
    elif type(clf) == type(LogisticRegression()):
        clf = DecisionTreeClassifier()
        print("Now using Decision Tree Classifier!")
    elif type(clf) == type(DecisionTreeClassifier()):
        clf = RandomForestClassifier()
        print("Now using Random Forest Classifier!")
    elif type(clf) == type(RandomForestClassifier()):
        clf = GaussianNB()
        print("Now using Gaussian Naive Bayes!")
    elif type(clf) == type(GaussianNB()):
        clf = LinearSVC()
        print("Now using Linear SVC!")

    status_label.config(text="Current Model: {}".format(type(clf).__name__))

'''
PAINTING PART
'''

'''
CONSTANTS FOR TKINTER
'''
width = 500
height = 500
center = height // 2
white = (255,255,255)
bwidth = 15

'''
Clears the whole canvas and overwrites the image
'''
def clear():
    cnv.delete("all")
    draw.rectangle([0,0,1000,1000], fill="white")

'''
Increases brush size
'''
def brushplus():
    global bwidth
    bwidth += 1
    print("Brush Size is ", bwidth)

'''
Decreases brush size
'''
def brushminus():
    global bwidth
    if(bwidth > 1):
        bwidth -= 1
    print("Brush Size is ",  bwidth)

'''
Saves the current drawing into the directory of the respective class
Also scales it down to 50x50
'''
def save(class_num):
    global class1_counter, class2_counter, class3_counter
    filename = "temp.png"
    image1.save(filename)
    im = PIL.Image.open(filename)
    im.thumbnail((50,50), PIL.Image.ANTIALIAS)

    if class_num == 1:
        im.save("{}/{}/{}.png".format(proj_name, class1, class1_counter), "PNG")
        class1_counter += 1
    elif class_num == 2:
        im.save("{}/{}/{}.png".format(proj_name, class2, class2_counter), "PNG")
        class2_counter += 1
    elif class_num == 3:
        im.save("{}/{}/{}.png".format(proj_name, class3, class3_counter), "PNG")
        class3_counter += 1

    clear()

'''
Function that handles the mouse movement and the drawing
'''
def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cnv.create_rectangle(x1, y1, x2, y2, fill="black", width=bwidth)
    draw.rectangle([x1,y1, x2+bwidth, y2+bwidth], fill="black", width=bwidth)

'''
Function that gets called when window is getting closed
Offers possibility to save the current state of everything
'''
def on_closing():
        answer = tkinter.messagebox.askyesnocancel("Quit", "Do you want to save your work?", parent=root)
        print(answer)
        if answer is not None:
            if answer:
                save_everything()
            root.destroy()
            exit()

'''
GUI INITIALIZATION
'''

# Defining root and setting title
root = Tk()
root.title("NeuralNine Drawing Classifier Alpha v0.1 - {}".format(proj_name))

# Defining Canvas
cnv = Canvas(root, width=width - 10, height=height - 10, bg="white")
cnv.pack()

# Defining the ImageDrawer for Pillow
image1 = PIL.Image.new("RGB", (width, height), white)
draw = ImageDraw.Draw(image1)

# Binding the Mouse Movement to the paint function
cnv.pack(expand=YES, fill=BOTH)
cnv.bind("<B1-Motion>", paint)

# Defining frame for our buttons and for the label
btn_frame = tkinter.Frame(root)
btn_frame.pack(fill=tkinter.X, side=tkinter.BOTTOM)

btn_frame.columnconfigure(0, weight=1)
btn_frame.columnconfigure(1, weight=1)
btn_frame.columnconfigure(2, weight=1)

'''
All the buttons with their respective functions
'''

class1_btn = Button(btn_frame, text=class1, command=lambda: save(1))
class1_btn.grid(row=0, column=0, sticky=tkinter.W+tkinter.E)

class2_btn = Button(btn_frame, text=class2, command=lambda: save(2))
class2_btn.grid(row=0, column=1, sticky=tkinter.W+tkinter.E)

class3_btn = Button(btn_frame, text=class3, command=lambda: save(3))
class3_btn.grid(row=0, column=2, sticky=tkinter.W+tkinter.E)

bm_btn = Button(btn_frame, text="Brush-", command=brushminus)
bm_btn.grid(row=1, column=0, sticky=tkinter.W+tkinter.E)

clear_btn = Button(btn_frame, text="Clear", command=clear)
clear_btn.grid(row=1, column=1, sticky=tkinter.W+tkinter.E)

bp_btn = Button(btn_frame, text="Brush+", command=brushplus)
bp_btn.grid(row=1, column=2, sticky=tkinter.W+tkinter.E)

train_btn = Button(btn_frame, text="Train Model", command=train_model)
train_btn.grid(row=2, column=0, sticky=tkinter.W+tkinter.E)

save_btn = Button(btn_frame, text="Save Model", command=save_model)
save_btn.grid(row=2, column=1, sticky=tkinter.W+tkinter.E)

load_btn = Button(btn_frame, text="Load Model", command=load_model)
load_btn.grid(row=2, column=2, sticky=tkinter.W+tkinter.E)

change_btn = Button(btn_frame, text="Change Model", command=change_model)
change_btn.grid(row=3, column=0, sticky=tkinter.W+tkinter.E)

predict_btn = Button(btn_frame, text="Predict", command=predict)
predict_btn.grid(row=3, column=1, sticky=tkinter.W+tkinter.E)

save_everything_btn = Button(btn_frame, text="Save Everything", command=save_everything)
save_everything_btn.grid(row=3, column=2, sticky=tkinter.W+tkinter.E)

status_label = Label(btn_frame, text="Current Model: {}".format(type(clf).__name__))
status_label.config(font=("Arial", 10))
status_label.grid(row=4, column=1, sticky=tkinter.W+tkinter.E)

# Binding on_closing() to closing event
# And getting screen in to the front
root.protocol("WM_DELETE_WINDOW", on_closing)
root.attributes("-topmost", True)
root.mainloop()
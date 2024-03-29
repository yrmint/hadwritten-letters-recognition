import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Define the model you want
NOISE = 0
TRANSFORM = 1
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']

name = 'work-models/model'
if NOISE:
    name += '-noised'
if TRANSFORM:
    name += '-transform'
name += '.h5'

# Load model
model = load_model(name)


# Draw using left mouse button
def draw(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='black')
    img_draw.line([x1, y1, x2, y2], fill='white', width=20)


# Erase using right mouse button
def erase(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill='white', outline='white')
    img_draw.line([x1, y1, x2, y2], fill='black', width=20)


# Predict the drawn number
def make_prediction():
    image = image1.resize((28, 28)).convert('L')
    data = np.array(image)
    data = data.reshape(1, 28, 28, 1)
    data = data.astype('float32') / 255
    prediction = model.predict(data)
    prediction = prediction[0]
    plt.bar(ALPHABET, prediction)
    plt.xticks(ALPHABET)
    plt.show()


# Erase everything on the canvas
def erase_all():
    canvas.create_rectangle(0, 0, 1000, 1000, fill='white', outline='white')
    img_draw.line([0, 0, 1000, 1000], fill='black', width=1000)


# Create the main window
root = tk.Tk()
root.title("Handwritten Letter Recognition")

# Create a canvas for drawing
canvas = Canvas(root, width=280, height=280, bg='white')
canvas.pack()

# Create blanc image
image1 = Image.new('RGB', (280, 280), 'black')
img_draw = ImageDraw.Draw(image1)

canvas.bind("<B1-Motion>", draw)  # Bind left mouse button motion
canvas.bind("<B3-Motion>", erase)  # Bind right mouse button motion

# Create a button to predict the drawn letter
predict_button = tk.Button(root, text="Predict", command=make_prediction)
predict_button.pack()

# Create a button to erase everything
erase_button = tk.Button(root, text="Erase all", command=erase_all)
erase_button.pack()

root.mainloop()

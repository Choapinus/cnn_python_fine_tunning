
# coding: utf-8

# In[1]:

from __future__ import print_function
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

# In[19]:

EPOCHS = 200
train_dir = './dataset/train'
validation_dir = './dataset/validation'

im_size = 224
INIT_LR = 1e-3

vgg_conv = VGG16(
	weights='imagenet',
	include_top=False,
	input_shape=(im_size, im_size, 3)
)


# In[3]:

model = Sequential()

for layer in vgg_conv.layers[:-1]: # without the last dense layer (that is the predict layer)
	model.add(layer)

# add your dense layer with the mount of classes

model.add(Dense(250))
model.add(Dense(2, activation='softmax', name='predictions'))

# In[12]:


for layer in model.layers[:-8]: # last 8 layers will be trainable
	layer.trainable = False

for layer in model.layers:
	print(layer, layer.trainable)

print('total layers:', model.layers.__len__())




# In[15]:


# Data augmentation

train_datagen = ImageDataGenerator(
	rescale=1./255,
	rotation_range=20,
	width_shift_range=0.2,
	height_shift_range=0.2,
	horizontal_flip=True,
	fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Change the batchsize according to your system RAM
train_batchsize = 50
val_batchsize = 50

# Data Generator for Training data
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(im_size, im_size),
	batch_size=train_batchsize,
	class_mode='categorical'
)

# Data Generator for Validation data
validation_generator = validation_datagen.flow_from_directory(
	validation_dir,
	target_size=(im_size, im_size),
	batch_size=val_batchsize,
	class_mode='categorical',
	shuffle=False
)


# In[20]:


# Compile the model
model.compile(
	loss='categorical_crossentropy',
	optimizer=SGD(lr=INIT_LR),
	metrics=['acc']
)

# Train the Model
# NOTE that we have multiplied the steps_per_epoch by 2. This is because we are using data augmentation.
H = model.fit_generator(
	train_generator,
	steps_per_epoch=2*train_generator.samples/train_generator.batch_size ,
	epochs=EPOCHS,
	validation_data=validation_generator,
	validation_steps=validation_generator.samples/validation_generator.batch_size,
	verbose=1
)

# Save the Model
model.save('final_last8_layers.h5')

print("[INFO] ploting...")

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Male/Female")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot")

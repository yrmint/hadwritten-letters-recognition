from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.src.saving.saving_api import save_model

from preprocess import preprocess_data
from plot import plot_history, plot_stats


# Define parameters
NOISE = 0
TRANSFORM = 0
NEURONS_IN_LAYER = 10000
ACTIVATION_FUNC = "relu"
ACTIVATION_FUNC_OUT = "softmax"
EPOCHS = 5


# Create the neural network model
def build_model(neurons_in_layer, activation_func, activation_func_out):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation=activation_func, input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation=activation_func))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation=activation_func))
    model.add(Flatten())
    model.add(Dense(neurons_in_layer, activation=activation_func))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation=activation_func_out))
    return model


def save_summary(acc, loss):
    f = open('stats/acc' + str(NEURONS_IN_LAYER) + '.txt', 'w')
    fh = open('stats/loss' + str(NEURONS_IN_LAYER) + '.txt', 'w')
    f.write(str(acc))
    fh.write(str(loss))
    f.close()
    fh.close()


# Preprocess data
x_train, y_train, x_test, y_test = preprocess_data('data/A_Z Handwritten Data.csv', NOISE, TRANSFORM)

model = build_model(NEURONS_IN_LAYER, ACTIVATION_FUNC, ACTIVATION_FUNC_OUT)
print(model.summary())

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=128, validation_data=(x_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

# Save the model
name = 'work-models/model'
if NOISE:
    name += '-noise'
if TRANSFORM:
    name += '-transform'
name += '.h5'
save_model(model, name)

# Plot graphs
plot_history(history, EPOCHS)
plot_stats()
save_summary(test_acc, test_loss)


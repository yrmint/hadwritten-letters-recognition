import numpy as np
import matplotlib.pyplot as plt


# Plot the history
def plot_history(model, epochs):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), model.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), model.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), model.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), model.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()


def plot_stats():
    print("accuracy for:")
    # Plot test accuracy
    stats = []
    f = open('stats/acc' + str(1) + '.txt', 'r')
    acc = float(f.read())
    stats.append(acc)
    print(str(1) + " neurons: " + str(acc))
    f.close()
    for i in range(1, 5):
        f = open('stats/acc' + str(16 * 2 ** (i-1)) + '.txt', 'r')
        acc = float(f.read())
        stats.append(acc)
        print(str(16 * 2 ** (i-1)) + " neurons: " + str(acc))
        f.close()
    plt.title("Accuracy")
    plt.plot([1, 16, 32, 64, 128], stats)
    plt.xlabel("Neurons")
    plt.ylabel("Test accuracy")
    plt.show()
    print("\n")

    print("loss for:")
    # Plot test loss
    stats = []
    f = open('stats/acc' + str(1) + '.txt', 'r')
    acc = float(f.read())
    stats.append(acc)
    print(str(1) + " neurons: " + str(acc))
    for i in range(1, 5):
        f = open('stats/loss' + str(16 * 2 ** (i-1)) + '.txt', 'r')
        loss = float(f.read())
        stats.append(loss)
        print(str(16 * 2 ** (i-1)) + " neurons: " + str(loss))
        f.close()
    plt.title("Loss")
    plt.plot([1, 16, 32, 64, 128], stats)
    plt.xlabel("Neurons")
    plt.ylabel("Test loss")
    plt.show()

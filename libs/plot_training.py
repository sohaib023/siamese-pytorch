import matplotlib.pyplot as plt

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	print(list(H.history.keys()))
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["accuracy"], label="train_accuracy")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["val_accuracy"], label="val_accuracy")
	plt.title("Training Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)
import matplotlib.pyplot as plt

def plot_accuracy(model):
	# accuracy of our model
	plt.figure(figsize=(12, 6))
	plt.plot(model.history["accuracy"])
	plt.plot(model.history["val_accuracy"])
	plt.xlabel("Epochs")
	plt.ylabel("Accuracy")
	plt.title("ACCURACY OF MODEL")
	plt.legend(['training_accuracy', 'validation_accuracy'])
	plt.show()
	
def plot_loss(model):
	# loss of our model
	plt.figure(figsize=(12, 6))
	plt.plot(model.history["loss"])
	plt.plot(model.history["val_loss"])
	plt.xlabel("Epochs")
	plt.ylabel("Loss")
	plt.title("LOSS OF MODEL")
	plt.legend(['training_loss', 'validation_loss'])
	plt.show()
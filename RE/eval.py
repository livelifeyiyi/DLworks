import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

ROOT_DIR = "E:\\newFolder\\code\\RE\\BiLSTM_LSTM_output\\"


def showPlot():
	y = np.load(ROOT_DIR+"loss.npy")
	plt.plot(y)
	plt.ylabel("loss")
	plt.show()
showPlot()
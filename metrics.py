from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

im_predicted = "C:/Users/jayde/IONOS HiDrive/R&D/PROJETS/BFTE_SPARSE/Datasets/WSSS4LUAD/WSSS4LUAD/2.validation/2.validation/img/01_GB_bin.tif"
im_real = "C:/Users/jayde/IONOS HiDrive/R&D/PROJETS/BFTE_SPARSE/Datasets/WSSS4LUAD/WSSS4LUAD/2.validation/2.validation/mask/01.png"

y_pred = imread(im_predicted, plugin='pil')
y_true_rgb = imread(im_real, plugin='pil')

rows = y_true_rgb.shape[0]
cols = y_true_rgb.shape[1]

y_true = np.zeros([rows, cols])

for row in range(rows):
    for col in range(cols):

        rgb = y_true_rgb[row, col, :]

        if rgb[0] == 0 and rgb[1] == 64 and rgb[2] == 128:
            y_true[row, col] = 1

        elif rgb[0] == 64 and rgb[1] == 128 and rgb[2] == 0:
            y_true[row, col] = 1 #2

        elif rgb[0] == 243 and rgb[1] == 152 and rgb[2] == 0:
            y_true[row, col] = 2 #3

        elif rgb[0] == 255 and rgb[1] == 255 and rgb[2] == 255:
            y_true[row, col] = 3 #4

        else:
            print("An error has occurred.")

report = classification_report(y_true.flatten(), y_pred.flatten())
print(report)

# cross_entropy = log_loss(y_true, y_pred)
accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())

print(accuracy)

cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

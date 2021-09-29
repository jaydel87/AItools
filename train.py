import imageOperations as im
import imageFeatures as imf
import imageAI as ai
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Ridge
from sklearn import tree
import matplotlib.pyplot as plt
import random as rnd
import time

# Initialise image properties
image = im.newImage()
image.numberOfCategories = 4

# Read and image and get its properties
trainingImage = im.readImage(image)
rows = trainingImage.shape[0]
cols = trainingImage.shape[1]
channels = image.imageChannels

# Name the pop-up window: Necessary for interactivity
cv2.namedWindow(image.imagePath, cv2.WINDOW_GUI_EXPANDED)

# Gives each pixel in the image a unique ID
pixelID = np.arange(0, rows*cols)

# Initialise a dataframe for the image information with a column "PixelID"
features_df = pd.DataFrame(data=pixelID, columns=["PixelID"])

# Add a column(s) for each pixel value. Data is stored in 1D vector format.
# For greyscale images: one column for Grey value
# For RGB images: one column for each value of R, G and B
# TO DO: HIDE THIS IN imageFeatures.py
if channels == 1:
    features_df["GreyValue"] = trainingImage.flatten()
    im.channelNames = [""]
    dispImage = trainingImage
elif channels == 3:
    features_df["BlueValue"] = trainingImage[:, :, 0].flatten()
    features_df["GreenValue"] = trainingImage[:, :, 1].flatten()
    features_df["RedValue"] = trainingImage[:, :, 2].flatten()
    im.channelNames = ["Blue", "Green", "Red"]
    dispImage = cv2.cvtColor(trainingImage, cv2.COLOR_BGR2RGB)
else:
    print("Strange number of channels detected!")

featureNames = []

# Calculate the features and add a column for each feature and each filter size in the dataframe.
# As before, data is stored in 1D vector format
for i in range(channels):
    features = imf.newFeatureSet()
    features.kernelSizes = [3, 7]
    image_features, feature_sizes = imf.get_selected_features(features, trainingImage, channels, i, features.kernelSizes)
    featureNames = [x + im.channelNames[i] for x in features.featureNames]

    for j in range(len(featureNames)):
        features_df[featureNames[j]] = features.features[j]

# Prints statistics from dataframe
#print(features_df.describe())

# Initialise a blank image and then fill it with pixels selected for training the model (those drawn by user)
trainingPixels = np.zeros([rows, cols])
for n in range(image.numberOfCategories):
    r = rnd.randint(0, 255)
    g = rnd.randint(0, 255)
    b = rnd.randint(0, 255)

    colour = (r, g, b)

    categories = im.getCategories(dispImage, colour)
    # Draw lines for training and store the features/categories corresponding to these pixels
    cv2.setMouseCallback(image.imagePath, categories.record_pixels)

    im.displayImage(image, dispImage)

    # Insert an integer value corresponding to the category of the pixel into the empty image
    for i in range(len(categories.xlist)):
        trainingPixels[categories.ylist[i], categories.xlist[i]] = n+1

# Display the image indicating the pixels selected by the user.
# Each category has a unique colour which is selected at random.
im.displayImage(image, dispImage)

# Create a dataframe containing all information relevant to the pixels to be used for training the model
features_df['Category'] = trainingPixels.flatten()
idx = features_df['Category'].to_numpy().nonzero()
training_df = features_df.iloc[idx]
#print(training_df.describe())
# X contains the features, y is the category.
y = training_df.Category
X = training_df.copy()
X.drop(['Category'], axis=1, inplace=True)
X.drop(['PixelID'], axis=1, inplace=True)

# Copy the initial dataframe for later applying the trained model
# Remove columns which should not be included for the prediction
X_pred = features_df.copy()
X_pred.drop(['Category'], axis=1, inplace=True)
X_pred.drop(['PixelID'], axis=1, inplace=True)

# Split the selected pixels into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, train_size=0.75, test_size=0.25)

# Initialisation of model hyperparameters (for random forest/gradient boosting)
trees = 100
depth = 6
rate = 0.1
L1reg = 0.1
L2reg = 1
subsample = 0.8
colsample = 0.5

model = ai.RandomForest(random_state=1, n_estimators=trees, max_depth=depth, bootstrap=True, max_samples=subsample)
model_boost = ai.GradientBooster("xgboost", booster='gbtree', random_state=1, n_estimators=trees, max_depth=depth,
                                     learning_rate=rate, reg_alpha=L1reg, reg_lambda=L2reg, subsample=subsample,
                                     colsample_bytree=colsample, tree_method='auto')
t0 = time.time()
ai.train_model(model, train_X, train_y)
val_pred = ai.predict(model, val_X)
train_pred = ai.predict(model, train_X)
prediction = ai.predict(model, X_pred)
train_loss = log_loss(train_y, model.predict_proba(train_X))
valid_loss = log_loss(val_y, model.predict_proba(val_X))
prob_cat, prob_all = ai.model_probability(model, X_pred, prediction)
t1 = time.time()
print("Time (s) for random forest: "+str(t1-t0))

#ai.plot_tree(image.imagePath, model, 5, train_X.columns, [str(x) for x in np.linspace(1, image.numberOfCategories, image.numberOfCategories)])

t0 = time.time()
#ai.train_model(model_boost, train_X, train_y, early_stopping_rounds=10, eval_set=[(val_X, val_y)], verbose=False)
ai.train_model(model_boost, train_X, train_y)
val_pred_boost = ai.predict(model_boost, val_X)
prediction_boost = ai.predict(model_boost, X_pred)
prob_boost_cat, prob_boost_all = ai.model_probability(model_boost, X_pred, prediction_boost)
train_loss_boost = log_loss(train_y, model_boost.predict_proba(train_X))
valid_loss_boost = log_loss(val_y, model_boost.predict_proba(val_X))
t1 = time.time()
print("Time (s) for gradient boosting: "+str(t1-t0))

print(valid_loss, valid_loss_boost)
print(np.mean(prob_all), np.mean(prob_boost_all))

for i in range(len(X_pred.columns)):
    print(X_pred.columns[i], model_boost.feature_importances_[i])

ai.save_model(model, "RandomForest1", image_features, feature_sizes)
ai.save_model(model_boost, "GradientBoost1", image_features, feature_sizes)

fig1 = plt.figure()
plt.imshow(prediction.reshape((rows, cols)), cmap='Greys')

fig2 = plt.figure()
plt.imshow(prediction_boost.reshape((rows, cols)), cmap='Greys')

fig3 = plt.figure()
plt.imshow(prob_boost_all.reshape((rows, cols)), cmap='Greys')

# fig3 = plt.figure()
# plt.imshow(prob_all.reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)
#
# fig4 = plt.figure()
# plt.imshow(prob_boost_all.reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)

plt.show()


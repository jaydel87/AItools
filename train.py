import imageOperations as im
import imageFeatures as imf
import imageAI as ai
import cv2
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, SMOTEN, BorderlineSMOTE
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, KFold, learning_curve
from sklearn.metrics import mean_absolute_error, accuracy_score, log_loss, plot_confusion_matrix, precision_score, recall_score, SCORERS
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Ridge
from sklearn import tree
import matplotlib.pyplot as plt
import random as rnd
import time
import os.path
from PIL import Image
import easygui
import glob
from skimage.io import imread
import random

# Initialise image properties
image = im.newImage()
image.numberOfCategories = 3

# Read and image and get its properties
trainingImage = im.readImage(image)
rows = trainingImage.shape[0]
cols = trainingImage.shape[1]
channels = image.imageChannels

image.useSavedPixels = 0
image.noDraw = 0

print(image.imagePath)

imName = image.imagePath.rsplit(".", 1)[0]
feature_df_name = imName + "_features.csv"
train_df_name = imName + "_train.csv"

if image.useSavedPixels == 1:
    if not (os.path.isfile(feature_df_name) and os.path.isfile(train_df_name)):
        print("No file was found for image features and/or training pixels. Please select pixels for training.")
        image.useSavedPixels = 0

# Name the pop-up window: Necessary for interactivity
cv2.namedWindow(image.imagePath, cv2.WINDOW_GUI_EXPANDED)

# Gives each pixel in the image a unique ID
pixelID = np.arange(0, rows*cols)

if image.useSavedPixels == 0:
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
            print(features.features[j])
            features_df[featureNames[j]] = features.features[j]

    # Prints statistics from dataframe
    #print(features_df.describe())

    # Initialise a blank image and then fill it with pixels selected for training the model (those drawn by user)
    trainingPixels = np.zeros([rows, cols])

    if image.noDraw == 1:
        image.maskPath = easygui.diropenbox("Please select a folder")

        masks = sorted(glob.glob(image.maskPath+'\*.png'))

    for n in range(image.numberOfCategories):
        r = rnd.randint(0, 255)
        g = rnd.randint(0, 255)
        b = rnd.randint(0, 255)

        colour = (r, g, b)

        categories = im.getCategories(dispImage, colour)


        if image.noDraw == 1:
            maskArray = imread(masks[n], plugin='pil')

            y, x = np.where(maskArray < 255)

            list_idx = [i for i in range(len(y))]
            #random.shuffle(list_idx)
            print('Crop the list')
            start = 150000
            list_idx = list_idx[start+ 0:start+10000]

            categories.ylist = y[list_idx]
            categories.xlist = x[list_idx]

        else:

            print("Draw pixels.")

            # Draw lines for training and store the features/categories corresponding to these pixels
            cv2.setMouseCallback(image.imagePath, categories.record_pixels)

            im.displayImage(image, dispImage)

        # Insert an integer value corresponding to the category of the pixel into the empty image
        print("Put in list")
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

    features_df.to_csv(feature_df_name)
    training_df.to_csv(train_df_name)

else:
    features_df = pd.read_csv(feature_df_name)
    training_df = pd.read_csv(train_df_name)

print("Setting up dataframes")
y = training_df.Category
X = training_df.copy()
X.drop(['Category'], axis=1, inplace=True)
X.drop(['PixelID'], axis=1, inplace=True)

# Copy the initial dataframe for later applying the trained model
# Remove columns which should not be included for the prediction
X_pred = features_df.copy()
X_pred.drop(['Category'], axis=1, inplace=True)
X_pred.drop(['PixelID'], axis=1, inplace=True)

print("Split data into test and train sets")
# Split the selected pixels into training and validation data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, train_size=0.75, test_size=0.25)

strategy = {1:1000, 2:1000, 3:1000, 4:1000}
strategy = 'all'
oversample = SMOTE(sampling_strategy=strategy)
X_smote, y_smote = oversample.fit_resample(X, y)

counter = Counter(y)
for k,v in counter.items():
    per = v / len(y) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

counter = Counter(y_smote)
for k,v in counter.items():
    per = v / len(y_smote) * 100
    print('Class=%d, n=%d (%.3f%%)' % (k, v, per))


# Initialisation of model hyperparameters (for random forest/gradient boosting)
trees = 200
depth = 10
rate = 0.1
L1reg = 0.5
L2reg = 0.1
subsample = 0.5
colsample_tree = 0.5
colsample_level = 0.5

t0 = time.time()
discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
mi_scores = mi_scores.sort_values(ascending=False)

t1 = time.time()
# print("Time (s) to evaluate MI scores: "+str(t1-t0))
#
# for name, value in mi_scores.items():
#     print(name, value)


print("Setting up models")
model = ai.RandomForest(random_state=1, n_jobs=-1)
model_boost = ai.GradientBooster("xgboost", booster='gbtree', random_state=1, n_estimators=trees, max_depth=depth,
                                 learning_rate=rate, reg_alpha=L1reg, reg_lambda=L2reg, subsample=subsample,
                                 colsample_bytree=colsample_tree, colsample_bylevel=colsample_level,
                                 tree_method='auto')

print(sorted(SCORERS.keys()))

kf = KFold(n_splits=3, random_state=None, shuffle=True)

scores = cross_val_score(model, X, y, cv=kf, scoring='precision_macro', n_jobs=-1)
scores_boost = cross_val_score(model_boost, X, y, cv=kf, scoring='precision_macro', n_jobs=-1)

print(scores)
print(scores_boost)

#print(learning_curve(model, X, y, cv=kf, n_jobs=-1))

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(model, X, y, cv=kf, n_jobs=-1, return_times=True)
train_sizes_boost, train_scores_boost, test_scores_boost, fit_times_boost, _boost = learning_curve(model_boost, X, y, cv=kf, n_jobs=-1, return_times=True)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

train_scores_boost_mean = np.mean(train_scores_boost, axis=1)
train_scores_boost_std = np.std(train_scores_boost, axis=1)
test_scores_boost_mean = np.mean(test_scores_boost, axis=1)
test_scores_boost_std = np.std(test_scores_boost, axis=1)
fit_times_boost_mean = np.mean(fit_times_boost, axis=1)
fit_times_boost_std = np.std(fit_times_boost, axis=1)

xx = X_smote
yy = y_smote

t0 = time.time()
ai.train_model(model, xx, yy)
prediction = ai.predict(model, X_pred)
prob_cat, prob_all = ai.model_probability(model, X_pred, prediction)
t1 = time.time()
print("Time (s) for random forest: "+str(t1-t0))

#ai.plot_tree(image.imagePath, model, 5, train_X.columns, [str(x) for x in np.linspace(1, image.numberOfCategories, image.numberOfCategories)])

# test = [0.1, 0.3, 0.5, 0.7, 1]
# train_loss = []
# valid_loss = []

t0 = time.time()
ai.train_model(model_boost, xx, yy)
prediction_boost = ai.predict(model_boost, X_pred)
prob_boost_cat, prob_boost_all = ai.model_probability(model_boost, X_pred, prediction_boost)
t1 = time.time()
print("Time (s) for gradient boosting: "+str(t1-t0))

# t0 = time.time()
# ai.train_model(model_boost, X_smote, y_smote)
# val_pred = ai.predict(model_boost, val_X)
# prediction= ai.predict(model_boost, X_pred)
# prob_cat, prob_all = ai.model_probability(model_boost, X_pred, prediction)
# train_loss = log_loss(y_smote, model_boost.predict_proba(X_smote))
# valid_loss = log_loss(val_y, model_boost.predict_proba(val_X))
# t1 = time.time()
# print("Time (s) for gradient boosting: "+str(t1-t0))

"""
model = ai.RandomForest(random_state=1, n_estimators=trees, max_depth=depth, bootstrap=True, max_samples=subsample,
                        n_jobs=-1)
model_boost = ai.GradientBooster("xgboost", booster='gbtree', random_state=1, n_estimators=trees, max_depth=depth,
                                 learning_rate=rate, reg_alpha=L1reg, reg_lambda=L2reg, subsample=subsample,
                                 colsample_bytree=colsample_tree, colsample_bylevel=colsample_level,
                                 tree_method='auto')
                                 
trees = 100
depth = 6
rate = 0.1
L1reg = 0.1
L2reg = 1
subsample = 0.8
colsample_tree = 0.5
colsample_level = 0.5
"""


# parameter_space_RF = {
#     'n_estimators': [10, 20, 50, 100, 200, 500, 1000],
#     'max_depth': [2, 4, 6, 8, 10, 20, 50, 100],
#     'max_samples': [0.2, 0.6, 0.8, 1.0]
# }
#
# parameter_space_GB = {
#     'n_estimators': [100, 200, 500, 1000],
#     'max_depth': [2, 4, 6, 8, 10, 20, 50],
#     'learning_rate': [0.01, 0.1],
#     'reg_alpha': [0, 0.1, 0.2, 0.5, 1],
#     'reg_lambda': [0, 0.1, 0.2, 0.5, 1],
#     'colsample_bytree': [0.2, 0.6, 0.8, 1.0],
#     'colsample_bylevel': [0.2, 0.6, 0.8, 1.0]
# }
#
# clf = GridSearchCV(model_boost, parameter_space_GB, n_jobs=-1, cv=kf, refit=True)
# clf.fit(X, y)
# print('Best parameters found:\n', clf.best_params_)
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

#



# parameter_space = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (20, 20, 20), (50, 150, 50), (100, 100, 100), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam'],
#     'alpha': [0.0001, 0.005, 0.05, 0.1],
#     'learning_rate': ['constant','adaptive'],
# }
#
# clf = GridSearchCV(model_mlp, parameter_space, n_jobs=-1, cv=3)
# clf.fit(train_X, train_y)
#
# # Best parameter set
# print('Best parameters found:\n', clf.best_params_)
#
# # All results
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# plt.plot(test, train_loss, label="Training loss")
# plt.plot(test, valid_loss, label="Validation loss")
# plt.xlabel("Colsample_bylevel")
# plt.ylabel("Loss")
# plt.xlim([test[0], test[-1]])
# #plt.ylim(0, 0.05)

# for i in range(len(X_pred.columns)):
#     print(X_pred.columns[i], model_boost.feature_importances_[i])

ai.save_model(model, "RandomForest", image_features, feature_sizes)
ai.save_model(model_boost, "GradientBoost", image_features, feature_sizes)

fig1 = plt.figure()
prediction_arr = prediction.reshape((rows, cols))
plt.imshow(prediction.reshape((rows, cols)), cmap='Greys')
prediction_im = Image.fromarray(prediction_arr)
prediction_im.save(imName + "_bin.tif")

fig2 = plt.figure()
prediction_arr = prediction_boost.reshape((rows, cols))
plt.imshow(prediction_boost.reshape((rows, cols)), cmap='Greys')
prediction_im = Image.fromarray(prediction_arr)
prediction_im.save(imName + "_GB_bin.tif")

# fig3 = plt.figure()
# plt.imshow(prob_boost_all.reshape((rows, cols)), cmap='Greys')

# fig3 = plt.figure()
# plt.plot(train_sizes, train_scores_mean)
# plt.plot(train_sizes, test_scores_mean)
#
# fig4 = plt.figure()
# plt.plot(train_sizes_boost, train_scores_boost_mean)
# plt.plot(train_sizes_boost, test_scores_boost_mean)

# fig5 = plot_confusion_matrix(model, val_X, val_y, display_labels=["BG", "Green", "Blue", "Red"])
# fig6 = plot_confusion_matrix(model_boost, val_X, val_y, display_labels=["BG", "Green", "Blue", "Red"])
# fig7 = plot_confusion_matrix(model_mlp, val_X, val_y, display_labels=["BG", "Green", "Blue", "Red"])

# fig5 = plot_confusion_matrix(model, val_X, val_y, display_labels=["BG", "Target"])
# fig6 = plot_confusion_matrix(model_boost, val_X, val_y, display_labels=["BG", "Target"])
# fig7 = plot_confusion_matrix(model_mlp, val_X, val_y, display_labels=["BG", "Target"])

# fig3 = plt.figure()
# plt.imshow(prob_all.reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)
#
# fig4 = plt.figure()
# plt.imshow(prob_boost_all.reshape((rows, cols)), cmap='Greys', vmin=0, vmax=1)

plt.show()


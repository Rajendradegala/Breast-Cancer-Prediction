# Breast-Cancer-Prediction
 Breast cancer prediction can be approached using machine learning algorithms. By analyzing a dataset that includes various features related to breast cancer, such as age, tumor size, and lymph node status, a predictive model can be built. 


The suggested method uses a trained deep learning neural network system to categorize breast cancer subtypes. According to data from 221 actual patients, the findings have an accuracy of 90.50 percent. Without needing any human intervention, this model can classify and identify breast cancer lesions.



![Preprocessing data](https://github.com/Rajendradegala/Breast-Cancer-Prediction/assets/140039152/a9fee7a7-a692-4433-be2a-1b14c398dcb2)

1.DATA  PREPROCESSING:

Data preprocessing is an essential step in working with datasets. It involves cleaning and transforming the data to make it suitable for analysis. This can include handling missing values, removing duplicates, scaling numerical features, encoding categorical variables, and more. By preprocessing the data, we can ensure that it is in a format that can be effectively used by machine learning algorithms. It's like preparing the ingredients before cooking a delicious meal! 


The Wisconsin Diagnostic Breast Cancer dataset is utilized, consisting of various features related to tumor characteristics. The dataset is explored, visualized, and divided into training and testing sets. A logistic regression model is trained and evaluated using accuracy metrics.


Step-by-step features processing:

Bra_Size: Although it looks numerical, it only ranges from 28 to 48, with most of the sizing lying around 34-38. It makes sense to convert this to categorical dtype. We'll fill the NA values into an 'Unknown' category. We can see above that most of the buyers have a bra-sizing of 34 or 36.


Bust- We can see by looking at the values which are not null, that bust should be an integer dtype. We also need to handle a special case where bust is given as - '37-39'.
We'll replace the entry of '37-39' with the mean, i.e.- 38, for analysis purposes. Now we can safely convert the dtype to int. However, considering that roughly 86% of the bust data is missing, eventually it was decided to remove this feature.


Category- none missing; change to dtype category.


Cup Size- Change the dtype to category for this column. This col has around 7% missing values. Taking a look at the rows where this value is missing might hint us towards how to handle these missing values.


mc_df.bra_size = mc_df.bra_size.fillna('Unknown')


mc_df.bra_size = mc_df.bra_size.astype('category').cat.as_ordered()


mc_df.at[37313,'bust'] = '38'

mc_df.bust = mc_df.bust.fillna(0).astype(int)

mc_df.category = mc_df.category.astype('category')

missing_data = pd.DataFrame({'total_missing': mc_df.isnull().sum(), 'perc_missing': (mc_df.isnull().sum()/82790)*100})

missing_data

![MISSING DATA](https://github.com/Rajendradegala/Breast-Cancer-Prediction/assets/140039152/5e8d81df-59f0-4ebb-a01f-f8209bf8de44)


![BREAST ANALYSIS](https://github.com/Rajendradegala/Breast-Cancer-Prediction/assets/140039152/cddab1fa-7d47-4675-9150-6c20a96194a0)


2.FEATURE SELECTION AND ENGINEERING:

Feature selection is a way of selecting the subset of the most relevant features from the original features set by removing the redundant, irrelevant, or noisy features.

Supervised Feature Selection technique:

Supervised Feature selection techniques consider the target variable and can be used for the labelled dataset.

Unsupervised Feature Selection technique:

Unsupervised Feature selection techniques ignore the target variable and can be used for the unlabelled data.

![FEATURE SELECTION](https://github.com/Rajendradegala/Breast-Cancer-Prediction/assets/140039152/ba4b7d4a-ef05-4750-a3bf-c557ffa2c107)

Materials and Methodology:

This section discusses the structure of the proposed approach along with the corresponding datasets.

 Dataset Description
 
we have used three breast cancer datasets: WBCO, WDBC, and WPBC acquired from the UCI 

machine learning . Table 1 contains a summary of the sample dataset.

Table 1. Sample dataset

Data Set Short Name No. of 

Attributes No. of Instances

No. of Class

Wisconsin Breast Cancer Original WBCO 10 699 2 (B=Benign, 
M=Malignant)

Wisconsin Diagnosis Breast Cancer WDBC 32 569 2 (B=Benign, 
M=Malignant)

Wisconsin Prognosis Breast Cancer WPBC 34 198 2 (N=Non-Recur, 
R=Recur)
3.2 Methodology

3. Machine Learning Model(SVM):

SVMs work by finding the hyperplane that best separates the two classes of data: Benign and Malignant tumors.

A hyperplane is a line or a plane that divides the data into two regions. The optimal hyperplane is the one that maximizes the margin between the two classes.

SVMs have several advantages for cancer classification, including:

-They are very accurate, especially for small datasets.

-They are robust to noise and outliers.

-They can be used to classify tumors of different types and stages.

-They can be used to predict the risk of cancer recurrence.



Machine learning algorithms can be used to help doctors diagnose cancer more accurately and efficiently.

 I will explain how support vector machines (SVMs) can be used for cancer classification:

How SVMs Work for Cancer Classification
SVMs work by finding the hyperplane that best separates the two classes of data: benign and malignant tumors.

A hyperplane is a line or a plane that divides the data into two regions. The optimal hyperplane is the one that maximizes the margin between the two classes.

The margin is the distance between the hyperplane and the closest points of each class. The larger the margin, the more confident the SVM is in its decision.

In the case of cancer classification, the two classes of data are benign and malignant tumors.

The SVM will find the hyperplane that best separates these two data classes.

The closer the points are to the hyperplane, the more confident the SVM is in its decision.

Advantages of Using SVMs for Cancer Classification

SVMs have several advantages for cancer classification, including:


import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

# Load the breast cancer dataset

cancer_data = load_breast_cancer()

# Select two features for the 2D plot (e.g., feature 0 and feature 1)

X = cancer_data.data[:, [0, 1]]

y = cancer_data.target

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier with RBF kernel

classifier = SVC(kernel=’rbf’, C=1.0)

# Train the classifier

classifier.fit(X_train, y_train)

# Predict the labels for the test data

predictions = classifier.predict(X_test)

# Calculate the accuracy

accuracy = np.mean(predictions == y_test)

print(“Accuracy:”, accuracy)

# Plot the decision boundary

plt.figure()

plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test)

ax = plt.gca()

xlim = ax.get_xlim()

ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50), np.linspace(ylim[0], ylim[1], 50))

Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, colors=’k’, levels=[-1, 0, 1], alpha=0.5, linestyles=[‘ — ‘, ‘-’, ‘ — ‘])

plt.xlabel(‘Feature 0’)

plt.ylabel(‘Feature 1’)

plt.title(‘SVM Decision Boundary for Breast Cancer Classification’)

plt.show()


![BOUNDARY](https://github.com/Rajendradegala/Breast-Cancer-Prediction/assets/140039152/ed29a57c-0be6-465b-ae81-3c805c65f4ed)

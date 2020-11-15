# Credit_Risk_Analysis
Machine Learning and Predicting Credit Risk\\

### 0. Project Overview

We are tasked with assessing the accuracy and useability of machine learning methods to predict credit risk. In this project, we will use various shallow learning methods and later compare for accuracy. In the first section, we will using resampling models. 

### 1. Resampling Models - Data Cleaning

 As a requirement for utilize the scikitlearn libraries, our data need to be strictly numerical. This is treated in this section by using the pandas .get_dummies() function to make unique, dichotomous columns for each unique string in non-numerical fields.

```
X = pd.get_dummies(X, columns=['home_ownership', 'verification_status', 'issue_d', 'pymnt_plan', 'initial_list_status', 'next_pymnt_d', 'application_type', 'hardship_flag', 'debt_settlement_flag'])

```

To start our oversampling methods, we break the dataset into two parts: test and train.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```
#### 2a. Oversampling

Oversampling is a method used by duplicating samples from the minority class. Since we believe there is an imbalance in the distribution of outcomes in our dataset, we increase the bias of the minority class (results we see less). Here, we determine which oversampling algorithm, either Naive Random or SMOTE, results in the best performance. Each algorithm will be summarized in the respective, named subsection below.

We find SMOTE oversampling to have the higher accuracy score.

#### 2b. Naive Random Oversampling

First, we initiate a random oversampling instance with scikitlearn's RandomOverSampler method.

```
ros = RandomOverSampler(random_state=1)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
Counter(y_resampled)
```

Then, our model is generated using the LogisticRegresssion function. We find the model to have a balanced accuracry score of  0.64946 and the following confusion matrix and imbalanced classification report:

LINK NAIVE RANDOM SCREENIE


#### 2c. SMOTE Oversampling

First, we initiate a random oversampling instance with scikitlearn's RandomOverSampler method.

```
X_resampled, y_resampled = SMOTE(random_state=1, ratio=1.0).fit_resample(X_train, y_train)
Counter(y_resampled)
```

Then, our model is generated using the LogisticRegresssion function. We find the model to have a balanced accuracry score of  0.65844 and the following confusion matrix and imbalanced classification report:

LINK SMOTE IMAGES SCREENIE

#### 3. Undersampling

Undersampling is the converse to Oversampling, where we are deleting samples from the majority class. This will decrease the bias of the more common results. here, we use the ClusterCentroids resampler as our undersampling tool.
```
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
```
We find the undersampling method to have a balanced accuracry score of 0.54744 and the following confusion matrix and imbalanced classification report:
#### 4. Combination
In combination, you guessed it, we use both over and undersamppling to balance the learning model. This is completed using the SMOTEENN algorithm.
```
sm = SMOTEENN(random_state=1)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
Counter(y_resampled)
```

This yields an accuracy score of 0.64892 and the following confusion matrix and imbalanced classification report:

LINK SMOTEENN IMAGE HERE


# Credit_Risk_Analysis
Machine Learning and Predicting Credit Risk

### 1. Project Overview

We are tasked with assessing the accuracy and useability of machine learning methods to predict credit risk. In this project, we will use various shallow learning methods and later compare for accuracy. In the first section, we will using resampling models. In the next section, we wrap up using ensemble methods like RandomForest and EasyEnsemble. Lastly, results are summarized and a 'winner' is announced. Comments are given on the actionability of the results. 

### 2. Resampling Models - Data Cleaning

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

![NaiveRandom](https://github.com/DenverSherman/Credit_Risk_Analysis/blob/main/images/NaiveRandom.png)


#### 2c. SMOTE Oversampling

First, we initiate a random oversampling instance with scikitlearn's RandomOverSampler method.

```
X_resampled, y_resampled = SMOTE(random_state=1, ratio=1.0).fit_resample(X_train, y_train)
Counter(y_resampled)
```

Then, our model is generated using the LogisticRegresssion function. We find the model to have a balanced accuracry score of  0.65844 and the following confusion matrix and imbalanced classification report:

![SMOTE](https://github.com/DenverSherman/Credit_Risk_Analysis/blob/main/images/SMOTE.png)

#### 3. Undersampling

Undersampling is the converse to Oversampling, where we are deleting samples from the majority class. This will decrease the bias of the more common results. here, we use the ClusterCentroids resampler as our undersampling tool.
```
cc = ClusterCentroids(random_state=1)
X_resampled, y_resampled = cc.fit_resample(X_train, y_train)
Counter(y_resampled)
```
We find the undersampling method to have a balanced accuracry score of 0.54744 and the following confusion matrix and imbalanced classification report:

![ClusterCentroid](https://github.com/DenverSherman/Credit_Risk_Analysis/blob/main/images/ClusterCentroid.png)

#### 4. Combination Resampling
In combination, you guessed it, we use both over and undersamppling to balance the learning model. This is completed using the SMOTEENN algorithm.
```
sm = SMOTEENN(random_state=1)
X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
Counter(y_resampled)
```

This yields an accuracy score of 0.64892 and the following confusion matrix and imbalanced classification report:

![Combination](https://github.com/DenverSherman/Credit_Risk_Analysis/blob/main/images/Combination.png)

#### 5. Ensemble Algorithms

Branching away from resampling, we explore the use of RandomForest and EasyEnsemble in the following two sections. As always, the dataset needs to be prepped before use in our machine learning environment. For this, we use the pandas .get_dummies method. This step was executed in section 1 and is repeated here for a new jupyter notebook.

#### 5a. RandomForestClassifier

Starting with a RandomForest algorithm, our training set is fit and prediction y-values generated.
```
rf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=1)
rf_model = rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
```
We find the Random Forest Model to have a balanced accuracry score of 0.78855 and the following confusion matrix and imbalanced classification report:

![BRF](https://github.com/DenverSherman/Credit_Risk_Analysis/blob/main/images/BRF.png)

#### 5b. Easy Ensemble AdaBoost Classifier

Our last ensemble method will be the EasyEnsemble AdaBoost Classifer Algorithm.
```
eec = EasyEnsembleClassifier(random_state=1)
eec.fit(X_train, y_train)
y_pred = eec.predict(X_test)

```
We find the EasyEnsemble to have the highest balanced accuracry score of 0.91548 and the following confusion matrix and imbalanced classification report:

![EEC](https://github.com/DenverSherman/Credit_Risk_Analysis/blob/main/images/EEC.png)

#### 6. Summary of Results

In summary, our highest observed balanced accuracy score belongs to the EasyEnsemble algorithm at 0.91548. Moving forward, this model seems balanced. Reviewing the randomForest classifier feature breakout, we see that loan principle, or the initial amount lent, is one of the most important features in determining risk. Random Forest doesn't produce a high accuracy score, however the outputs of the results shed light on where a regression model might be most powerful. Printing out the most important features, even if it isn't the strongest predictor, is valuable in progressing and directing our analysis as we move forward.
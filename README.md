# Regression Analysis

Regression analysis is a form of predictive modelling technique which investigates the relationship between a dependent (target) and independent variable (s) (predictor). In simple terms the regression can be defined as, “Using the relationship between variables to find the best fit line or the regression equation that can be used to make predictions”.

![Regression](https://github.com/Ansu-John/Regression-Models/blob/main/resources/Regression.png)

This technique is used for forecasting, time series modelling and finding the causal effect relationship between the variables. There are various kinds of regression techniques available to make predictions. These techniques are mostly driven by three metrics (number of independent variables, type of dependent variables and shape of regression line).

## 1. Linear Regression

The objective of a linear regression model is to find a relationship between one or more features(independent variables) and a continuous target variable(dependent variable). When there is only feature it is called Uni-variate Linear Regression and if there are multiple features, it is called Multiple Linear Regression.

![DiffLinearRegression](https://github.com/Ansu-John/Regression-Models/blob/main/resources/DiffLinearRegression.png)

The linear regression model can be represented by the following equation

![LinearRegression](https://github.com/Ansu-John/Regression-Models/blob/main/resources/LinearRegression.png)

**Y** is the predicted value
**θ₀** is the bias term.
**θ₁,…,θₙ** are the model parameters
**x₁, x₂,…,xₙ** are the feature values.

The line for which the the error between the predicted values and the observed values is minimum is called the best fit line or the regression line. These errors are also called as residuals. The residuals can be visualized by the vertical lines from the observed data value to the regression line.

![LinearRegressionLine](https://github.com/Ansu-John/Regression-Models/blob/main/resources/LinearRegressionLine.png)

To define and measure the error of our model we define the cost function as the sum of the squares of the residuals. The cost function is denoted by
![costFunction](https://github.com/Ansu-John/Regression-Models/blob/main/resources/costFunction.png)
where the hypothesis function h(x) is denoted by
![hypothesisFunction](https://github.com/Ansu-John/Regression-Models/blob/main/resources/hypothesisFunction.png)
and m is the total number of training examples in our data-set.

Our objective is to find the model parameters so that the cost function is minimum. We will use Gradient Descent to find this.
### Cost Function
Cost Function is a function that measures the performance of a Machine Learning model for given data.
Cost Function is basically the calculation of the error between predicted values and expected values and presents it in the form of a single real number.
Many people gets confused between Cost Function and Loss Function,
Well to put this in simple terms Cost Function is the average of error of n-sample in the data and Loss Function is the error for individual data points.In other words,Loss Function is for one training example,Cost Function is the for the entire training set.
### Gradient descent
Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost). Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm.
+ Batch gradient descent refers to calculating the derivative from all training data before calculating an update.
+ Stochastic gradient descent refers to calculating the derivative from each training data instance and calculating the update immediately.

## 2. Logistic Regression

Logistic regression is the appropriate regression analysis to conduct when the dependent variable is categorical. Like all regression analyses, the logistic regression is a predictive analysis.  Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

### Types of Logistic Regression

**1. Binary Logistic Regression** - The categorical response has only two 2 possible outcomes. Example: Spam or Not

**2. Multinomial Logistic Regression** - Three or more categories without ordering. Example: Predicting which food is preferred more (Veg, Non-Veg, Vegan)

**3. Ordinal Logistic Regression** - Three or more categories with ordering. Example: Movie rating from 1 to 5

## 3. Polynomial Regression

Polynomial Regression is a special case of Linear Regression where we fit the polynomial equation on the data with a curvilinear relationship between the dependent and independent variables. It is a form of regression analysis in which the relationship between the independent variables and dependent variables are modeled in the nth degree polynomial. 
Polynomial Regression models are usually fit with the method of least squares.The least square method minimizes the variance of the coefficients,under the Gauss Markov Theorem.

![PolynomialRegression](https://github.com/Ansu-John/Regression-Models/blob/main/resources/PolynomialRegression.png)

Polynomial Regression does not require the relationship between the independent and dependent variables to be linear in the data set,This is also one of the main difference between the Linear and Polynomial Regression. It is generally used when the points in the data are not captured by the Linear Regression Model and the Linear Regression fails in describing the best result clearly.

## 4. Decision Tree Regression

Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. A decision node has two or more branches, each representing values for the attribute tested. Leaf node represents a decision on the numerical target. The topmost decision node in a tree which corresponds to the best predictor called root node.

![DecisionTree](https://github.com/Ansu-John/Regression-Models/blob/main/resources/DecisionTree.png)

Note : Tree based models are not designed to work with very sparse features. When dealing with sparse input data (e.g. categorical features with large dimension), we can either pre-process the sparse features to generate numerical statistics, or switch to a linear model, which is better suited for such scenarios.

## 5. Random Forest Regression

A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging. The trees in random forests are run in parallel. There is no interaction between these trees while building the trees.

These results from the various decision trees are aggregated, through model votes or averaging, into a single ensemble model that ends up outperforming any individual decision tree’s output.

![RandomForest](https://github.com/Ansu-John/Regression-Models/blob/main/resources/RandomForest.png)

### Ensemble Learning
An Ensemble method is a technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual model. A model comprised of many models is called an Ensemble model.

Types of Ensemble Learning:
+ Boosting.
+ Bootstrap Aggregation (Bagging).

**1. Boosting**

Boosting refers to a group of algorithms that utilize weighted averages to make weak learners into stronger learners. Boosting is all about “teamwork”. Each model that runs, dictates what features the next model will focus on.
In boosting as the name suggests, one is learning from other which in turn boosts the learning.

Random forest is a bagging technique and not a boosting technique. 

**2. Bootstrap Aggregation (Bagging)**

Bootstrap refers to random sampling with replacement. Bootstrap allows us to better understand the bias and the variance with the dataset. Bootstrap involves random sampling of small subset of data from the dataset.
It is a general procedure that can be used to reduce the variance for those algorithm that have high variance, typically decision trees. Bagging makes each model run independently and then aggregates the outputs at the end without preference to any model.

## 6. Support Vector Regression

Support Vector Machines are a type of supervised machine learning algorithm that provides analysis of data for classification and regression analysis. While they can be used for regression, SVM is mostly used for classification.

The basic principle behind the working of Support vector machines is simple – Create a hyperplane that separates the dataset into classes. Let us start with a sample problem. Suppose that for a given dataset, you have to classify red triangles from blue circles. Your goal is to create a line that classifies the data into two classes, creating a distinction between red triangles and blue circles.

According to SVM, we have to find the points that lie closest to both the classes. These points are known as support vectors. In the next step, we find the proximity between our dividing plane and the support vectors. The distance between the points and the dividing line is known as margin. The aim of an SVM algorithm is to maximize this very margin. When the margin reaches its maximum, the hyperplane becomes the optimal one.

![SVM](https://github.com/Ansu-John/Regression-Models/blob/main/resources/SVM.png)

The SVM model tries to enlarge the distance between the two classes by creating a well-defined decision boundary. In the above case, our hyperplane divided the data. While our data was in 2 dimensions, the hyperplane was of 1 dimension. For higher dimensions, say, an n-dimensional Euclidean Space, we have an n-1 dimensional subset that divides the space into two disconnected components.

# Model evaluation 

Model evaluation leads a Data Scientist in the right direction to select or tune an appropriate model.There are three main errors (metrics) used to evaluate regression models, Mean absolute error, Mean Squared error and R2 score.

**Mean Absolute Error (MAE)** tells us the average error in units of y, the predicted feature. A value of 0 indicates a perfect fit. 

**Root Mean Square Error (RMSE)** indicates the average error in units of y, the predicted feature, but penalizes larger errors more severely than MAE. A value of 0 indicates a perfect fit. 

**R-squared (R2 )** tells us the degree to which the model explains the variance in the data. In other words how much better it is than just predicting the mean. 
+ A value of 1 indicates a perfect fit.
+ A value of 0 indicates a model no better than the mean. 
+ A value less than 0 indicates a model worse than just predicting the mean.

# REFERENCE 

https://scikit-learn.org/

https://towardsdatascience.com/linear-regression-using-python-b136c91bf0a2

https://www.statisticssolutions.com/what-is-logistic-regression/

https://medium.com/analytics-vidhya/understanding-polynomial-regression-5ac25b970e18

https://data-flair.training/blogs/svm-support-vector-machine-tutorial/

https://medium.com/swlh/random-forest-and-its-implementation-71824ced454f

https://medium.com/@bhartendudubey/decision-tree-regression-e202008c2df

https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/

https://openclassrooms.com/en/courses/6401081-improve-the-performance-of-a-machine-learning-model/6519016-evaluate-the-performance-of-a-regression-model

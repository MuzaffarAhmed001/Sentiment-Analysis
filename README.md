# Sentiment-Analysis
Predicting the Sentiments of Employee Reviews.

**1. Business Problem**

Our task would be to predict the rating/predict the text is positive or negative based on the review text. Also do some analysis of the text to get some insights and trends based on individual company/organization.
We need to create a Machine Learning Model to predict whether a given review is positive or negative.

**2. Mapping to ML Problem**

This is a Binary classification problem in which we need to predict whether a review is positive or negative based on Review text.

We could use the Score/Rating. A rating of 4 or 5 could be considered a positive review. A rating of 1 or 2 could be considered negative. A rating of 3 is neutral and ignored. This is an approximate and proxy way of determining the polarity (positivity/negativity) of a review.

**3. Dataset Overview**

This is a textual data in the form of json file. Its has more than 145k records. Each record has attributes such as

  1. Review Title — — — — — — — — — Contains Text

  2. Review Body — — — — — — — — — Contains Text about review description

  3. Review Rating — — — — — — — — -Contains Rating on a scale of 1 to 5.

  4. Reviewed Company — — — — — — -Contains Company in form of URL

  5. Review description — — — — — — Contains employee status, Date/time and location.
  
  
**4.Feature Engineering**

We are creating new Features with following two techniques.

1. Vader Sentiment Analyzer:

From this technique, four new features were created namely positive score, negative score, neutral score and compound score.

neg, neu, pos: These three scores sum up to 1. These scores show the proportion of text falling in the category.

compound: This score ranges from -1 (the most negative) to 1 (the most positive).

2. Text Blob Sentiment Analyzer:

From this technique, two new features were created namely Polarity and Subjectivity.

polarity: ranges from -1 (the most negative) to 1 (the most positive)

subjectivity: ranges from 0 (very objective) to 1 (very subjective).

In total 6 new features will be created with these two techniques. We will remove any feature, if feature is not contributing to model.

**5. Exploratory Data Analysis**

Target Variable Analysis

Here our Dataset is highly Imbalanced. Let’s have a look into it.

![target](https://user-images.githubusercontent.com/67824198/169252383-4c8020db-1b5c-4887-bb70-d5b0c3890724.png)

Here Majority class is positive review and Minority class is negative review. Here we are using class weights to balance the dataset.

Company Name

Here we are extracting the company name from URL. Let’s have a look on count plot of Company Names with respect to classes.

![company](https://user-images.githubusercontent.com/67824198/169252611-391bd8c2-5f81-45f0-8792-8d69d50bf384.png)

Here we can see that all the Companies are having more number of positive reviews than negative reviews. So this feature is not helpful in classification. We will drop this feature.

Employee Type

There are two types of Employees, namely Former Employee and Current Employee.

![employee type](https://user-images.githubusercontent.com/67824198/169253140-c94357c8-d57d-4b19-a251-4a3f6163a9fc.png)

Here we can see that both types of Employees are having more number of positive reviews than negative reviews. This feature is also not helpful in classification. We will be dropping this feature.

Same is the case for Location and Year feature. Both these features are having more number of positive reviews than negative reviews. We are dropping both of these features.

Featured Engineered Variables

Let’s have a look into Correlation Analysis of features which are created by Vader Sentiment Analyzer and Text Blob Analyzer.

![corr](https://user-images.githubusercontent.com/67824198/169253264-f4e7e5e2-27a1-4a3a-8c51-d02899e1dfdf.png)

We can see that there is high correlation between ‘pos’ and ‘neu’ feature. We need to drop one of the feature. We are dropping ‘neu’ feature .

The features which have been removed are URL, Rating, Review Details, Company name, Employee type, location, Year, Date and neu.

**6. Text to Vector.**

First of All we are cleaning the Text from URL’s, Special Characters, Punctuations and Numbers. Then we are merging Review Title and Review Body into one Column. Then we are converting our text into vector using the following Techniques.

BOW
TFIDF
Tokenizer
BOW and TFIDF can be used for both Machine learning Models and Deep Learning Models, where as Tokenizer is used for LSTM models.

**7. Modelling**

We have done the modelling using Machine learning Models, Deep learning Models and Stacking Classifier. Let's have a look into it.

Stacking Classifier.

![stack model](https://user-images.githubusercontent.com/67824198/169253872-6de4e983-6fc5-486e-addb-e2d65a263a7a.jpg)

Here in the Stacking Model, we are training the TF-IDF data using Logistic Regression, SVC and SGD model . Then we are predicting the class of data points using these Models. Then we are taking majority vote on these predictions.

**8. Results:**

Lets compare the results of different models.

![results](https://user-images.githubusercontent.com/67824198/169254333-11c3995b-4796-435d-b3d1-3bd90daa2f34.PNG)

Features created from Sentiment Analyzer improves the performance of Models.

Machine learning Models perform better with TF-IDF Data and Deep Learning Models perform better with BOW.

N grams(tri-gram) will give better results than unigram.

Deep learning Models will give good ROC AUC score compared to Machine Learning Models.

Overall Deep learning models perform better than ML models in terms of Accuracy, AUC Score and Log loss.

**Best Model is MLP Model trained on BOW Dataset. Best Accuracy achieved is 95.48%, AUC score of 0.71 and Log Loss of 0.16.**

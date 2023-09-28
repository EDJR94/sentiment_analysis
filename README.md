# Amazon Product Review Sentiment Analysis

![sentiment](https://github.com/EDJR94/sentiment_analysis/assets/128603807/27237fb6-35f8-4c2e-97b3-667b53e99d37)

# Business Problem

## Background:

Amazon, the world's largest online retailer, prides itself on customer-centricity. Customer reviews form the backbone of Amazon's product ecosystem, guiding millions of purchasing decisions daily and playing a pivotal role in vendor and product rankings. As the platform continues to scale, the sheer volume of reviews makes manual monitoring an insurmountable task.

## Problem Statement:

While Amazon's current review system with star ratings gives a quantitative insight into product quality, it lacks the granularity needed to understand the nuanced opinions of its vast user base. Moreover, with millions of reviews generated daily, manual analysis becomes an insurmountable challenge.

## The Limitations of Star Ratings:

- Lack of Specificity: A customer might give a product 3 stars, but what does that mean? Are they somewhat satisfied, or did they encounter specific issues? Without the accompanying text, it's hard to say.
- Varied Interpretations: One person's 5-star experience might be another person's 4-star experience. Relying solely on star ratings can be misleadig.
- No Actionable Insights: Star ratings don't tell you what the problems are. If a product receives a 1-star rating, what should the seller improve? The packaging? The product quality? The shipping time? Without a textual review, it's all guesswork

### Examples of Star Ratings:

- Here we have 2 opposite reviews of the same product both with 3 stars
    ![3_stars_negative](https://github.com/EDJR94/sentiment_analysis/assets/128603807/79a6b463-3d83-47cf-a9be-106700b5f039)


    ![3_stars_positive](https://github.com/EDJR94/sentiment_analysis/assets/128603807/3e970536-ba3b-45c6-aeb5-1c3e66dedf5b)

- Example of 4 stars review with Negative Sentiment
    
    ![4_stars_negative](https://github.com/EDJR94/sentiment_analysis/assets/128603807/e1d94d50-eff7-4a59-94f1-b0e44b181e3f)

    
- We can even find 5 stars with Negative Sentiment:
    
    ![5_stars_negative](https://github.com/EDJR94/sentiment_analysis/assets/128603807/501b4814-1f4d-4967-a679-e7c9fd946b4c)

    

## Objective:

Develop an automated solution using machine learning and natural language processing techniques to:

1. Accurately identify negative reviews among the millions of reviews posted on Amazon daily.
2. Categorize the severity of the negative feedback to prioritize actions.
3. Extract actionable insights from negative reviews to provide to vendors, helping them improve product quality and customer experience.

## Value Proposition:

By effectively identifying and addressing negative reviews, Amazon aims to:

1. Enhance customer trust by showing responsiveness to feedback.
2. Increase customer retention by resolving issues proactively.
3. Boost overall platform sales by improving product and vendor quality through actionable feedback.
4. Potentially save millions in revenue by preventing customer churn and fostering brand loyalty.

## Measure of Success:

A successful solution will showcase a significant improvement in the **Recall metric**, ensuring that the vast majority of negative reviews are captured. The financial implications of improved customer retention and vendor product enhancement will serve as a testament to the initiative's success.

# Data

| Column Name | Description                                   |
|-------------|-----------------------------------------------|
| sentiment   | Integer identifying one of 2 sentiments:<br>1: Negative<br>2: Positive |
| title       | Title of the review                           |
| text        | Review text                                   |


# Solution Strategy

The strategy employed was the CRISP method, a scientific method based on cycles:
![Untitled](https://github.com/EDJR94/sentiment_analysis/assets/128603807/d41a3187-1a13-4478-8933-66f512fc274e)

The project cycles were divided into the following phases:

- Problem Understanding
- Data Description
- Data Understanding
- Text Processing
- Data Preparation
- Model Training
- Model Evaluation
- Model Deployment
- Business Performance

# Data Understanding

In this section I analized how our review texts were composed, like the size of the texts, most used words, ponctuation, etc.

For example, this was the most used words in general:

![Untitled (1)](https://github.com/EDJR94/sentiment_analysis/assets/128603807/bf1e44e4-33d8-4f30-a7e5-1bc2ab6df2ab)

Notice that 'book' is most used word, because the majority of sales in amazon are from books. But there are some words that capture the sentiment that are used a lot, like: 'good', 'great', 'love'.

Here is a WordCloud of the most used words in Positive Reviews:

![Untitled (2)](https://github.com/EDJR94/sentiment_analysis/assets/128603807/12268e06-58e4-417b-86e9-d30d75703aed)

And for Negative Reviews:

![Untitled (3)](https://github.com/EDJR94/sentiment_analysis/assets/128603807/d29146c8-dc10-420f-a5ba-2f9192664f4e)

# Data Preparation and Model Training

To prepare our data for the model I used some techniques:

- Tokenization
- Word Embedding
- Lemmatization
- Remove StopWords
- Remove Ponctuation/extra spaces
- Uncase Words

You can see more detailed explanatation of the process in the notebook [here](notebooks/sentiment_analysis_LSTM_Final.ipynb).

The model used to classify the texts is a Sequential Model in Deep Learning. The main object in this model architecture is the LSTM(Long-Short Term Memory) that can capture long-range patterns and dependencies in the text. Here is a example of how the model will work:

![Untitled (4)](https://github.com/EDJR94/sentiment_analysis/assets/128603807/a96dc535-f1b0-48e8-b27d-075b4cd716de)

You can see more the model with more detail in the notebook [here](notebooks/sentiment_analysis_LSTM_Final.ipynb).

# Results

After the training here are some reviews that the model classified:

### Negative Sentiment Correctly Classified

`Original Text: The Betrothed is an excellent book, but this is not the book, and it's not obvious from looking at this page. I have a wee baby so am sleep-deprived and perhaps that's why I didn't notice, but I definitely think it could be made more obvious. I bought this for a gift anf it was a bit embarrassing.`

```
Original Sentiment: Negative
Predicted Sentiment: Negative
```

### Positive Sentiment Correctly Classified

`Original Text: This is by far the best Jeff healey album ever, and it's incredible that almost none of the songs are presented in his complitation "THE VERY BEST OF JEFF HEALEY". This fact alone killed the compilation.`

```
Original Sentiment: Positive
Predicted Sentiment: Positive
```

### Negative Sentiment Incorrectly Classified

`Original Text: "Give war a chance!" Get it? The title is a rebuke to the John Lennon song "give peace a chance!" And that is as funny as this book gets. The author attempts to be funny in this way. He is white, well to do, and adored by many conservatives and libertarians for he mocks those who try to change the world for the better. In an earlier time, he would have made fun of Negroes and injuns, but he is not that crude. Here he merely states that war is ok, just don't let me fight in one! In all, a very funny and nasty book for the fat cat on your gift list.Let others fight wars, PJ is too busy making rationales for them!`

```
Original Sentiment: Negative
Predicted Sentiment: Positive
```

## Finding Optimal Threshold for Recall Metric
By analyzing the model's error, I observed that the majority of errors occur when the probability for the positive class falls between 40% and 60%. This range is where the model tends to make the most mistakes, as illustrated in the graphs below:

![Captura de tela 2023-09-28 172927](https://github.com/EDJR94/sentiment_analysis/assets/128603807/b83d4a2b-2c36-413d-9a81-7a390592f149)

![Captura de tela 2023-09-28 172949](https://github.com/EDJR94/sentiment_analysis/assets/128603807/1ec2e31e-0d70-4519-a0dd-444dd655c0af)

I decided that, for increasing the recall the most, I will classify these uncertain probabilities as Negative, so my final threshold is 0.6:
- If Probability more than 60% -> Positive Sentiment
- If Probability less than 60% -> Negative Sentiment 

### Confusion Matrix

My data as divided as follows:

- 63,000 examples on for training the model
- 27,000 examples to validate the model
- 40,000 examples to test the model

Here is the confusion matrix for the Validation Set:

![confusion_matrix_val2](https://github.com/EDJR94/sentiment_analysis/assets/128603807/b99e3bf4-eb9e-45ee-a40f-7fc40025f8c7)

And for the Test Dataset:

![confusion_matrix_test2](https://github.com/EDJR94/sentiment_analysis/assets/128603807/831a483d-0da6-4735-90f7-9009bc1e0c5c)

Looking at the Test Confusion Matrix, we can see that our model correctly classified 16,836 positives reviews of the 20.000 Positive reviews.

Also, it correctly classified 17,010 Negative reviews of the 20.000 total Negative reviews.

### Classification Metrics

I used Accuracy, Precision, Recall and F1-Score metrics to evaluate my model, that the final metrics on the Test Set:

| Metric    | Value |
| --- | --- |
| Accuracy   | 0.8404 |
| Precision   | 0.8060 |
| Recall   | 0.8967 |
|  F1 Score | 0.8489 |

## Introducing the Neutral Sentiment Category for User Deployment
I made an App on Streamlit for users to try some reviews and get the sentiment.

Upon analyzing the predictions of our model, I observed that for certain reviews, the model exhibited uncertainty or lacked strong confidence in its predictions. To accommodate such instances and offer a more nuanced classification, I've introduced a "Neutral" sentiment category.

The revised sentiment labeling based on the model's probability output is as follows:

- **Positive Sentiment**: When there's over 60% probability for a positive outcome.
- **Negative Sentiment**: When the positive probability is less than 40%.
- **Neutral Sentiment**: For cases where the model's positive probability lies between 40% and 60%, indicating a level of uncertainty.

This approach ensures a more comprehensive and nuanced analysis of reviews, capturing sentiments that may not strictly fall into the traditional positive or negative categories.


# Deploy to Production
I deployed the model in Streamlit Sharing so you can test some reviews for yourself! Here is an example:

![gif_streamlit](https://github.com/EDJR94/sentiment_analysis/assets/128603807/35e4eaa6-d401-4b9d-86a9-a5305c2395ef)

Try to trick it by writing some tricky reviews! Link: [Streamlit](https://appsentiment-6pkqwypnalszzgk2kp9br7.streamlit.app/).

# Business Performance

## Assumptions

- On average, Amazon garners **10,000 reviews daily**.
- Of these, **5% (or 500 reviews)** are negative.
- Among the negative feedback, **20% are disguised as star ratings**. This means Amazon could potentially overlook **100 subtly critical reviews** each day.

## Cost of Dissatisfaction

- Each unidentified and unaddressed negative review risks losing a customer permanently.
- The estimated **Lifetime Value (LTV) of an Amazon customer stands at $500**, representing the potential profit derived from a customer throughout their association with the company.
- For the sake of this analysis, let's assume Amazon loses all customers associated with the **100 camouflaged negative reviews**.

## Recall: Before vs. After the Model Implementation

- Without the model in place, Amazon faces a potential loss of **100 customers daily**, translating to a revenue loss of **100 x $500 = $50,000 daily**.
- With the model's deployment, the recall rate surges to **89%**. This means Amazon now captures feedback from **84 of the camouflaged negative reviews daily**, missing only 11 such reviews.
- By this metric, instead of losing $50,000 daily, the model helps Amazon salvage 89 x $500 = $44,500 each day. Annually, this amounts to a substantial recovery of **$16,242,500**!

# References

Author: Edilson Santos, Data Scientist.

Author Linkedin: https://www.linkedin.com/in/edilsonsantosjr/

Database: https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews

Portfolio: https://edjr94.github.io/portfolio_english/

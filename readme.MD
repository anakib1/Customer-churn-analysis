
  

# User chart prediction

  Total results are stated in the [results](#Results) section. All files required to reproduce results are located in the github repo. 

## Overall data analytics

  

The code required to reproduce this section is located in the notebook named ``analytics.py``.

  

In this part we won't focus on sequential nature of the data and will view every user as a separate "set" of actions. We will create data as follows:

  

  

0) Filter out all the data of users who did not got premium subscription at all. This is useless for our task, so we won't need them.

  

1) For each type of event in the given data we add feature ``#count of event`` (# number of times user contacted support, obtained order, logged into the app etc).

  

2) Additionally we find time between first and last user transaction (in days), average between-transaction time, number of transactions, phone model and state of user.

  

  

Then, we add for each user ``Value`` - indicator that user cancelled his premium subscription.

  

Let's look at density estimations of users on this created features.

  

  

![Numerical plots](https://i.imgur.com/w1qxbEY.png)

  

  

We see that most charts show similar behavior between churned and unchurned users, but some of the charts stand out.

  

Lets examine further charts that stand out - for instance support contact charts.

  

  

![Support KDE](https://i.imgur.com/P1gKPG9.png)

  

  

As we see, churned users tend to contact support more often than unchurned ones.

  

Also, users who make more orders tend to stay in the up and don't cancel the subscription, and the ones who do a lot of actions (but no orders) tend to leave.

  

  

Let's look at categorical features

  

  

![](https://i.imgur.com/QUnIkD1.png)

  

  

Overall there is small differences between states. However, there are almost no churners from NY and there are quite a disproportionate amount of churners in both IL and FL states. This could be a signal to improve quality of services there, or to find differences to more prosperous states like CA and NY.

  

Phone model also gives us small findings. There are less churners using IOS than android, so IOS app could be slightly better, or there could be some other factors connected to the phone operating system choice.

  

  

This analysis should guide as to more detailed one using different statistical models as described in the following sections.

  

  

## Linear regression model

  

The code required to reproduce this section is located in the notebook named ``linear_regression.py``.

  

  

We use linear regression for two reasons:

  

1) It requires low resources for fitting and prediction.

  

2) It offers better interpretability compared to more sophisticated methods like Random Forests and Neural networks.

  

  

Analysis of data in the first step shows us that some features are unnecessary (for instance features like ``transaction refund``, where only 3 users have non-zero value).  We drop these features from dataset, encode the whole dataset using one-hot encodings for categorical features ``state`` and ``phone model`` and then fit two linear models on target ``Value`` (Ridge on targets -1, 1 and Logistic). 

We also obviously discard target feature ``Subscription Premium Cancel``. 

For penalty we use 5-fold cross validation on accuracy.


  

  

Our logistic regression model makes only 3 errors on the whole test dataset achieving spectacular performance of 98% accuracy.

  ![](https://i.imgur.com/BBNkNCa.png)

  

We will visualize features in three ways - using Permutational feature importances , using SHAP values and using partial dependence plots to gain insight into the model behavior.

  

  

### Permutational importances

  

We will use sklearn permutational importances module computing importances on test set as suggested in [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/feature-importance.html).

  ![](https://i.imgur.com/lm1f1zm.png)

  

Model shows that most important features are: ``number of actions``, ``order``, ``account transaction history``, ``wallet`` , ``chat conversation started``, ``chat conversation opened``.

  

  

Features with no importance are: ``State``, ``Phone model``(despite achieving non-zero coefficient in model weights), ``account setup``, ``sign up``, ``time of use``, ``average time``.

  

### SHAP feature importances

  

Using ``shap`` [package](https://github.com/slundberg/shap) for python we compute shap values importance for our model. This values could be interpreted as contribution of each particular value of feature for given instance to the total prediction. We compute these values for test set as well to maintain consistency between two shown methods.

  

  

![](https://i.imgur.com/eG9IQuc.png)

  

  

The SHAP importances show the similar result as permutational. We see high dependence on the`` number of action``, ``order``, ``support`` and other.

  

  

Also, SHAP values could be used to explain each individual prediction. We can use them to visualize all of 3 incorrect predictions.

  

  

![](https://i.imgur.com/lxF9Aw5.png)

  

![](https://i.imgur.com/52bJQzR.png)

  

![](https://i.imgur.com/0qWF2nB.png)

  

  

Two of three incorrect predictions where near the decision boundary, so it ones again confirms the power of our model.

  

  

As we see, high ``number of actions`` and low ``orders`` are pushing model for performing "churn" prediction. As well there is a strange artifact that number of ``chat support calls`` actually pushes the model to the "unchurned" prediction contrary to prevoius belief that high support contact rate means dissatisfaction with product. We believe that this could be consequence of different correlated features in the model.

  

  

### Partial dependence plots

  

  

We will use only two main features - ``number of actions`` and ``number of orders``. Given partial dependence plots show us how model understands the behavior of this features. We see positive relationship of prediction function with ``number of actions`` and negative one with ``number of orders``.

  

  

![](https://i.imgur.com/Zt1mYqA.png)

  

  

If we look at the correlation coefficient of these features it is 0.9 - highly correlated. So, these coefficients are unstable and prune to be misleading. We only note that all methods found these two features to be important for the model, but the coefficients and even their sign are prune to ambiguity.

## Decision tree

  

The code required to reproduce this section is located in the notebook named ``decision_tree.py``.

  

In search of optimal and interpretable model we could use low depth decision trees.

  

We will use the same set of features as earlier and fit a tree using 5-fold cross validation of ``min_samples_split``, ``max_depth``, ``ccp_alpha``, ``criterion``. The target of cross validation is balanced accuracy score. 

  

We get the tree like this:

  

![](https://i.imgur.com/IOjiqqa.png)

  

This tree gets comforting accuracy 0.83 on both test and train set, making the model viable.

![](https://i.imgur.com/BBrGNrM.png)

 The model uses only features ``Chat Conversation Started``, ``Wallet Opened``, ``Time of use`` and is extremely simple. We see that churned customers have started support chat, and used the app for less then 63 days. Also people who often open their wallet are classified as unchurned, but decision boundary there is vague, so this feature also characterizes churners.

  

![](https://i.imgur.com/9YwnPSV.png)

  

SHAP values importances for this tree also show the same results as stated above. Number of support contacts dominates the importances.

  

**We note, that single decision tree is extremely sensitive to random seed and unstable. This model is only a demonstration of possible explanation of data, it might generalize poorly.**

## Random Forest

The code required to reproduce this section is located in the notebook named ``random_forest.ipynb``.


To overcome the decision tree limitations of high variance we could use Random Forests - ensemble of randomized trees. 

We fit Random Forest model on the same features as above and tune hyper parameters using 5-fold cross validation on balanced accuracy metric. 

Final model is an ensemble of 100 trees with maximal depth equal to 9, entropy criterion, without bootstrap and with entropy criterion. 
Model achieves 90% accuracy on train test and 80% accuracy on the train set which might suggest slight overfit of data. Nevertheless, we are interested in feature importances. 

![](https://i.imgur.com/OldsOy4.png)

Tree feature importances are following:

![](https://i.imgur.com/9KalLN3.png)

We see strong domination by ``chat conversation`` feature as in a decision tree. Other notable features are ``number of actions``, ``wallet opened``, ``average time``, ``history transactions``. Ensemble does not use ``payment method``,  all account setup features and uses state and phone model only slightly. 

![](https://i.imgur.com/NNzb5Zj.png)

SHAP values suggest similar behavior. The most important features are ``chat conversation``, ``wallet opened``, ``number of actions``. 

Viewing dependence on the ``wallet`` feature we see weak positive correlation with churn. 

![](https://i.imgur.com/U6mgWK3.png)
On ``Support`` we see obvious positive correlation. 

![](https://i.imgur.com/kkd38xg.png)

On ``Actions`` we see also positive correlation with churn.

![](https://i.imgur.com/RBHis6b.png)

And, we see negative correlation with churn on ``Orders`` as expected

![](https://i.imgur.com/761Qcww.png)
If we look on not so important (for this particular model) feature time we see strong negative correlation with churn for long-term customers

![](https://i.imgur.com/Idem82W.png)


## RNN
The code required to reproduce this section is located in the notebook named ``rnn.ipynb``.


Our last experiment was to try to use feature sequential dependencies and to train recurrent neural network model for action predictions. This experiment did not succeed as well as previous models, but it has given interesting results and ideas for similar task and so was included in the report. 

### Architecture

We have chosen LSTM model with embedding dimension of 32 neurons, size of hidden layer of 64 neurons, and linear layer between LSTM outputs and prediction to the dictionary space. Before the final layer output of the sequential part of input was concatenated to the user embedding based on phone model and state. 

### Training
Model was trained on cross-entropy with increased weight for ``subscription cancel`` prediction. The goal of training was to predict next event in the sequence. Model was validated as "accuracy" metric on separate test set - where we defined accuracy as an ability to predict will customer churn or not. LSTM model was not explicitly penalized for small "accuracy", so it did not learn to just output whether churn was present in the sequence - it learned to predict the whole sequence of events. it achieved "accuracy" of 91% on train set and 72% on test set. 

![](https://i.imgur.com/ig5ANmg.png)


### Embedding analysis 

However, for our experiment it was not the main goal. 

After trained LSTM we trained Logistic Regression model on the sequence embeddings from LSTM (Long memory embeddings ``c`` as they showed superior performance) and targets as in earlier sections (``Value``). This Logistic Regression model achieved 93% test set accuracy (94% train) beating RF and DT models by significant margin. 

![](https://i.imgur.com/jjdFAtS.png)

It proofs that LSTM learned meaningful embeddings (that maybe indicate explicitly whether ``subscription cancelled`` is present in the sequence, so they are useless by themself). What is really important is a space spanned by this embedding. We clustered embeddings of all test sequences using KMedoids algorithm and examined centroids of derived clusters. They are shown on the picture. (Top row are embeddings of churned customers and bottom row is for not churned users). 

![](https://i.imgur.com/YTisAhp.png)

We see some visual differences between plots and obviously, logistic regression sees even more. Here are embeddings activations after dot product with logistic regression weights. 

![Activations after logistic](https://i.imgur.com/KTbKczx.png)


Next we can visualize this medoids as text sequences of the users they represent. 

This takes quite a lot of space, so you could access it in file ``prototypes.txt``. Here we will only list some of them:

Churned:

``Add Vehicle Success -> Add Payment Method Success -> Subscription Premium -> Chat Conversation Opened -> Chat Conversation Started -> Subscription Premium Cancel -> Transaction Refund``

``Email Confirmation Success -> Add Payment Method Success -> Subscription Premium -> Add Vehicle Break -> Chat Conversation Started -> Transaction Refund -> Subscription Premium Cancel -> Sign Out``

``Email Confirmation Success -> Add Vehicle Success -> Add Payment Method Success -> Subscription Premium -> Account History Transaction Details -> Calculator View -> Calculator View -> Account History Transaction Details -> Chat Conversation Opened -> Chat Conversation Started -> Subscription Premium Cancel``

Unchurned:

``Add Vehicle Success -> Add Payment Method Success -> Subscription Premium``
	
``Add Vehicle Success -> Add Payment Method Success -> Subscription Premium -> Order -> Order -> Subscription Premium Renew -> Order -> Order``

``Add Vehicle Success -> Add Payment Method Success -> Subscription Premium -> Wallet Opened -> Calculator View -> Wallet Opened -> Order -> Order -> Order -> Subscription Premium Renew``

As you see, churned customers have in common transaction refunds, contacting support, breaking vehicle, viewing transaction history, estimating price of a service twice etc. Unchurned customers tend to make more orders, or have short sequences (maybe new customers). 

So, LSTM embeddings allow us not only to construct the good performing sequential model, but also view prototypes of "good" and "bad" customers and to access overall behavior of them - thing that was harder with simpler models. 

We can also look at some of model's mistakes: False Positives and False Negatives (full list is also listed in file ``prototypes.txt``).

False Positives:

``Sign Up Success -> Add Vehicle Success -> Add Payment Method Failed -> Add Payment Method Failed -> Add Payment Method Failed -> Add Payment Method Success -> Subscription Premium -> Chat Conversation Opened -> Chat Conversation Started``
``Sign Up Success -> Email Confirmation Success -> Add Vehicle Success -> Add Payment Method Success -> Subscription Premium -> Wallet Opened -> Chat Conversation Opened -> Chat Conversation Started -> Chat Conversation Opened``

We see that model thinks that contacting support can lead to further subscription cancellation. 

False Negatives:

``Sign Up Success -> Email Confirmation Success -> Account Setup Skip -> Account Setup Profile Skip -> Account Setup Profile Skip -> Account Setup Profile Skip -> Chat Conversation Opened -> Wallet Opened -> Add Payment Method Success -> Subscription Premium -> Account History Transaction Details -> Wallet Opened -> Subscription Premium Cancel``
``Sign Up Success -> Add Vehicle Success -> Add Payment Method Success -> Subscription Premium -> Account History Transaction Details -> Account History Transaction Details -> Account History Transaction Details -> Wallet Opened -> Subscription Premium Cancel -> Order -> Order -> Account History Transaction Details -> Wallet Opened``

In these examples user did not even make an order between subscribing and unsubscribing - so it is quite hard for the model to predict cancellation. 
 
 Overall, we see that RNN embeddings serve a perfect role to looking at data dimension and finding typical examples of customers that give us further inside into their actions than just analyzing features in other models. We could dive even deeper and find (using some gradient methods) which features cause the most significant change in resulting embeddings and find feature importances this way. It is certainly a subject for future work for similar tasks.
 
# Results 
 
 Overall models performances are shown in the following table. 

![](https://i.imgur.com/X0xxKg9.png)

Overall important features are:

1) ``Support chat`` - found to be important by all models in the study. Hypothesis that people who contact support might cancel subscription stays valid. Support should carefully examine their complains and try to prevent this. 

2) ``Wallet view`` - found to be important by all models in the study. People might be concerned with price of services - we noted positive correlation with churn in Random Forest model. 

3) ``Actions`` - two models notes this as top-1 and top-2 most important feature. There is positive correlation with churn - if user spends more time on actions then on orders he is probably dissatisfied with product and want to cancel subscription.

4) ``Orders`` shows negative correlation with churn. Users who use product often are satisfied and not prune to churning at most. 

5) ``Total time``, ``Average time`` - data analysis showed us that customers who make lot's of orders and use product more often tend to remain  subscribed. The subscription cancellation occurs on new customers who have not tried product for a long time yet. We conclude that the company succeeds for keeping long-term customers.

Unimportant features are all demographic information - such as ``phone model``, ``user state`` and their actions are much more representative for churn prediction than these factors. Also ``setup details`` are unimportant for predictions.

Out last (RNN) model found the typical customers prototypes that could be analyzed further by industry experts to gain more information about churn. for instance conducting surveys of churned customers etc. 

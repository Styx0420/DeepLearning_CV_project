# Report: Predict Bike Sharing Demand with AutoGluon Solution
Yixun Zhou

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
Kaggle will reject submissions if we have any negative prediction result. I have to set all negative results to 0. 

### What was the top ranked model that performed?
WeightedEnsemble_L3 was the top ranked model. This makes sense as this model integrated other models' results and weighted for the most possible prediction.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
Many feature values(such as season,holiday...) are categorical inputs where we need to mark them and use OneHotEncoder for proper modelling.
Seperated 'hour' from datetime column as 'hour' can be a reasonable variable for predicting 'count'. Here datetime column inputs are of object type and to_datetime function is applied before seperating datetime details.

### How much better did your model preform after adding additional features and why do you think that is?
The root mean squared error dropped from 53.14 to 30.40(drop by 42.8%) compared with initial model, which suggested an improved performance of our model. The reason is the categorical variables are correctly specified after adding additional features, giving model fitting a reasonable higher accuracy.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
My model kaggle score has increased from 0.65136 to 0.69657(improved by 6.94%) compared with add-feature model. Four hyper parameter kwargs are set and the best practice is when searcher= 'grid', num_trials = 50 and num_folds = 10. 

### If you were given more time with this dataset, where do you think you would spend more time?
I shall spend more time on learning hyper parameters mean in different models, and in what way can I finetune the parameters efficiently for individual models such as RandomForest. This avoid random guessing and practice of model parameter tunning, which is likely to leads to trivial performance.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|searcher|num_trials|num_folds|score|
|-----|--------------|---------------|---------------|-----|
|initial|default|default|default|1.80676 |
|add_features|default|default|default|0.65136|
|hpo1|bayesopt|20|5|0.53086|
|hpo2|grid|20|5|0.64207|
|hpo3|skopt|20|5|0.54356|
|hpo4|grid|50|10|0.69657|

### Create a line plot showing the top model score for the three (or more) training runs during the project.

![model_test_score.png](img/model_eval_matrix.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.

![model_test_score.png](img/model_test_score.png)

## Summary
This project is to predict bike sharing demand by AutoGluon from both categorical variables like season and numerical variables like humidity. The hpo4(model tuned by hyperparameter kwargs where searcher = 'grid', num_trials = 50 and num_folds = 10) is proved to be the best model for demand prediction. 

Performing EDA at the start allows us to add informative features and perform One-Hot encoding to categorical feature variables. The kaggle score after adding features and OneHotEncoding dropped from 1.84676 to 0.65136. By contrast, the matrix measurement of initial and add-feature model are 53.137261 and 30.392739 respectively, showing a decrease of root mean squared error by 42.8%. This matrix score clearly indicates a direct improvement on model predictability. 

Hyper parameters were then modified for exploring model outputs. An initial try was to modify Hyper parameters in CatBoost, RandomForest and ExtraTreeMSE models. I chosed those three models at the top of leaderboard for hyper parameter tuning and model performance were changed accordingly. However, the tuning result was poorer than default model output. It suggests that random trys of individual ML model parameters could bring in trivial or poor results if exploratory time is insufficient. As a result, I turned to hyperparameter_tune_kwargs instead of hyperparameters, enabling automated hyperparameter tuning by modeifying searcher,num_trials, and num_folds. A 'grid' searcher with 50 trials and 10 folds yields a kaggle score of 0.69657, showing an improvement of performance by 6.94%. 

A future direction could be getting more exploration on hyper parameters for AutoGluon models to further improve model score. Another direction is applying Neural Networks which might yield a better performance.
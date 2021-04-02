# CS590 Data Science Sp21
## In-class Competition: Stress Prediction

#### Author: Jun Zhuang


### 0. About The Project
In this competition, I employ several machine learning models to predict the stress level based on the survey data. The prediction is evaluated by RMSE (The smaller, the better).


### 1. Dataset
The dataset comes from Kaggle competition: <https://www.kaggle.com/c/stress-prediction/data>


### 2. Preprocessing
* **Features Selection**  
At first, I employ the backward/forward selection with a given threshold (**ts**) that filters out the columns whose non-null value is lower than the given threshold. The result shows that using **ts=0.86** can maintain most features (filter one out) and achieves the best score in the leaderboard.  
I also implement **Sparse Regression** to find out the top 10 important features by using glmnet. These features are *'hisp', 'hincome', 'fam_exp1_cv', 'fam_actions_cv_10', 'fam_discord_cv', 'child_social_media_time_cv', 'physical_activities_hr_cv', 'sitting_weekday_hour_cv', 'SEX_F', 'SEX_M'*. The result shows that the family discord caused by COVID-19, the time that child spends on social media per week, and female has a strong positive relationship with the stress, whereas the physical activities per week and house income are negative to the stress. I'm wondering whether or not female is easier to feel stressed when they realize that their children spend so much time on social media (father doesn't seem to care about this). On the other hand, physical activities help release stress. According to this information, I select these ten features to predict but unfortunately got a bad RMSE (**2.8197**) in local data. Thus, I decide not to submit this prediction.

* **Fill the Null Values**  
I fill the null values of selected columns with three methods: **mean, mode, and class-mean**. For the first two methods, I fill the null values by the mean/mode value of the current column. For the third method, I find out the mean value of each class and then fill the null values with their corresponding mean. The result shows that the third method suffers serious overfitting problems. The first method achieves the best performance and thus I use this method in the competition.

* **Implement One-hot Embedding**  
I implement one-hot embedding on *'SEX' and 'higheduc'* as these two features are categorical strings. 

* **Drop Unnecessary Columns**  
After that, I drop some unnecessary columns such as *'ID', 'interview_date', 'pstr', 'train_id' or 'test_id'*.

* **Normalization**  
Lastly, I normalize all values within [0, 1]. I repeat the above-mentioned preprocessing procedure for both train data and test data.


### 3. Methods
In this competition, I attempt several base models as follows:  
1. Generalized linear model (glm);
2. Generalized boosted regression model (gbm);
3. Random forest (rf);
4. Support vector regression (svr);
5. XGBoost (xgb);
6. Ligthgbm (lgb);
7. Neural Networks (cnn).


* At first, I try glm, gbm, rf, and svr. For svr, I try the kernel with 'linear', 'poly', and 'rbf'. The '**poly**' kernel has better performance so that I use this kernel for svr in this competition. The result shows that the gradient boosting method (**gbm**) achieved better performance (lower RMSE). Thus, I decided to go in this direction and try xgb and lgb.  
* For xgb and lgb, I apply **grid search** to find out the optimal parameters (presented in the script). The submission reveals that using lgb with **85 features** (fill nan by mean) achieves the best performance (**RMSE=2.74570**) so far.  
* In the next step, I try two ensemble methods to boost the performance. At first, I use "**stacking**" method by stacking the top-3-performance model's predicted labels (lgb, svr, xgb) and then apply linear regression to generate the final prediction. This result (**RMSE=2.76409**) is worse than the previous one.  
* Furthermore, I also propose a new ensemble method, **iterative averaging (IA)**, to achieve better performance in the public leaderboard. This method averages the top-n predicted labels at first and then iteratively replaces the worst one if the new generated label achieves better performance. The process repeats until the generated label converges. **The intuition is that the prediction from different base models may be close to the optimum. Iterative averaging of the top-n predictions helps approach the optimum better.** In this competition, I use the **top 3** predictions in this method and achieve the best ***RMSE=2.74244*** in the public leaderboard. The pseudocode is presended below:  
```python
def iterative_averaging(Top_n_pred, Y_gt):
    """
    @topic: Ensemble method: iterative averaging (IA).
    @input: 
        Top_n_pred (mxn): the list of predicted labels;
        Y_gt (mx1): the ground truth label.
    @return:
        Y_pred_mean (mx1): the new predicted label.
    """
    Y_pred_mean = mean(Top_n_pred) # Averages the top-n predicted labels
    while gap > 1e-4:
        # If the new generated label achieves better performance
        if RMSE(Y_pred_mean) < RMSE(Top_n_pred).any(): 
            Top_n_pred.replace(Y_pred_mean) # Replaces the worst one
            Y_pred_mean_new = mean(Top_n_pred) # New prediction
            gap = RMSE(Y_pred_mean_new) - RMSE(Y_pred_mean)
            Y_pred_mean = Y_pred_mean_new # Update the prediction
     return Y_pred_mean
```

* Besides, I also employ cnn in this task. The RMSE without dropout is **12.4962** on the test set whereas using dropout=0.3 improves the RMSE to **2.9544**. Note that both results are good on the train/validation set but the RMSE gets worse on the test without dropout. This result indicates using dropout can largely mitigate the overfitting. However, RMSE=2.9544 is not good so that I decide not to consider CNN.


### 4. Conclusion
In this competition, I implement Sparse Regression to find out the important features related to the stress level and investigate several base machine learning models. Moreover, I propose a new ensemble method, **iterative averaging (IA)**, to achieve better performance in the public leaderboard.


### Acknowledgments
 Thanks professor Mohler for his insightful suggestions.

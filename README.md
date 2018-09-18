# Supervised Learning
## Project: Finding Donors for CharityML
### Files Description
- `project description.md`: Project overview, highlights, evaluation and software requirement. **Read Firstly**
- `README.md`: this file.
- `finding_donors.ipynb`: This is the main file where I will be performing your work on the project.
- `census.csv`: The project dataset. I'll load this data in the notebook.
- `visuals.py`: A Python file containing visualization code that is run behind-the-scenes. **Not my work**
- `finding_donors.html`: `html` version of the main file.

### Run
#### 1.Want Modify
In a command window (OS: Win7), navigate to the top-level project directory that contains this README and run one of the following commands:
`jupyter notebook finding_donors.ipynb`
This will open the Jupyter Notebook software and project file in your browser.
#### 2.Just Have a Look
Double click `finding_donors.html` file. You can see this file in your browser.

## Project Implementation
### Exploring the Data
#### Implementation: Data Exploration
Correctly calculate the following:
Number of records
Number of individuals with income >$50,000
Number of individuals with income <=$50,000
Percentage of individuals with income > $50,000
```
# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[data.income=='>50K'].shape[0]

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data[data.income=='<=50K'].shape[0]

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k * 100. / n_records
```
### Preparing the Data
#### Data Preprocessing
Correctly implement one-hot encoding for the feature and income data.
Use `pandas.get_dummies()` to perform one-hot encoding on the `'features_raw'` data.

Convert the target label `'income_raw'` to numerical entries.
Set records with "<=50K" to 0 and records with ">50K" to 1.
```
# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
features = pd.get_dummies(features_raw)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)
```
### Evaluating Model Performance
#### Naive Predictor Performace
Correctly calculate the benchmark score of the naive predictor for both accuracy and F1 scores.
```
# TODO: Calculate accuracy
accuracy = greater_percent

# TODO: Calculate F-score using the formula above for beta = 0.5
TP = n_greater_50k
FP = n_at_most_50k
FN = 0
TN = 0
beta = 0.5
precision = TP*1. / (TP + FP)
recall = TP*1. / (TP + FN)
fscore = (1 + beta**2)*precision*recall/(beta**2 * precision+recall)
```
#### Model Application
List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
- Describe one real-world application in industry where the model can be applied. (You may need to do research for this — give references!)
- What are the strengths of the model; when does it perform well?
- What are the weaknesses of the model; when does it perform poorly?
- What makes this model a good candidate for the problem, given what you know about the data?

```
Decision Trees
    [A decision tree that determines whether or not to offer a credit card invitation](http://booksite.elsevier.com/9780124438804/leondes_expert_vol1_ch3.pdf)
    pros:
        Applicable for continuous and categorical inputs
        Data classification without much calculations
        Can generate rules helping experts to formalize their knowledge.
    cons:
        all terms are assumed to interact, you can't have two explanatory variables that behave independently
        prone to overfitting
    I choose this model not only because of pros but also its fast, mitigating its overfitting by setting optimal minimum number of samples
K-Nearest Neighbors (KNeighbors)
    [Predicting Economic Events](http://www.ijera.com/papers/Vol3_issue5/DI35605610.pdf)
    pros:
        Simple and powerful. No need for tuning complex parameters to build a model.
        No training involved (“lazy”). New training examples can be added easily.
    cons:
        Expensive and slow: O(md), m= # examples, d= # dimensions
    I choose this model not only because of pros but also do not need to make any assumptions on the underlying data distribution, its computational costs don't bother me
Support Vector Machines (SVM)
    [applications for quality assessment in manufacturing industry](http://www.metrology-journal.org/articles/ijmqe/pdf/2015/04/ijmqe150023-s.pdf)
    pros:
        Performs similarly to logistic regression when linear separation
        Performs well with non-linear boundary depending on the kernel used
        Handle high dimensional data well
    cons:
        Inefficient to train
        Picking/finding the right kernel can be a challenge
    I choose this model because SVM can capture much more complex relationships between your datapoints without having to perform difficult transformations on my own, data set provided is not so large, so I can have a try
```
#### Creating a Training and Predicting Pipeline
Successfully implement a pipeline in code that will train and predict on the supervised learning algorithm given.
- Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
- Fit the learner to the sampled training data and record the training time.
- Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
    - Record the total prediction time.
- Calculate the accuracy score for both the training subset and testing set.
- Calculate the F-score for both the training subset and testing set.
    - Make sure that you set the `beta` parameter!
```
# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    start = time() # Get start time
    learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)
        
    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[:300],predictions_train, average='binary', beta=0.5)
        
    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test, average='binary', beta=0.5)
       
    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)
        
    # Return the results
    return results
```
#### Implementation: Initial Model Evaluation
Correctly implement three supervised learning models and produces a performance visualization.
- Import the three supervised learning models you've discussed in the previous section.
- Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
    - Use a `'random_state'` for each model you use, if provided.
    - Note: Use the default settings for each model — you will tune one specific model in a later section.
- Calculate the number of records equal to 1%, 10%, and 100% of the training data.
    - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.
```
# TODO: Import the three supervised learning models from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# TODO: Initialize the three models
clf_A = DecisionTreeClassifier(random_state=0)
clf_B = KNeighborsClassifier()
clf_C = SVC(random_state=0)

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(len(X_train) * 0.01)
samples_10 = int(len(X_train) * 0.1)
samples_100 = int(len(X_train))
```
### Improving Results
#### Choosing the Best Model
Justification is provided for which model appears to be the best to use given computational cost, model performance, and the characteristics of the data.

#### Describing the Model in Layman's Terms
Clearly and concisely describe how the optimal model works in layman's terms to someone who is not familiar with machine learning nor has a technical background.
#### Implementation: Model Tuning

The final model chosen is correctly tuned using grid search with at least one parameter using at least three settings. If the model does not need any parameter tuning it is explicitly stated with reasonable justification.
- Import `sklearn.grid_search.GridSearchCV` and `sklearn.metrics.make_scorer`.
- Initialize the classifier you've chosen and store it in `clf`.
    - Set a `random_state` if one is available to the same state you set before.
- Create a dictionary of parameters you wish to tune for the chosen model.
    - Example: `parameters = {'parameter' : [list of values]}`.
    - Note: Avoid tuning the `max_features` parameter of your learner if that parameter is available!
- Use `make_scorer` to create an `fbeta_score` scoring object (with  β=0.5).
- Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
- Fit the grid search object to the training data (`X_train, y_train`), and store it in `grid_fit`.

```
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# TODO: Initialize the classifier
clf = KNeighborsClassifier()

# TODO: Create the parameters list you wish to tune
parameters = {'n_neighbors':[3, 7, 9]}  

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
grid_obj = GridSearchCV(clf, parameters,scoring=scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters

grid_fit = grid_obj.fit(X_train, y_train)

```

#### Final Model Evaluation

- Report the accuracy and F1 score of the optimized, unoptimized, and benchmark models correctly in the table provided. 
- Compare the final model results to previous results obtained.

### Feature Importance
#### Feature Relevance Observation
Rank five features which they believe to be the most relevant for predicting an individual's’ income. Discussion is provided for why these features were chosen.
- Import a supervised learning model from sklearn if it is different from the three used earlier.
- Train the supervised model on the entire training set.
- Extract the feature importances using `'.feature_importances_'`.
```
# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import RandomForestClassifier

# TODO: Train the supervised model on the training set 
model = RandomForestClassifier()
model.fit(X_train, y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_
```
#### Extracting Feature Importances
Correctly implement a supervised learning model that makes use of the `feature_importances_` attribute. 
Discuss the differences or similarities between the features they considered relevant and the reported relevant features.

#### Effects of Feature Selection
Analyze the final model's performance when only the top 5 features are used and compares this performance to the optimized model

How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?

If training time was a factor, would you consider using the reduced data as your training set?
```
Final Model trained on full data
    Accuracy on testing data: 0.8299
    F-score on testing data: 0.6540
Final Model trained on reduced data
    Accuracy on testing data: 0.8289
    F-score on testing data: 0.6522
Scores almost the same! Other features play a less importmant role on the predict, from the visualization above, the top five most important features contribute nearly 0.6 of the importance of all features present in the data.
If training time was a factor, I would consider using the reduced data as my training set, because of the almost same result.
```





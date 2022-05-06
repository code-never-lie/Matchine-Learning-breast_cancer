from collections import Counter
import pandas as pd
import numpy as np
import seaborn as sns
import random

from sklearn.model_selection import train_test_split
# Michelle Tatara
class Node:

    '''
    Node class holds the information for the data found in the decision tree. For example,
    `right` is used for right nodes and `left` is used for left nodes
    '''
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        return self.value is not None
    

class DecisionTreeModel:
    '''
    This class would build a decision tree model.
    ## Input:
    criterion is the criteria that measures the information gain.
    min_samples_split the amount of splits we have to make in our decision tree.
    max_depth measures the highest the tree will reach
    '''
    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # TODO
        # call the _fit method
        # this will convert the y to a numpy array later.
        self.fit(X.to_numpy(),y)
        # end TODO
        print("Done fitting")

    def predict(self, X: pd.DataFrame):
        '''
        accepts an x value in the dataframe and gives back an array with predictions for each point.
        ## Input:
        **Dataframe** type. `X`
        :return: np.ndarray will contain all the predictions
        The following: return self.predict(X.to_numpy())
        '''
        # TODO
        # end TODO
        
    def _fit(X: np.ndarray, y:pd.Series,self):
        # checks to see that if there are any non integer values in the set.
        y = self._change_if_categorical(y).to_numpy()
        # this converts it to a numpy array
        self.root = self._build_tree(X, y)
        
    def _predict(X: np.ndarray, self):
        # climbs through the tree and returns all the values into an array
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth):
        # TODO: for graduate students only, add another stopping criteria
        # modify the signature of the method if needed
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self):
        # TODO: for graduate students only
        result = False
        # end TODO
        return result
                              
    def _build_tree(X: np.ndarray, y: pd.Series, self, depth=0):
        '''
        Using the input criteria, a decision tree is crafted by splitting the date based on
        where the prime information gain is
        '''
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)
    
    def _change_if_categorical(y:pd.Series,self):
        '''
        Obtains all `y` labels and goes through them via loop and assigns Ordinal Encoding
        from 0-n where `n` represents the length of each unique label
        '''
        if self._is_categorical(y):
            # If the values are categorical, obtain the unique classes of that feature
            # Then you have to go through each value and check it using a for loop and len
            # If the current class label applies, we return the index of this label as a number using .apply

    def _is_categorical(y: pd.Series, self) -> bool:
        '''
        We assume that values found that are in str aren't ints and are thus deemed categorical.
        goes through `y` and if any value is found that isn't an integer, the value of true would be returned
        which signifies the set to be categorical otherwise the value of false will be given back which means
        everything is an integer.
        '''
        for _, val in y.iteritems():
            if type(val) != int:
                return True
            return False

    def _gini(y:np.ndarray, self):
        # TODO
        """
        Due to Ordinal Encoding, the values inside have already been altered to numerical if it was categorical
        """
        # Due to _build_tree(), the y's are changed to the following
        proportions = np.bincount(y) / len(y)
        gini = np.sum([p * (1 - p) for p in proportions if p > 0])
        # return 1 - (a1 + b1)
        # end TODO
        return gini
    
    def _entropy(y: pd.Series, self):
        # the y will always have a number attached to it because it was altered in the _build_tree()
        '''
        finds the entropy of the branch. If the set given is categorical, the values
        will be converted to numerical ones from 0-x where `x` is the number of classes of that section
        '''

        # TODO: the following won't work if y is not integer
        # make it work for the cases where y is a categorical variable
        proportions = np.bincount(y) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        # end TODO
        return entropy
        
    def _create_split(self, X: np.ndarray, thresh):
        # when x is >= the left's capacity and the rest goes to the right
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def gini_or_entropy(y: pd.Series, self):
        '''
        based on the criteria given, either `self._gini(y)` or `self._entropy(y) will be returned
        '''
        return self._entropy(y) if self.criterion == 'entropy' else self._gini(y)

    def _information_gain(self, X, y, thresh):
        # TODO: fix the code so it can switch between the two criterion: gini and entropy 
        # Either entropy of gini based criteria will be derived from the criteria
        parent_loss = self._entropy(y)
        # All the indices that would be on the right and left side of the tree will be extrapolated
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0: 
            return 0
        
        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
        # end TODO
        return parent_loss - child_loss
       
    def _best_split(self, X, y, features):
        '''TODO: add comments here
        ##Returns: both threshold and feature are split as tuples
        All the features of the tree are looped through. The column of each unique feature is collected and placed
        into an array with all the distinct values found in the column. After going through each unique value, the
        information gain score will be calculated. We see how the gain score would be affected if the data was split
        at that point and if the `score` was greater than the previous value, it would be stored.
        '''
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        '''TODO: add some comments here
        Starting at the root node, the tree will be traversed. Next, we compare the current value of x to the
        threshold to see if the values are less than or equal to each other. If that is the case, it will go to
        the left node. If that isn't the case, we go to to the right node. This comparison will happen until we
        stumble upon a leaf node and then the value of that node will be returned
        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(n_estimators: int, self, min_samples_split=2, impurity_stopping_threshold = 1, criterion='gini',
                  max_depth=100):
        '''
        The required parameters from the class are passed in and an array of Decision trees
        are passed in as well. For the amount of trees, `n_estimators` are created. After each tree
        is *fitted* the data given to it is recompiled.
        ## Inputs:
        `criteria` = `gini` this is the metric used to split the data
        `n_estimators`: int - The number of trees that need to be created.
        `max_depth` = 100- the limit to how deep the tree can get
        '''
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.n_estimators = n_estimators
        # TODO:
        # trees contained in the forest
        self.forest = []
        # end TODO

    def fit(self, X: pd.DataFrame, y: pd.Series):
        for i in range(self.n_estimators, 0):
           # randomises the data
           idxs = np.random.choice(replace=True,size=len(y),a=len(y))
           # makes a tree
           tree = DecisionTreeModel(criterion=self.criterion, max_depth=self.max_depth,
                                    impurity_stopping_threshold=self.impurity_stopping_threshold,
                                    min_samples_split=self.min_samples_split)
           # pass in the randomised indexes and fit the tree into that.
           tree.fit(X.iloc[idxs], y.iloc[idxs])
           #append the current tree to our array which represents an array of forests
           self.forest.append(tree)

    def _commonalities(values:list, self):
        '''
        Each column of values is observed. Each value that is predicted represents a possible tree in the forest. The
        most common result is chosen which also represents the most common prediction. This array of predictions is a
        2D array
        '''
        return np.array([Counter(col).most_common(1)[0][0] for col in zip(*values)])

    def predictions(self, X: pd.DataFrame):
        # TODO:
        '''
        The result is formed from the 2D array where each value was a possible guess of a tree.Each tree in the forest
        will get the `predict(X)` method called on them. The results are placed into its own array to another array.
        '''
        values_in_tree = []
        for tree in self.forest:
           values_in_tree.append(tree.predict(X))
        return self._common_result(values_in_tree)

def accuracy_score(y_true, y_pred):
    '''Using the `y_pred` values the accuracy score is calculated.'''
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def _num_of_occurrences(y_pred):
    '''The number of times `0` and `1` appear in the array of predictions are counted and returned as a split list'''
    one,zero=0,0
    for i in y_pred:
        if i == 1:
            one += 1
        else:
            zero += 1
    return zero, one

def classification_report(y_test, y_pred):
    # calculate weighted avg,macro avg, precision, recall, f1-score, accuracy,support
    # TODO:
    bottom, top = confusion_matrix(y_test, y_pred)
    tp, fn = top
    fp, tn = bottom

    recall1 = (tp / (tp + fn))
    recall0 = (tn / (tn + fp))
    precision1 = (tp / (tp + fp))
    precision0 = (tn / (tn + fn))
    support0, support1 = _num_of_occurrences(y_pred)
    f1_score1 = 2 * (recall1 * precision1) / (recall1 + precision1)
    f1_score0 = 2 * (recall0 * precision0) / (recall0 + precision0)

    end_result = f'''                    precision    recall    f1-score    support
    0               {"%.2f" % round(precision0, 2)}        {"%.2f" % round(recall0, 2)}        {"%.2f" % round(f1_score0, 2)}        {support0}
    1               {"%.2f" % round(precision1, 2)}        {"%.2f" % round(recall1, 2)}        {"%.2f" % round(f1_score1, 2)}        {support1}
    accuracy                                {"%.2f" % round(accuracy_score(y_test, y_pred), 2)}        {support0 + support1}
    macro avg       {"%.2f" % round((precision0 + precision1) / 2, 2)}        {"%.2f" % round((recall0 + recall1) / 2, 2)}        {"%.2f" % round((f1_score0 + f1_score1) / 2, 2)}        {support0 + support1}
    weighted avg    {"%.2f" % round(((precision0 * support0) + (precision1 * support1)) / (support0 + support1), 2)}        {"%.2f" % round(((recall0 * support0) + (recall1 * support1)) / (support0 + support1), 2)}        {"%.2f" % round(((f1_score0 * support0) + (f1_score1 * support1)) / (support0 + support1), 2)}        {support0 + support1}
    '''
    return end_result

def confusion_matrix(y_test, y_pred):
    '''
    return the 2x2 matrix that will show the true negative, true positive, false negative, false positive
    ## returns a 2d array with the structure of a confusion matrix
    '''
    # return the 2x2 matrix
    tp, fn, fp, tn = 0, 0, 0, 0
    # goes through all the predictions
    for i, val in y_test.reset_index(drop=True).iteritems():
        p_val = y_pred[i]
        # At this index, y_test is compared to the current value given by the prediction
        if val == 1 and p_val == 1:
            tp += 1
        elif val == 1 and p_val == 0:
            fn += 1
        elif val == 0 and p_val == 1:
            fp += 1
        else:
            tn += 1
    result = np.array([[tp, fn], [fp, tn]])
    return(result)

def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Accuracy:", acc)

if __name__ == "__main__":
    _test()

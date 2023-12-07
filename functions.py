import inspect
import itertools
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import seaborn as sns
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, confusion_matrix, auc, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score, roc_curve, recall_score
from sklearn.model_selection import learning_curve, cross_val_score, KFold, train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_curve,
    auc,
)

########### Model Explanation ###########
## Plotting AUC ROC curve
def plot_roc(y_actual, y_pred):
    """
    Function to plot AUC-ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    plt.plot(
        fpr,
        tpr,
        color="b",
        label=r"Model (AUC = %0.2f)" % (roc_auc_score(y_actual, y_pred)),
        lw=2,
        alpha=0.8,
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        lw=2,
        color="r",
        label="Luck (AUC = 0.5)",
        alpha=0.8,
    )
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.show()


def plot_precisionrecall(y_actual, y_pred):
    """
    Function to plot AUC-ROC curve
    """
    average_precision = average_precision_score(y_actual, y_pred)
    precision, recall, _ = precision_recall_curve(y_actual, y_pred)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = (
        {"step": "post"} if "step" in inspect.signature(plt.fill_between).parameters else {}
    )

    plt.figure(figsize=(9, 6))
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title("Precision-Recall curve: AP={0:0.2f}".format(average_precision))


## Plotting confusion matrix
def plot_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=False,
    title="Confusion matrix",
    cmap=plt.cm.Blues,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


## Variable Importance plot
def feature_importance(model, X):
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel("Relative Importance")
    plt.title("Variable Importance")
    plt.show()
    


def print_classification_performance2class_report(model,X_test,y_test):
    """ 
        Program: print_classification_performance2class_report
        Author: Siraprapa W.
        
        Purpose: print standard 2-class classification metrics report
    """
    import seaborn as sns
    
    sns.set()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]
    conf_mat = confusion_matrix(y_test,y_pred)
    TN, FP, FN, TP = conf_mat.ravel()
    PC = precision_score(y_test, y_pred, zero_division=0)
    RC = recall_score(y_test, y_pred, zero_division=0)
    specificity = TN / (TN + FP) if TN + FP > 0 else 0
    FS = f1_score(y_test, y_pred)
    AP = average_precision_score(y_test,y_pred)
    ACC = accuracy_score(y_test,y_pred)
    gmean = np.sqrt(RC * specificity) if RC * specificity >= 0 else 0
    
    # Calculate ROC curve and AUC
    pfr, tpr, _ = roc_curve(y_test,y_pred_proba)
    roc_auc = auc(pfr, tpr)
    
    print("Accuracy: {:.2%}".format(ACC))
    print("Precision: {:.2%}".format(PC))
    print("Sensitivity (Recall): {:.2%}".format(RC))
    print("Specificity: {:.2%}".format(specificity))
    print("Fscore: {:.2%}".format(FS))
    print("Average precision: {:.2%}".format(AP))
    print("G-Mean: {:.2%}".format(gmean))
    print("ROC AUC: {:.2%}".format(roc_auc))

    
    fig = plt.figure(figsize=(20,3))
    fig.subplots_adjust(hspace=0.2,wspace=0.2)
    
    #heatmap
    plt.subplot(141)
    labels = np.asarray([['True Negative\n{}'.format(TN),'False Positive\n{}'.format(FP)],
                         ['False Negative\n{}'.format(FN),'True Positive\n{}'.format(TP)]])
    sns.heatmap(conf_mat,annot=labels,fmt="",cmap=plt.cm.Blues,xticklabels="",yticklabels="",cbar=False)
    
    #ROC
    plt.subplot(142)
    # pfr, tpr, _ = roc_curve(y_test,y_pred_proba)
    # roc_auc = auc(pfr, tpr)
    gini = (roc_auc*2)-1
    plt.plot(pfr, tpr, label='ROC Curve (area =  {:.2%})'.format(roc_auc) )
    plt.plot([0,1], [0,1])
    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Charecteristic Curve with Gini {:.2}'.format(gini))
    plt.legend(loc='lower right')
    
    #pr
    plt.subplot(143)
    precision, recall, _ = precision_recall_curve(y_test,y_pred_proba)
    step_kwargs = ({'step':'post'}
                  if 'step'in inspect.signature(plt.fill_between).parameters
                   else {})
    plt.step(recall,precision,color='b',alpha=0.2, where='post')
    plt.fill_between(recall,precision,alpha=0.2,color='b',**step_kwargs)
    plt.ylim([0.0,1.05])
    plt.xlim([0.0,1.0])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.title('2-class Precision-Recall Curve: AP={:.2%}'.format(AP))
    
    #hist
    plt.subplot(144)
    tmp = pd.DataFrame(data=[y_test,y_pred_proba]).transpose()
    tmp.columns=['class','proba']
    mask_c0 = tmp['class']==0
    mask_c1 = tmp['class']==1
    plt.hist(tmp.loc[mask_c0,'proba'].dropna(),density=True,alpha=0.5,label='0',bins=20)
    plt.hist(tmp.loc[mask_c1,'proba'].dropna(),density=True,alpha=0.5,label='1',bins=20)
    plt.ylabel('Density')
    plt.xlabel('Probability')
    plt.title('2-class Distribution' )
    plt.legend(loc='upper right')
    
    plt.show()
    
    return y_pred,ACC,PC,RC,FS,AP,roc_auc,gini



#if there is imbalance, you can handle it by over-sampling or under-sampling the dataset

def handle_imbalanced_data(X, y, strategy='over-sampling'):
    """
    Handle imbalanced data using imblearn library.
    
    Parameters:
    -----------
    X: array-like of shape (n_samples, n_features)
        The input data.
    y: array-like of shape (n_samples,)
        The target values.
    strategy: str, default='over-sampling'
        The strategy to use for handling imbalanced data. Possible values are
        'over-sampling' and 'under-sampling'.
        
    Returns:
    --------
    X_resampled: array-like of shape (n_samples_new, n_features)
        The resampled input data.
    y_resampled: array-like of shape (n_samples_new,)
        The resampled target values.
    """
    if strategy == 'over-sampling':
        # Initialize the RandomOverSampler object
        ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
        # Resample the data
        X_resampled, y_resampled = ros.fit_resample(X, y)
    elif strategy == 'under-sampling':
        # Initialize the RandomUnderSampler object
        rus = RandomUnderSampler(sampling_strategy='majority', random_state=0)
        # Resample the data
        X_resampled, y_resampled = rus.fit_resample(X, y)
    else:
        raise ValueError("Invalid strategy. Possible values are 'over-sampling' and 'under-sampling'.")
    
    return X_resampled, y_resampled


def cv_learning_curve(model, X, y, cv, train_sizes):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy')
                                                #scoring parameter -  #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter 
    train_mean = np.mean(-train_scores, axis=1)
    train_std = np.std(-train_scores, axis=1)
    test_mean = np.mean(-test_scores, axis=1)
    test_std = np.std(-test_scores, axis=1)
    
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Error')
    plt.plot(train_sizes, test_mean, 'o-', color='green', label='Validation Error')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.show()
    
    return train_sizes, train_mean, train_std, test_mean, test_std


import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_model_performance(model, X, y):
    # Generate predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    sensitivity = recall_score(y, y_pred)  # Also known as recall
    specificity = tn / (tn + fp)
    gmean = np.sqrt(sensitivity * specificity)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"G-Mean: {gmean:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


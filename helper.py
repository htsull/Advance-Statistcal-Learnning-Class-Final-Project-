from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import normalize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

def model_accuracy(model):
    """Perform the fitting, print out the accuracy, classification report and confusion matrix with the given model

    Args:
        model(string) : model to perform. 'rf' : random forest, 'xgb' : XGBoost, 'knn' : K-Nearest Neighbors.

    Returns:
        classifier: model fit to the train set.
    """
    if model == 'rf':
        print("Random forest model")
        # fitting 
        model_rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
        # Predictions (test)
        predictions_test = model_rf.predict(X_test)
        model_rf_score = model_rf.score(X_test, y_test)
        print('Base model accuracy  : {:04.3f}'.format(model_rf_score))
        print("=========================================================================")
        print("Classification report : ")
        print(classification_report(y_test, predictions_test, digits=3, zero_division = 1)),
        disp = plot_confusion_matrix(model_rf, X_test, y_test, cmap='Reds', values_format='d',
                                    #  display_labels=labels_disp,
                                    xticks_rotation= 'vertical')
        print("=========================================================================")
        print("Confusion matrix : ")
        return model_rf
    
    elif model == 'xgb':
        print("Extreme gradient boosting model")
        # fitting 
        model_xgb = GradientBoostingClassifier().fit(X_train, y_train)
        print("=========================================================================")
        model_xgb_score = model_xgb.score(X_test, y_test)
        print("Fitting to the train set : {:04.2f}".format(model_xgb_score))
        # Predictions (test)
        print("=========================================================================")
        predictions_test = model_xgb.predict(X_test)
        print('Base model accuracy  : {:04.3f}'.format(model_xgb.score(X_test, y_test)))
        print("=========================================================================")
        print("Classification report : ")
        print(classification_report(y_test, predictions_test, digits=3, zero_division = 1)),
        disp = plot_confusion_matrix(model_xgb, X_test, y_test, cmap='Reds', values_format='d',
                                    #  display_labels=labels_disp,
                                    xticks_rotation= 'vertical')
        print("=========================================================================")
        print("Confusion matrix : ")
        return model_xgb
    
    elif model == 'knn':
        print("K-Nearest Neighbors model")
        # fitting 
        model_knn = KNeighborsClassifier().fit(X_train, y_train)
        predictions_test = model_knn.predict(X_test)
        model_knn_score = model_knn.score(X_test, y_test)
        print("Fitting to the train set : {:04.2f}".format(model_knn_score))
        # Predictions (test)
        print("=========================================================================")
        print('Base model accuracy  : {:04.3f}'.format(model_knn_score))
        print("=========================================================================")
        print("Classification report : ")
        print(classification_report(y_test, predictions_test, digits=3, zero_division = 1)),
        disp = plot_confusion_matrix(model_knn, X_test, y_test, cmap='Reds', values_format='d',
                                    #  display_labels=labels_disp,
                                    xticks_rotation= 'vertical')
        print("=========================================================================")
        print("Confusion matrix : ")
        return model_knn
    
    elif model == 'lda':
        print("Linear Discriminant Ananlysis model (LDA)")
        # fitting 
        model_lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
        predictions_test = model_lda.predict(X_test)
        model_lda_score = model_lda.score(X_test, y_test)
        print('Base model accuracy  : {:04.3f}'.format(model_lda_score))
        print("=========================================================================")
        print("Classification report : ")
        print(classification_report(y_test, predictions_test, digits=3, zero_division = 1)),
        disp = plot_confusion_matrix(model_lda, X_test, y_test, cmap='Reds', values_format='d',
                                    #  display_labels=labels_disp,
                                    xticks_rotation= 'vertical');
        print("=========================================================================")
        print("Confusion matrix : ")
        return model_lda
    
    else :
        print("Enter a valid choice.")
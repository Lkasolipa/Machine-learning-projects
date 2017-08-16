#1. Clean outliers
def outliersCleaner(features, labels):
    """
    This function clean outliers from original data
    """
    # Convert to list of features + labels
    data_set = zip(features, labels)
    
    #Sort it to find outliers
    features_sorted = list(data_set)
    features_sorted.sort(key = lambda column: column[0][0])
    
    # After observation, removed 43 outliers from data
    features_sorted = list(features_sorted[:130])
    features_sorted.sort(key = lambda column: column[0][0])
    features_sorted = list(features_sorted[:120])
    features_sorted.sort(key = lambda column: column[0][2])
    features_sorted = list(features_sorted[:110])
    features_sorted.sort(key = lambda column: column[0][3])
    features_sorted = list(features_sorted[:100])
    
    #Split into original features and labels
    cleaned_features = [features[0] \
                        for features in features_sorted]
    cleaned_labels = [labels[1] \
                      for labels in features_sorted]
    
    return cleaned_features, cleaned_labels

#2. Visualize model[4]
def visualizeModel(clf, X_train, y_train,
                   X_test, y_test, model_names):
    """
    This function visualizes predicted models
    """
    #Config size of plot
    plt.figure(figsize=(7,7))

    # Plot the decision boundary. For that,
    # we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X_train[:, 0].min() - .5, \
                   X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 0].min() - .5, \
                   X_train[:, 0].max() + .5

    #Step size in the mesh
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    #Crete color for training point and test point
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    cm_bright_test =ListedColormap(['',''])

    # Predict meshg rid point to assign color
    #Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    score = clf.score(X_test, y_test)
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    #Plot the features_train point and features_test point
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.9)
    plt.scatter(X_train[:, 0], X_train[:, 1],
                c=y_train, cmap=cm_bright)
    plt.scatter(X_test[:, 0], X_test[:, 1],
                c=y_test, cmap=cm_bright, alpha=0.4)

    #Also plot the legend
    plt.scatter(0, 0, color = 'r', label='train point_[0]')
    plt.scatter(0, 0, color = 'b', label='train point_[1]')
    plt.scatter(0, 0, color = 'r', label='test point_[0]', alpha=.4)
    plt.scatter(0, 0, color = 'b', label='test point_[1]', alpha=.4)
    plt.legend()

    #Plot the accuracy for this model
    plt.text(xx.max() - .3,
             yy.min() + .3,('%.2f' % score).lstrip('0'),
             size=15, horizontalalignment='right', color='w')
    
    #Plot the name of model
    plt.title(model_names)

    #Show plot
    plt.show()

#3. Plot confusion matrix[5]
def plotConfusionMatrix(names, classifiers,
                        X_train, y_train,
                        X_test, y_test):
    """
    This function prints and plots the confusion matrix.
    """
    #Figure size of plot
    step = 1
    plt.figure(figsize=(16,16))
    
    #Plot each of sub confusion matrix
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        #Create subplot
        ax = plt.subplot(1, len(classifiers), step)
        #Force value of x and y axis to integer
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        thresh = cm.max() / 1.5
        for i, j in itertools.product(range(cm.shape[0]),
                                      range(cm.shape[1])):
            ax.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        ax.set_ylabel('True label')
        ax.set_xlabel('Predicted label')
        ax.set_title(name)
        step += 1
        
    #Show plot
    plt.tight_layout()
    plt.show()

#4. Plot Classifier Accuracy Comparison
def plotAccuaracyComparison(acc):
    sorted_acc = sorted(acc.items(), key=lambda x: x[1])
    x_pos = np.arange(len(sorted_acc))
    
    names = zip(*sorted_acc)[0]
    values = zip(*sorted_acc)[1]
    
    plt.figure(figsize=(7,7))
    bar_list = plt.bar(x_pos, values , align='center')
    plt.xticks(x_pos, names)
    plt.title('Classifier Accuracy Comparison Chart')
    bar_list[len(names) - 1].set_color('r')
    plt.show()
    
#5. Plot Classification Report
def plotClassificationReport(names, classifiers,
                             X_train, y_train,
                             X_test, y_test):
    #Figure size of plot
    step = 1
    plt.figure(figsize=(16,16))
    
    #Set information for x-axis
    axis = np.arange(3)
    axis_names = ['Precision', 'Recall', 'F1-Score']
    
    #Plot each of sub confusion matrix
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        #Caculate classification report
        cr = classification_report(y_test, pred)
        lines = cr.split('\n')
        
        #Split score from original confusion matrices
        scores_set = []
        for line in lines[2 : (len(lines) - 3)]:
            tmp = line.split()
            score = [float(item) \
                     for item in tmp[1: len(tmp) - 1]]
            scores_set.append(score)

        ax = plt.subplot(1, len(classifiers), step)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        scores_set = np.reshape(scores_set, (2, 3))

        ax.imshow(scores_set, interpolation='nearest', \
                  cmap=plt.cm.Blues)

        thresh = scores_set.max() / 2.2
        for i, j in itertools.product(range(scores_set.shape[0]),
                                      range(scores_set.shape[1])):
            ax.text(j, i, scores_set[i, j],
                    horizontalalignment="center",
                    color="white" if scores_set[i, j] > thresh else "black")
        ax.set_title(name)
        plt.xticks(axis, axis_names)
        step += 1

    #Show plot
    plt.tight_layout()
plt.show()

### Task 6: Tune classifier to achieve better than the previous task
names = ['Naive Bayes', 'SVC1-Linear',
         'SVC2-RBF', 'Decision Tree',
         'Adaboost', 'Random Forest']
classifiers = [GaussianNB(),
               SVC(kernel="linear", C=0.025),
               SVC(gamma=2, C=0.0025),
               DecisionTreeClassifier(min_samples_split=10),
               AdaBoostClassifier(n_estimators=100,
                                  learning_rate=0.025),
               RandomForestClassifier(n_estimators=10,
                                      max_features=2,
                                      min_samples_split=8)]

h = .01  # step size in the mesh

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

x_min, x_max = features_train_pca[:, 0].min() - .5, \
               features_train_pca[:, 0].max() + .5
y_min, y_max = features_train_pca[:, 1].min() - .5, \
               features_train_pca[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#Caculate accuracy for another task
acc = {}

figure = plt.figure(figsize=(20, 16))
i = 1
for name, clf in zip(names, classifiers):
    ax = plt.subplot(3, len(classifiers), i)
    clf.fit(features_train_pca, labels_train)
    score = clf.score(features_test_pca, labels_test)
    
    #Caculate accuracy for plotting in next task
    acc.update({name: score * 100})
    
    # Plot the decision boundary. For that, we will assign a color
    # to each point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(),
                                        yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(),
                                    yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot also the training points
    ax.scatter(features_train_pca[:, 0],
               features_train_pca[:, 1],
               c=labels_train, cmap=cm_bright)
    # and testing points
    ax.scatter(features_test_pca[:, 0],
               features_test_pca[:, 1],
               c=labels_test, cmap=cm_bright,
               alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(name)
    ax.text(xx.max() - .3, yy.min() + .3,
            ('%.2f' % score).lstrip('0'),
            size=25, horizontalalignment='right',
            color = 'b')
    i += 1
#Show plot
plt.tight_layout()
plt.show()

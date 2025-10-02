import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

CLASS1_PATH = r'C:\Users\Thirshith\Desktop\Dataset\n01704323'
CLASS2_PATH = r'C:\Users\Thirshith\Desktop\Dataset\n01532829'

# ====== A1: Load Dataset and Feature Extraction ======
def load_dataset(path1, path2, max_imgs=100):
    X, y = [], []
    for label, folder in enumerate([path1, path2]):
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:max_imgs]
        print(f'Loading {len(files)} images from {folder}')
        for fname in files:
            img = Image.open(os.path.join(folder, fname)).convert('L').resize((32, 32))
            X.append(np.array(img).flatten().mean())
            y.append(label)
    return np.array(X).reshape(-1, 1), np.array(y)

try:
    X, y = load_dataset(CLASS1_PATH, CLASS2_PATH)
    print('Data loaded. Total samples:', len(X))
except Exception as e:
    print('A1 ERROR: Could not load dataset. Confirm that these folders exist and contain images.')
    print('Error:', e)
    exit()

# ====== A2: Hyperparameter Tuning with RandomizedSearchCV ======
classifiers = {
    'SVC': (SVC(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [3, 5, None]}),
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [10, 50, 100]}),
    'AdaBoost': (AdaBoostClassifier(), {'n_estimators': [50, 100]}),
    'NaiveBayes': (GaussianNB(), {}),
    'MLP': (MLPClassifier(max_iter=500), {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh']})
}

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

results = []

for name, (clf, params) in classifiers.items():
    print(f'\n====== A2: Tuning & Training {name} ======')
    if params:
        search = RandomizedSearchCV(clf, param_distributions=params, n_iter=3, cv=3, random_state=42)
        try:
            search.fit(X_train, y_train)
            best_clf = search.best_estimator_
        except Exception as e:
            print(f'Error with {name}:', e)
            continue
    else:
        clf.fit(X_train, y_train)
        best_clf = clf

    # ====== A3 & A4: Training, Predictions, and Metrics ======
    y_train_pred = best_clf.predict(X_train)
    y_test_pred = best_clf.predict(X_test)

    metrics = {
        'Train Acc': accuracy_score(y_train, y_train_pred),
        'Test Acc': accuracy_score(y_test, y_test_pred),
        'Train Prec': precision_score(y_train, y_train_pred),
        'Test Prec': precision_score(y_test, y_test_pred),
        'Train Recall': recall_score(y_train, y_train_pred),
        'Test Recall': recall_score(y_test, y_test_pred),
        'Train F1': f1_score(y_train, y_train_pred),
        'Test F1': f1_score(y_test, y_test_pred)
    }

    results.append({'Model': name, **metrics})
    print(f'Done. Metrics:', metrics)

# ====== A5: Tabulating All Classifier Results ======
print('\n====== A5: Model Performance Summary Table ======')
print(f"{'Model':<12} {'TrAcc':<7} {'TeAcc':<7} {'TrPrec':<8} {'TePrec':<8} {'TrRecall':<9} {'TeRecall':<9} {'TrF1':<6} {'TeF1':<6}")
for r in results:
    print(f"{r['Model']:<12} {r['Train Acc']:.3f}  {r['Test Acc']:.3f}  {r['Train Prec']:.3f}  {r['Test Prec']:.3f}  {r['Train Recall']:.3f}  {r['Test Recall']:.3f}  {r['Train F1']:.3f}  {r['Test F1']:.3f}")

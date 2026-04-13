# Task 1 — Reproduce and Identify Leakage

# I scaled the entire dataset before splitting, which caused data leakage.
# The scaler used test data statistics during training, making the test set
# no longer truly unseen. This inflates accuracy and won't reflect real
# world performance.

from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy: ", model.score(X_test, y_test))


# Task 2 — Fix the Workflow Using a Pipeline

# Split the data first, then used a Pipeline to bundle the scaler and model.
# This ensures the scaler only learns from training folds, never touching
# test data. Cross-validation gives a reliable mean and standard deviation
# instead of a single accuracy score.

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')

print("CV Scores:     ", cv_scores.round(4))
print("Mean Accuracy: ", round(cv_scores.mean(), 4))
print("Std Deviation: ", round(cv_scores.std(), 4))


# Task 3 — Experiment with Decision Tree Depth

# Depth 1 — too simple, underfits, low accuracy on both train and test.
# Depth 20 — memorises training data, 100% train but drops on test, overfits.
# Depth 5 — best balance, good train accuracy and test accuracy holds up.
# Depth 5 is the right choice as it generalises well without memorising.

from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

depths = [1, 5, 20]

print(f"{'Depth':<10} {'Train Accuracy':<20} {'Test Accuracy'}")
print("-" * 45)

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)

    train_acc = dt.score(X_train, y_train)
    test_acc  = dt.score(X_test, y_test)

    print(f"{depth:<10} {train_acc:<20.4f} {test_acc:.4f}")
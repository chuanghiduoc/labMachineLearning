import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return [self._predict(sample, self.tree) for sample in X]

    def _predict(self, sample, tree):
        # If leaf node, return the predicted class
        if not isinstance(tree, dict):
            return tree

        feature = tree['feature']
        value = tree['value']
        if sample[feature] < value:
            return self._predict(sample, tree['left'])
        else:
            return self._predict(sample, tree['right'])

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # If only one class left or max depth reached
        if len(unique_classes) == 1 or (self.max_depth and depth == self.max_depth):
            return unique_classes[0]

        # Find the best split
        best_split = self._best_split(X, y)
        if best_split is None:
            return np.bincount(y).argmax()  # Return majority class

        left_indices = X[:, best_split['feature']] < best_split['value']
        right_indices = X[:, best_split['feature']] >= best_split['value']

        left_tree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {
            'feature': best_split['feature'],
            'value': best_split['value'],
            'left': left_tree,
            'right': right_tree
        }

    def _best_split(self, X, y):
        best_gain = -1
        best_split = None
        num_features = X.shape[1]

        for feature in range(num_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold

                if len(np.unique(y[left_indices])) == 0 or len(np.unique(y[right_indices])) == 0:
                    continue

                gain = self._information_gain(y, y[left_indices], y[right_indices])
                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature': feature, 'value': threshold}

        return best_split

    def _information_gain(self, y, left_y, right_y):
        p = float(len(left_y)) / (len(left_y) + len(right_y))
        return self._entropy(y) - (p * self._entropy(left_y) + (1 - p) * self._entropy(right_y))

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return -np.sum(proportions * np.log2(proportions + 1e-9))

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            # Select random samples from training data
            indices = np.random.choice(len(X), len(X), replace=True)
            tree.fit(X[indices], y[indices])
            self.trees.append(tree)

    def predict(self, X):
        # Make predictions by majority vote from all trees
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.array([np.bincount(tree_pred).argmax() for tree_pred in tree_preds.T])

# Simulated data for the example
X = np.random.rand(15000, 5)  # 15,000 samples, 5 features
y = np.random.randint(0, 2, size=15000)  # Labels 0 or 1

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForest(n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print("Predictions:", y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
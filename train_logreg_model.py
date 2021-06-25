from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold

class ModelTrainer():
    def __init__(self, x_train, y_train) -> None:
        self.x_train, self.y_train = x_train, y_train
        self.anomaly_weights = [5, 10]
        self.kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    def train(self):
        logs = []
        for i in range(len(self.anomaly_weights)):
            fold = 1
            
            for train, test in self.kfold.split(self.x_train, self.y_train):
                weight =self.anomaly_weights[i]
                class_weights = {0:1, 1:weight}
                print(f'Training for weight {weight}, fold {fold}')
                model = LogisticRegression(random_state=42, max_iter=400, 
                                            solver='newton-cg', 
                                            class_weight=class_weights)
                model.fit(self.x_train[train], self.y_train[train])
                preds = model.predict(self.x_train[test])
                log = {
                    "Weight": weight,
                    "Fold": fold,
                    "F1_score": f1_score(self.y_train[test], preds),
                    "Precision": precision_score(self.y_train[test], preds),
                    "Recall": recall_score(self.y_train[test], preds),
                    "Model": model
                }
                logs.append(log)
                fold = fold + 1
        return logs
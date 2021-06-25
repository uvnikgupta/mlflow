import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

class DataLoader:
    def load_data(self, path: str, targetCol: list, ignoreCols: list):
        cc = pd.read_csv(path)
        non_feature_cols = ignoreCols + targetCol
        self.featureColumns = cc.drop(non_feature_cols, axis=1).columns
        features = cc.drop(non_feature_cols, axis=1).values
        target = cc[targetCol].values

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for idx1, idx2 in split.split(features, target):
            x_trval, self.x_test = features[idx1], features[idx2]
            y_trval, self.y_test = target[idx1], target[idx2]
            
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        for idx1, idx2 in split.split(x_trval, y_trval):
            self.x_train, self.x_val = x_trval[idx1], x_trval[idx2]
            self.y_train, self.y_val = y_trval[idx1], y_trval[idx2]
            
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        self.x_val = scaler.fit_transform(self.x_val)
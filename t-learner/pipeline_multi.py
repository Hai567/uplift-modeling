from xgboost import XGBClassifier
import numpy as np

class MultiTreatmentTLearner:
    def __init__(
        self, 
        learning_rate=0.005, 
        max_depth=3,
        subsample=0.7, 
        colsample_bytree=0.7, 
        reg_lambda=0.1, 
        min_child_weight=0.001, 
        gamma=0.001,
        device=None,
        n_estimators=500,
        seed=42
    ):
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.seed = seed
        self.device = device
        
        self.models = {} # Dictionary lưu các model: {0: model_ctrl, 1: model_men, 2: model_women...}
        self.treatment_classes = []

    def fit(self, X, y, t):
        """
        X: Features
        y: Outcome
        t: Treatment column (vd: 0, 1, 2)
        """
        self.treatment_classes = np.unique(t)
        params = {
            "learning_rate": self.learning_rate, 
            "max_depth": int(self.max_depth),
            "subsample": self.subsample, 
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda, 
            "min_child_weight": self.min_child_weight, 
            "gamma": self.gamma,
            "n_estimators": self.n_estimators,
            "device": self.device,
            # "tree_method": "hist",
            "random_state": self.seed 
        }
        
        # Với mỗi nhóm treatment (kể cả control t=0), ta train 1 model riêng
        for group in self.treatment_classes:
            print(f"  -> Training model for group {group}...")
            
            # Lọc dữ liệu của nhóm đó
            X_grp = X[t == group]
            y_grp = y[t == group]
            
            # Khởi tạo và train model
            model = XGBClassifier(**params)
            model.fit(X_grp, y_grp)
            
            self.models[group] = model
            
        return self

    def predict(self, X, treatment_group):
        """
        Dự đoán Uplift cho một nhóm treatment cụ thể so với Control (group 0).
        Uplift = P(y|do(t=treatment_group)) - P(y|do(t=0))
        """
        # Dự đoán xác suất nếu thuộc nhóm Control (Model 0)
        p0 = self.models[0].predict_proba(X)[:, 1]
        
        # Dự đoán xác suất nếu thuộc nhóm Treatment đang xét
        pt = self.models[treatment_group].predict_proba(X)[:, 1]
        
        return pt - p0
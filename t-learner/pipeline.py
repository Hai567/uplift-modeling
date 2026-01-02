from xgboost import XGBClassifier
import torch
class TLearnerPipeline():
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
        if device:
            self.model_device = device
        else:
            self.model_device = "cuda" if torch.cuda.is_available() else "cpu"
            
        
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.n_estimators = n_estimators
        self.seed = seed
        
        self.model_0 = None # Model for control group
        self.model_1 = None # Model for treatment group

    def fit(self, X, y, t):
        """
        Huấn luyện T-Learner theo 2 bước tách biệt.
        Args:
            X: Ma trận đặc trưng (Covariates)
            y: Kết quả quan sát (Outcome)
            w: Chỉ thị can thiệp (Treatment indicator: 0 hoặc 1)
        """
        
        # Model's params
        params = {
            "learning_rate": self.learning_rate, 
            "max_depth": int(self.max_depth),
            "subsample": self.subsample, 
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda, 
            "min_child_weight": self.min_child_weight, 
            "gamma": self.gamma,
            "n_estimators": self.n_estimators,
            "device": self.model_device,
            # "tree_method": "hist",
            "random_state": self.seed 
        }
        
        # Clone base learner để đảm bảo 2 mô hình độc lập
        self.model_0 = XGBClassifier(**params)
        self.model_1 = XGBClassifier(**params)

        # Tách dữ liệu thành 2 nhóm
        # Control group (t=0)
        X_0 = X[t == 0]
        y_0 = y[t == 0]

        # Treatment group (t=1)
        X_1 = X[t == 1]
        y_1 = y[t == 1]

        self.model_0.fit(X_0, y_0)
        self.model_1.fit(X_1, y_1)

        return self

    def predict(self, X):
        """
        returns: 
            y0_pred: Dự đoán kết quả nếu KHÔNG có can thiệp
            y1_pred: Dự đoán kết quả nếu CÓ can thiệp
            cate_pred: Dự đoán CATE cho mỗi cá thể
            
        Formula: CATE = y1_pred(x) - y0_pred(x) 
        """
        # Dự đoán kết quả nếu KHÔNG có can thiệp
        # Chỉ lấy cột index 1 (xác suất của class Positive/Mua hàng)
        # predict_proba trả về [prob_0, prob_1], ta cần prob_1
        y0_pred = self.model_0.predict_proba(X)[:, 1]

        # Dự đoán kết quả nếu CÓ can thiệp
        y1_pred = self.model_1.predict_proba(X)[:, 1]

        # CATE là hiệu của hai dự đoán
        cate_pred = y1_pred - y0_pred
        return y0_pred, y1_pred, cate_pred
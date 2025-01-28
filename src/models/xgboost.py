from models.model_ import Model_
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score



class XGBoost(Model_):
    def __init__(self):
        self.model = xgb.XGBClassifier()
        super().__init__(self.model)

    def objective(self, trial, X, y):
        # Define hyperparameter search space
        eta = trial.suggest_float("eta", 0.01, 0.3, log=True)  # Learning rate
        max_depth = trial.suggest_int("max_depth", 3, 20)  # Maximum tree depth
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)  # Minimum child weight
        subsample = trial.suggest_float("subsample", 0.5, 1.0)  # Subsample ratio
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)  # Feature sampling ratio
        n_estimators = trial.suggest_int("n_estimators", 50, 500)  # Number of trees
        gamma = trial.suggest_float("gamma", 0, 5)  # Minimum loss reduction

        # Create the model with the sampled hyperparameters
        model = xgb.XGBClassifier(
            eta=eta,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_estimators=n_estimators,
            gamma=gamma,
            random_state=42,
            use_label_encoder=False,  # Avoid warnings in recent XGBoost versions
            eval_metric="logloss",  # Set a common evaluation metric
        )

       # Use recall as the scoring metric
        recall_scorer = make_scorer(recall_score, average="weighted")
        scores = cross_val_score(model, X, y, cv=5, scoring=recall_scorer)

        return scores.mean()

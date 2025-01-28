from models.model_ import Model_
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


class SVM(Model_):
    def __init__(self):
        self.model = SVC()
        super().__init__(self.model)

    def objective(self, trial, X, y):
        C = trial.suggest_float("C", 0.001, 100, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        kernel = trial.suggest_categorical(
            "kernel", ["linear", "poly", "rbf", "sigmoid"]
        )
        degree = trial.suggest_int("degree", 2, 5)  # Only for poly
        coef0 = trial.suggest_float("coef0", 0.0, 1.0)  # Only for poly and sigmoid

        model = SVC(C=C, gamma=gamma, kernel=kernel, degree=degree, coef0=coef0)

        # Use recall as the scoring metric
        recall_scorer = make_scorer(recall_score, average="weighted")
        scores = cross_val_score(model, X, y, cv=5, scoring=recall_scorer)
        
        return scores.mean()

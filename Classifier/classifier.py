import os
import pandas as pd
import numpy as np
from joblib import load


class Classifier():
    def __init__(self):
        self.base_path = "./Classifier/"
        self.logs_path = "./logs/"
        self.logs_file_name = "summary.csv"
        self.fault_label = ['optimizer', 'learning_rate', 'active_func', 'loss', 'epoch']

        self.result = self.fault_label

        self.load_models()

    def voting(self, predict_list):
        predict_sum = np.sum(predict_list, axis=1)
        predict_sum_shift = predict_sum - np.array([0, 2, 3, 6, 5])
        predict_sum_shift[predict_sum_shift <= 0] = 0
        predict_sum_shift[predict_sum_shift > 0] = 1

        predict_list_np = np.array(predict_sum_shift)
        threshold = 1
        pred_voting = (np.sum(predict_list_np, axis=0) >= threshold).astype(int)  # if # of 1 > # of 0 -> 1, else -> 0
        return pred_voting

    def load_models(self):
        self.model_path = os.listdir(f"{self.base_path}models/")
        self.model_path = [f"{self.base_path}models/" + path for path in self.model_path]

    def predict(self):
        df = pd.read_csv(os.path.join(self.logs_path , self.logs_file_name))
        df = df.fillna(0.0) # replace NaN values with 0

        features = list(filter(lambda x: x.startswith("ft_"), df.columns))
        X = df[features]

        mask_index = ~(np.max(X, axis=0) == np.min(X, axis=0))
        mask_index = mask_index.values
        X.loc[:, mask_index] = (X.loc[:, mask_index] - np.min(X.loc[:, mask_index], axis=0)) / (
                np.max(X.loc[:, mask_index], axis=0) - np.min(X.loc[:, mask_index], axis=0))
        X.loc[:, ~mask_index] = 0.0
        X = X.astype(np.float32)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0.0)


        predictList = []
        for model in self.model_path:
            classifier = load(model)
            predictList.append(classifier.predict(X))

        self.prediction_result = self.voting(predictList)

        self.prediction_result_faults = []
        for i in range(len(self.prediction_result)):
            if self.prediction_result[i] == 1:
                self.prediction_result_faults.append(self.fault_label[i])

def main():
    classifier = Classifier()
    classifier.predict()


if __name__ == '__main__':
    main()

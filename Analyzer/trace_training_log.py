import dill
from scipy.stats import skew, sem
import numpy as np
import statistics
import csv
import os
import sys
from math import floor

import traceback

def has_nan_inf(data):
    if data[0] is None:
        return False
    np_list = np.array(data)
    result = (np.isnan(np_list).any() or np.isinf(np_list).any())
    return result

def flatten_weight(weight):
    weigth_history_flatten = np.ndarray([])
    for item_i in weight:
        for item_j in item_i:
            tmp_flat = np.concatenate([x.ravel() for x in item_j])
            weigth_history_flatten = np.append(weigth_history_flatten, tmp_flat)
    return weigth_history_flatten


def flatten_last_batch_weight(weight):
    weigth_history_flatten = np.ndarray([])
    for layer_weight in weight:
        tmp_flat = np.concatenate([x.ravel() for x in layer_weight])
        weigth_history_flatten = np.append(weigth_history_flatten, tmp_flat)
    return weigth_history_flatten

def check_weight_treshold(treshold, weight):
    large_weight = 0
    for weight_array in weight:
        weight_end_epoch = weight_array[-1]
        tmp_flat = list(np.concatenate([x.ravel() for x in weight_end_epoch]))
        if np.abs(tmp_flat).max() > treshold:
            large_weight += 1

    return large_weight


def check_large_weight_treshold(treshold, weight):
    large_weight = 0
    for weight_array in weight:
        # weight_end_epoch = weight_array[-1]
        tmp_flat = list(np.concatenate([x.ravel() for x in weight_array]))
        if np.abs(tmp_flat).max() > treshold:
            large_weight += 1

    return large_weight

def gradient_zero_radio(gradient_list):
    kernel = []
    bias = []
    total_zero = 0
    total_size = 0
    for i in range(len(gradient_list)):
        zeros = np.sum(gradient_list[i] == 0)
        total_zero += zeros
        total_size += gradient_list[i].size
        # if i % 2 == 0:
        #     kernel.append(zeros / gradient_list[i].size)
        # else:
        #     bias.append(zeros / gradient_list[i].size)
    total = float(total_zero) / float(total_size)
    return total, kernel, bias

def gradient_normalize(gradient_list):
    try:
        assert len(gradient_list) % 2 == 0
    except AssertionError:
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb)

    norm_kernel_list = []
    norm_bias_list = []
    for i in range(int(len(gradient_list) / 2)):
        norm_kernel_list.append(np.linalg.norm(np.array(gradient_list[2 * i])))
        norm_bias_list.append(np.linalg.norm(np.array(gradient_list[2 * i + 1])))
    return norm_kernel_list, norm_bias_list

class Analyzer:
    def __init__(self, logs_path = "/"):
        self.file_path = f".{logs_path}logs/meta_data.pickle"
        self.ft_nan_loss = []
        self.ft_nan_acc = []
        self.ft_nan_weight = []
        self.ft_nan_gradient = []
        self.ft_large_weight = []
        self.ft_decrease_acc = []
        self.ft_increase_loss = []
        self.ft_cons_mean_weight = []
        self.ft_cons_std_weight = []
        self.ft_gap_train_test = []
        self.ft_test_turn_bad = []
        self.ft_weight_change_little = []
        self.ft_died_relu = []
        self.ft_vanish_gradient = []
        self.ft_explode_gradient = []
        self.ft_slow_converge = []
        self.ft_loss_oscillation = []
        self.ft_sc_accuracy = []

        self.threshold_die = 0.7
        self.gradient_rate_low_threshold = 1e-3
        self.gradient_rate_high_threshold = 70
        self.gradient_kernel_low_threshold = 1e-4
        self.determine_threshold = 5
        self.unstable_threshold = 0.05
        self.unstable_rate = 0.25
        self.large_weight_treshold = 5
        self.epsilon = 10e-3
        self.satisfied_acc = 0.7
        self.sc_threshold = 0.01


        # run some initializer functions
        self.load_logs()
        self.data_preprocessing()

        # save the result of the analyzer
        # self.create_dataset_for_classifiers()

    def load_weight_gradient_logs(self):
        self.weight_logs = []
        self.gradient_logs = []

        self.tmp_log_directory= "./logs/tmp_logs/"
        files = os.listdir(self.tmp_log_directory)

        for file_name in files:
            file = open(f"./logs/tmp_logs/{file_name}", "rb")
            tmp_data = dill.load(file)

            self.weight_logs.extend(tmp_data.get('weigth_history'))
            self.gradient_logs.extend(tmp_data.get('gradient_history'))


    def load_logs(self):
        file = open(self.file_path, "rb")
        self.data = dill.load(file)
        # self.weight_logs = self.data.train_logs['weigth_history']
        # self.gradient_logs = self.data.train_logs['gradient_history']

        self.acc = self.data.history.get('accuracy_epoch')
        self.loss = self.data.history.get('loss_epoch')
        self.acc_valid = self.data.history.get('val_accuracy_epoch')
        self.loss_valid = self.data.history.get('val_loss_epoch')

        self.batch_acc = self.data.history.get('loss_batch')
        self.bact_loss = self.data.history.get('acc_batch')

    def check_loss_oscillation(self, epoch):
        # check either loss is oscillating or not
        maximum = []
        minimum = []
        counter = 0
        acc_epoch = self.batch_acc[epoch]
        for i in range(len(acc_epoch)):
            if i == 0 or i == len(acc_epoch) - 1:
                continue
            if acc_epoch[i] - acc_epoch[i - 1] >= 0 and acc_epoch[i] - acc_epoch[i + 1] >= 0:
                maximum.append(acc_epoch[i])
            if acc_epoch[i] - acc_epoch[i - 1] <= 0 and acc_epoch[i] - acc_epoch[i + 1] <= 0:
                minimum.append(acc_epoch[i])
        for i in range(min(len(maximum), len(minimum))):
            if maximum[i] - minimum[i] >= self.unstable_threshold:
                counter += 1
        if counter >= self.unstable_rate * len(acc_epoch):
            return True
        else:
            return False

    def maximum_delta(self, list):
        maximum_delta = 0
        for i in range(len(list) - 1):
            if list[i + 1] - list[i] > maximum_delta:
                maximum_delta = list[i + 1] - list[i]
        return maximum_delta

    def data_preprocessing(self):
        self.ft_loss = self.loss
        self.ft_acc = self.acc
        self.ft_loss_val = self.loss_valid
        self.ft_acc_valid = self.acc_valid


        acc_none = False
        acc_valid_none = False
        if self.acc.count(None) == len(self.acc):
            acc_none = True

        if self.acc_valid.count(None) == len(self.acc_valid):
            acc_valid_none = True

        # calculate nan_loss amd nan_acc
        epoch_loss_by_batch = self.data.history.get('loss_batch')
        epoch_acc_by_batch = self.data.history.get('acc_batch')



        # calculation for each epoch
        for epoch in range(self.data.epochs-1):
            # load weigth and gradient
            if (epoch == 0) or ((epoch + 1) % 2 == 0) or ((epoch + 1) == self.data.epochs):
                weigth_gradient_file_num= floor(epoch / 2)
                weigth_gradient_file = open(f"./logs/tmp_logs/train_log_tmp_{weigth_gradient_file_num}.pickle", "rb")
                weigth_gradient_data = dill.load(weigth_gradient_file)
            weigth_gradient_order = epoch % 2
            epoch_weight = weigth_gradient_data.get('weigth_history')[weigth_gradient_order]
            epoch_gradient = weigth_gradient_data.get('gradient_history')[weigth_gradient_order]


            # calculate nan_loss
            epoch_loss = epoch_loss_by_batch[epoch]
            self.ft_nan_loss.append(has_nan_inf(epoch_loss))

            # calculate increase_loss
            increase_loss = 0
            if len(epoch_loss) > 1:
                for i in range(len(epoch_loss) - 1):
                    if epoch_loss[i] < epoch_loss[i + 1]:
                        increase_loss += 1
            self.ft_increase_loss.append(increase_loss)

            # calculate nan_acc
            epoch_acc = epoch_acc_by_batch[epoch]
            self.ft_nan_acc.append(has_nan_inf(epoch_acc))

            # calculate decrease_acc
            decrease_acc = 0
            if not acc_none:
                if len(epoch_acc) > 1:
                    for i in range(len(epoch_acc) - 1):
                        if epoch_acc[i] > epoch_acc[i + 1]:
                            decrease_acc += 1
            self.ft_decrease_acc.append(decrease_acc)

            # calculate nan_weight
            epoch_weight_flatten = flatten_last_batch_weight(epoch_weight)
            self.ft_nan_weight.append(has_nan_inf(epoch_weight_flatten))

            # calculate nan_gradient
            epoch_gradient_flatten = flatten_last_batch_weight(epoch_gradient)
            self.ft_nan_gradient.append(has_nan_inf(epoch_gradient_flatten))

            self.ft_large_weight.append(check_large_weight_treshold(self.large_weight_treshold, epoch_weight))

            last_weight_mean = 0
            last_weight_std = 0

            cons_mean_weight = 0
            cons_std_weight = 0

            # calculating 'cons_mean_weight' and 'cons_std_weight'
            for weight in epoch_weight:
                flatten_weight = flatten_last_batch_weight(weight)
                weight_mean = np.mean(flatten_weight)
                weight_std = np.std(flatten_weight)

                if last_weight_mean == 0:
                    last_weight_mean = weight_mean
                elif weight_mean == last_weight_mean:
                    cons_mean_weight += 1
                    last_weight_mean = weight_mean
                else:
                    last_weight_mean = weight_mean

                if last_weight_std == 0:
                    last_weight_std = weight_std
                elif weight_std == last_weight_std:
                    cons_std_weight += 1
                    last_weight_std = weight_std
                else:
                    last_weight_std = weight_std

            self.ft_cons_mean_weight.append(cons_mean_weight)
            self.ft_cons_std_weight.append(cons_std_weight)

            # calculation for gap_train_test
            if acc_none and acc_valid_none:
                self.ft_gap_train_test.append(0)
            elif (self.acc[epoch] <= 0.9 and self.acc[epoch] - self.acc_valid[epoch] >= 0.1) or (self.acc[epoch] > 0.9 and self.acc[epoch] - self.acc_valid[epoch] >= 0.07):
                self.ft_gap_train_test.append(1)
            else:
                self.ft_gap_train_test.append(0)

            # calculation for test_turn_bad
            if (epoch > 0) and (self.loss[epoch] - self.loss[epoch-1] < -self.epsilon) and (self.loss_valid[epoch] - self.loss_valid[epoch-1] > self.epsilon):
                self.ft_test_turn_bad.append(1)
            else:
                self.ft_test_turn_bad.append(0)

            # calculation for slow_converge
            if acc_none:
                self.ft_slow_converge.append(False)
            elif (max(epoch_acc) < self.satisfied_acc) or (self.acc_valid[epoch] < self.satisfied_acc):
                self.ft_slow_converge.append(True)
            else:
                self.ft_slow_converge.append(False)

            # calculation for loss_oscillation
            self.ft_loss_oscillation.append(self.check_loss_oscillation(epoch=epoch))

            if acc_valid_none:
                max_delta_valid_acc = 0
            elif epoch == 0:
                max_delta_valid_acc = self.acc_valid[epoch]
            else:
                max_delta_valid_acc = self.acc_valid[epoch] - self.acc_valid[epoch-1]


            if acc_none:
                self.ft_sc_accuracy.append(False)
            elif max_delta_valid_acc < self.sc_threshold and self.maximum_delta(
                    self.batch_acc[epoch]) < self.sc_threshold:
                self.ft_sc_accuracy.append(True)
            else:
                self.ft_sc_accuracy.append(False)

            # calculation for 'died_relu', 'vanish_gradient', and 'explode_gradient'
            died_relu = 0
            vanish_gradient = 0
            explode_gradient = 0

            for gradient_batch in epoch_gradient:
                # last_batch_gradient = gradient_epoch[-1]
                total_ratio, kernel_ratio, bias_ratio = gradient_zero_radio(gradient_batch)
                if total_ratio >= self.threshold_die:
                    died_relu += 1

                gradient_norm_kernel, gradient_norm_bias = gradient_normalize(gradient_batch)
                gradient_rate = (gradient_norm_kernel[0] / gradient_norm_bias[-1])

                if gradient_rate < self.gradient_rate_low_threshold and gradient_norm_kernel[
                    0] < self.gradient_kernel_low_threshold:
                    vanish_gradient += 1

                if gradient_rate > self.gradient_rate_high_threshold:
                    explode_gradient += 1

            self.ft_died_relu.append(died_relu)
            self.ft_vanish_gradient.append(vanish_gradient)
            self.ft_explode_gradient.append(explode_gradient)

            # This is just a constant value
            self.ft_weight_change_little.append(0)

    def create_dataset_for_classifiers(self):
        self.feature_dict = {
            "ft_not_converge": self.ft_slow_converge,
            "ft_unstable_loss": self.ft_loss_oscillation,
            "ft_nan_loss" : self.ft_nan_loss,
            "ft_test_not_wel": self.ft_gap_train_test,
            "ft_test_turn_bad": self.ft_test_turn_bad,
            "ft_sc_accuracy": self.ft_sc_accuracy,
            "ft_died_relu": self.ft_died_relu,
            "ft_vanish_gradient": self.ft_vanish_gradient,
            "ft_explode_gradient": self.ft_explode_gradient,
            "ft_nan_gradient": self.ft_nan_gradient,
            "ft_large_weight": self.ft_large_weight,
            "ft_nan_weight": self.ft_nan_weight,
            "ft_weight_change_little": self.ft_weight_change_little, # check this one
            "ft_decrease_acc": self.ft_decrease_acc,
            "ft_increase_loss": self.ft_increase_loss,
            "ft_cons_mean_weight": self.ft_cons_mean_weight,
            "ft_cons_std_weight": self.ft_cons_std_weight,
            "ft_nan_acc" : self.ft_nan_acc,
        }


        csv_write_header = []
        csv_write_data = []

        csv_write_header.extend(["ft_loss", "ft_accuracy", "ft_val_loss", "ft_val_accuracy"])
        csv_write_data.extend([
            self.ft_loss[-1], self.ft_acc[-1], self.ft_loss_val[-1], self.ft_acc_valid[-1]
        ])

        for key in self.feature_dict.keys():
            feature = self.feature_dict.get(key)
            if feature[0] == True or feature[0] == False:
                feature = list(map(int, feature))

            max_feature = max(feature)
            min_feature = min(feature)
            median_feature = statistics.median(feature)
            mean_feature = statistics.mean(feature)
            var_feature = statistics.variance(feature)
            std_feature = statistics.pstdev(feature)
            skew_feature = skew(feature)
            # if np.isnan(skew_feature):
            #     skew_feature = 0
            sem_feature = sem(feature)

            csv_write_header.extend([f"{key}_mean",
                                    f"{key}_std",
                                    f"{key}_skew",
                                    f"{key}_median",
                                    f"{key}_var",
                                    f"{key}_sem",
                                    f"{key}_max",
                                    f"{key}_min"
                                    ])
            csv_write_data.extend([mean_feature,
                                  std_feature,
                                  skew_feature,
                                  median_feature,
                                  var_feature,
                                  sem_feature,
                                  max_feature,
                                  min_feature
                                  ])


        csvfile = open('./logs/summary.csv', 'w', newline='')
        writer = csv.writer(csvfile)
        writer.writerow(csv_write_header)
        writer.writerow(csv_write_data)

        csvfile.close()

def main():
    training_logs_analyzer = Analyzer()

if __name__ == "__main__":
    main()
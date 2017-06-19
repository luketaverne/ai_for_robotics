#!/usr/bin/env python

import Features as features
import LinearRegressionModel as model
import DataSaver as saver
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np

# Remove lapack warning on OSX (https://github.com/scipy/scipy/issues/5998).
import warnings
warnings.filterwarnings(action="ignore", module="scipy",
                        message="^internal gelsd")

plt.close('all')

# TODO decide if you want to show the plots to compare input and output data
show_plots = False

# data_generator = data.DataGenerator()
data_saver = saver.DataSaver('data', 'data_samples.pkl')
input_data, output_data = data_saver.restore_from_file()
n_samples = input_data.shape[0]
if(show_plots):
    plt.figure(0)
    plt.scatter(input_data[:, 0], output_data[:, 0])
    plt.xlabel("x1")
    plt.ylabel("y")
    plt.figure(1)
    plt.scatter(input_data[:, 1], output_data[:, 0])
    plt.xlabel("x2")
    plt.ylabel("y")
    if (input_data.shape[1] > 2):
        plt.figure(2)
        plt.scatter(input_data[:, 2], output_data[:, 0])
        plt.xlabel("x3")
        plt.ylabel("y")
        plt.figure(3)
        plt.scatter(input_data[:, 3], output_data[:, 0])
        plt.xlabel("x4")
        plt.ylabel("y")


# Split data into training and validation
# TODO Overcome the problem of differently biased data
ratio_train_validate = 0.8
# idx_switch = int(n_samples * ratio_train_validate)
# training_input = input_data[:idx_switch, :]
# training_output = output_data[:idx_switch, :]
# validation_input = input_data[idx_switch:, :]
# validation_output = output_data[idx_switch:, :]

n_parts = 4
n_samples_part = n_samples/n_parts
training_input = [0]*4
training_output = [0]*4
validation_input = [0]*4
validation_output = [0]*4
lm = [0]*4
mse_part = [0]*4
mse = 0

for i in range(n_parts):
    idx_train_start = int(i * n_samples_part)
    idx_test_start = int(idx_train_start + n_samples_part * ratio_train_validate)
    idx_test_end = int(idx_train_start + n_samples_part)
    training_input[i] = input_data[idx_train_start:idx_test_start, :]
    training_output[i] = output_data[idx_train_start:idx_test_start, :]
    validation_input[i] = input_data[idx_test_start:idx_test_end, :]
    validation_output[i] = output_data[idx_test_start:idx_test_end, :]

    # Fit model
    lm[i] = model.LinearRegressionModel()
    # TODO use and select the new features
    lm[i].set_feature_vector([features.LinearX1(), features.LinearX2(),
                            features.LinearX3(), features.LinearX4(),
                            features.SquareX1(), features.SquareX2(), 
                            features.SquareX3(), features.SquareX4(), 
                            features.ExpX1(), features.ExpX2(),
                            features.ExpX3(), features.ExpX4(),
                            features.LogX1(), features.LogX2(), 
                            features.LogX3(), features.LogX4(),
                            features.SinX1(), features.SinX2(),
                            features.SinX3(), features.SinX4(),
                            features.CrossTermX1X2(), features.CrossTermX1X3(),
                            features.CrossTermX1X4(), features.CrossTermX2X3(),
                            features.CrossTermX2X4(), features.CrossTermX3X4(),
                            features.Identity()])
    lm[i].fit(training_input[i], training_output[i])

    # Validation
    mse_part[i] = lm[i].validate(validation_input[i], validation_output[i])
    mse += mse_part[i]/4
    print('MSE for part {}: {}'.format(i, mse_part[i]))
    print('feature weights for part {}:\n{}'.format(i, lm[i].beta))
    print(' ')

print('MSE for all: {}'.format(mse))

# load submission data
submission_loader = saver.DataSaver('data', 'submission_data.pkl')
submission_data = submission_loader.load_submission()
# predict output
# submission_output = lm.predict(submission_input)

submission_input = [0]*4
submission_output = np.array([])

for i in range(n_parts):
    idx_sub_start = int(i * n_samples_part)
    idx_sub_end = int(idx_sub_start + n_samples_part)
    submission_input[i] = submission_data[idx_sub_start:idx_sub_end, :]
    submission_output = np.append(submission_output, lm[i].predict(submission_input[i]))
    # submission_output.append(lm[i].predict(submission_input))

print(np.shape(submission_output))
# submission_output = submission_output.reshape((n_sample, 1))

#save output
pkl.dump(submission_output, open("results.pkl", 'wb'))

plt.show()



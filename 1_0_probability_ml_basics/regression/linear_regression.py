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

n_parts = 4;
samp_per_part = n_samples / n_parts;
training_input = [0]*n_parts
training_output = [0]*n_parts
validation_input = [0]*n_parts
validation_output = [0]*n_parts
lm = [0]*n_parts
mse_part = [0]*n_parts
mse = 0


# Fit model for each part
for part in range(0,n_parts):
    train_start_i = int(part * samp_per_part)
    test_start_i = int(train_start_i + samp_per_part * ratio_train_validate)
    test_end_i = int(train_start_i + samp_per_part)
    training_input[part] = input_data[train_start_i:test_start_i, :]
    training_output[part] = output_data[train_start_i:test_start_i, :]
    validation_input[part] = input_data[test_start_i:test_end_i, :]
    validation_output[part] = output_data[test_start_i:test_end_i, :]

    lm[part] = model.LinearRegressionModel()
    # TODO use and select the new features
    lm[part].set_feature_vector([features.LinearX1(), features.LinearX2(),
                            features.LinearX3(), features.LinearX4(),
                            features.SquareX1(), features.SquareX2(),
                            features.SquareX3(), features.SquareX4(),
                            features.ExpX1(), features.ExpX2(),
                            features.ExpX3(), features.ExpX4(),
                            features.LogX1(), features.LogX2(),
                            features.LogX3(), features.LogX4(),
                            features.SinX1(), features.SinX2(),
                            features.SinX3(), features.SinX4(),
                            # features.X1Cube(), features.X2Cube(),
                            # features.X3Cube(), features.X4Cube(),
                            # features.TanX1(), features.TanX2(),
                            # features.TanX3(), features.TanX4(),
                            # features.X1OverX2(), features.X1OverX3(),
                            # features.X1OverX4(), features.X2OverX3(),
                            # features.X2OverX4(), features.X3OverX4(),

                            features.CrossTermX1X2(), features.CrossTermX1X3(),
                            features.CrossTermX1X4(), features.CrossTermX2X3(),
                            features.CrossTermX2X4(), features.CrossTermX3X4(),
                            features.Identity()])
    lm[part].fit(training_input[part], training_output[part])


    # Validation
    mse_part[part] = lm[part].validate(validation_input[part], validation_output[part]);

    print('MSE for part {}: {}'.format(part, mse_part[part]))
    print(' ')
    print('feature weights for part {} \n{}'.format(part, lm[part].beta))

mse = np.sum(mse_part)/n_parts;
print('MSE combined: {}'.format(mse))
print(' ')


# load submission data
submission_loader = saver.DataSaver('data', 'submission_data.pkl')
submission_input = submission_loader.load_submission()


# predict output
submission_output = np.array([])

for part in range(n_parts):
    start_index = int(part * samp_per_part)
    end_index = int(start_index + samp_per_part)

    submission_output = np.append(submission_output, lm[part].predict(submission_input[start_index:end_index, :]))

#save output
pkl.dump(submission_output, open("results.pkl", 'wb'))

plt.show()

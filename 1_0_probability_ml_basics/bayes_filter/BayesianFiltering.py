import numpy as np
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
import time


def histogram_plot(pos, title=None, c='b'):
    axis = plt.gca()
    x = np.arange(len(pos))
    axis.bar(x, pos, color=c)
    plt.ylim((0, 1))
    plt.xticks(np.asarray(x) + 0.4, x)
    if title is not None:
        plt.title(title)


def normalize(input):
    normalized_value = np.ones(len(input))
    # TODO: Implement the normalization function #DONE
    normalized_value = input / np.sum(input)
    return normalized_value


def compute_likelihood(map, measurement, prob_correct_measurement):
    likelihood = np.zeros(len(map))
    if measurement == 1:
        likelihood = map * prob_correct_measurement + (1 - map)*(1 - prob_correct_measurement)
    else:
        likelihood = (1 - map) * prob_correct_measurement + (map)*(1 - prob_correct_measurement)
    # TODO: compute the likelihood #DONE

    return likelihood


def measurement_update(prior, likelihood):
    # TODO: compute posterior, use function normalize #DONE
    posterior = np.ones(len(prior))
    posterior = likelihood * prior
    posterior = normalize(posterior)


    return posterior  # TODO: change this line to return posterior #DONE


def prior_update(posterior, movement, movement_noise_kernel):
    # TODO: compute the new prior
    new_prior = np.ones(len(posterior))

    #Can't think of how to do this in a clever way without a loop
    for i in range(0,len(new_prior)):
        #place the movement_noise_kernel in a padded vector for convenience
        temp_kernel = np.zeros(len(new_prior))

        possible_move_indicies = [(i - movement + np.sign(movement)) % 20, (i - movement) % 20, (i - movement - np.sign(movement)) % 20]

        temp_kernel[possible_move_indicies] = movement_noise_kernel
        new_prior[i] = np.sum(posterior * temp_kernel)

    # HINT: be careful with the movement direction and noise kernel!
    return new_prior  # TODO: change this line to return new prior


def run_bayes_filter(measurements, motions, plot_histogram=False):
    map = np.array([0] * 20)  # TODO: define the map #DONE
    doors = [1, 5, 9, 10, 14, 16, 18] #indexes of doors
    map[doors] = 1; #set those indexes to 1 where there is a door. Index represents state number
    sensor_prob_correct_measure = 0.9  # TODO: define the probability of correct measurement #DONE
    movement_noise_kernel = [0.15, 0.8, 0.05]  # TODO: define noise kernel of the movement command #DONE? Is this correct? Undershoot, perfect, overshoot?

    # Assume uniform distribution since you do not know the starting position
    prior = np.array([1. / 20] * 20)
    likelihood = np.zeros(len(prior))

    number_of_iterations = len(measurements)
    if plot_histogram:
        fig = plt.figure("Bayes Filter")
    for iteration in range(number_of_iterations):
        # Compute the likelihood
        likelihood = compute_likelihood(map, measurements[iteration],
                                        sensor_prob_correct_measure)
        # Compute posterior
        print("Measurement: {}".format(measurements[iteration]))
        print("Iteration: {}".format(iteration))
        posterior = measurement_update(prior, likelihood)
        if plot_histogram:
            plt.cla()
            histogram_plot(map, title="Measurement update", c='k')
            histogram_plot(posterior, title="Measurement update", c='y')
            fig.canvas.draw()
            plt.show(block=False)
            time.sleep(.01)

        # Update prior
        print("Motion: {}".format(motions[iteration]))
        prior = prior_update(posterior, motions[iteration],
                             movement_noise_kernel)
        if plot_histogram:
            plt.cla()
            histogram_plot(map, title="Prior update", c='k')
            histogram_plot(prior, title="Prior update")
            fig.canvas.draw()
            plt.show(block=False)
            time.sleep(.01)
    plt.show()
    return prior

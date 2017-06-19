import numpy as np
import cv2
import math
from scipy.stats import multivariate_normal
from sklearn import mixture
import itertools


class ImageSegmenter(object):
    beta = 1  # strength of the second order clique criteria
    threshold = 0.05  # convergence threshold
    max_iterations = 100  # maximum number of iterations
    temperature_init = 5  # initial temperature
    discount = 1  # temperature scheduler's factor
    number_of_labels = 0  # number of labels in the image
    number_of_features = 0  # number of features to use
    size_ratio = 1  # increases the image by size_ratio, just for displaying

    image_height = 0
    image_width = 0

    label_mean = np.array([])
    label_covariance = np.array([])
    image_labels = np.array([])
    features = np.array([])
    image = np.array([])

    def __init__(self, beta, threshold, max_iterations, temperature_init,
                 discount, number_of_labels, size_ratio, features, image):
        self.beta = beta
        self.threshold = threshold
        self.max_iterations = max_iterations
        self.temperature_init = temperature_init
        self.discount = discount
        self.number_of_labels = number_of_labels
        self.size_ratio = size_ratio

        self.features = features.copy()
        self.image = image.copy()

        self.image_height, self.image_width, self.number_of_features = self.features.shape

        self.image_labels = np.random.randint(
            0,
            self.number_of_labels, (self.image_height, self.image_width),
            dtype=np.uint8)

        self.label_mean, self.label_covariance = computeLabelMeanAndVariance(
            self.number_of_labels, self.number_of_features, self.features)

        pass

    def singleton(self, i, j, label):
        # = -log(P(z|x_i)) -> "log likelihood of the features given the label"
        value = self.features[i, j]
        mean = self.label_mean[label]
        cov = self.label_covariance[label]
        mv = multivariate_normal(mean=mean,cov=cov)
        potential = -np.log(mv.pdf(value))
        #TODO: Calculate the singletone clique potential
        return potential

    def doubleton(self, i, j, label):
        #Beta * delta(x_i,x_j)
        possible_i_indicies = [ii for ii in range(i - 1, i + 2) if (ii >= 0 and ii < self.image.shape[0])]
        possible_j_indicies = [jj for jj in range(j - 1, j + 2) if (jj >= 0 and jj < self.image.shape[1])]

        neighborhood = [(ii, jj) for ii in possible_i_indicies for jj in possible_j_indicies]
        equal_mask = []
        for n in neighborhood:
            if label == self.image_labels[n]:
                equal_mask.append(-self.beta)
            else:
                equal_mask.append(self.beta)

        potential = np.sum(equal_mask)
        #TODO: Calculate the doubleton clique potential
        return potential

    def calculateGlobalEnergy(self):
        energy = 0.0
        singletons = 0.0
        doubletons = 0.0

        for ii in range(0, self.image.shape[0]):
            for jj in range(0, self.image.shape[1]):
                #Finish this implementation!
                singletons += self.singleton(ii, jj, self.image_labels[ii, jj])
                doubletons += self.doubleton(ii, jj, self.image_labels[ii, jj])

        #TODO: Calculate the energy of the whole image
        energy = singletons + doubletons / 2.0
        return energy

    def calculateLocalEnergy(self, i, j, label):
        #TODO: Calculate the energy of one pixel
        energy = self.singleton(i, j, label) + self.doubleton(i, j, label)

        return energy

    def displayCurrentLabels(self):
        red = [255, 0, 0]
        green = [0, 255, 0]
        blue = [0, 0, 255]
        white = [255, 255, 255]
        black = [0, 0, 0]
        magenta = [255, 0, 255]
        yellow = [255, 255, 0]
        cyan = [0, 255, 255]

        colors = [red, green, blue, white, black, magenta, yellow, cyan]

        if self.number_of_labels > len(colors):
            print(
                "Too many labels, cannot display. If you want to increase the number of labels, add additional colors to displayCurrentLabels method."
            )
            return False

        cv2.namedWindow("Current Labels")
        current_labels = np.zeros(
            (np.size(self.image_labels, 0), np.size(self.image_labels, 1), 3),
            dtype=np.uint8)
        for i in range(self.number_of_labels):
            current_labels[np.where(self.image_labels == i)] = colors[i]

        cv2.imshow("Current Labels",
                   cv2.resize(
                       np.concatenate((self.image, current_labels), axis=1),
                       (self.size_ratio * 2 * self.image_width,
                        self.size_ratio * self.image_height)))
        cv2.waitKey(1)
        return True

    def segmentImage(self):
        iterator = 0

        temperature = 0
        E_old = self.calculateGlobalEnergy()
        E = 0.0
        sum_deltaE = E - E_old
        Ek = np.zeros((self.number_of_labels, 1))
        temperature = self.temperature_init
        while iterator < self.max_iterations and math.fabs(
                sum_deltaE) > self.threshold:
            for i in range(self.image_height):
                for j in range(self.image_width):
                    for s in range(self.number_of_labels):
                        Ek[s] = math.exp(-self.calculateLocalEnergy(i, j, s) /
                                         temperature)
                    epsilon = np.random.uniform(0, 1)
                    sum_Ek = np.sum(Ek)
                    z = 0.0
                    for s in range(self.number_of_labels):
                        z += Ek[s] / sum_Ek

                        if z > epsilon:
                            self.image_labels[i][j] = s
                            break
            E = self.calculateGlobalEnergy()
            sum_deltaE = math.fabs(E_old - E)
            E_old = E

            if temperature > 0.1:
                temperature *= self.discount
            iterator += 1

            self.displayCurrentLabels()
            print(
                "Running iteration {}: current energy {}".format(iterator, E))
        cv2.destroyAllWindows()
        return self.image_labels


def computeLabelMeanAndVariance(number_of_labels, number_of_features,
                                features):
    gmm = mixture.GaussianMixture(
        n_components=number_of_labels,
        covariance_type='diag').fit(features.reshape(-1, number_of_features))
    return gmm.means_, gmm.covariances_

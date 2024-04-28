__author__ = "Zafeirios Fountas, Kyriacos Nikiforou, Warrick Roseboom"
__credits__ = ["David Bhowmik", "Murray Shanahan", "Anil Seth"]
__license__ = "GPLv3"
__version__ = "0.1"
__maintainer__ = "Zafeirios Fountas"
__email__ = "fountas@outlook.com"
__status__ = "Published"

import numpy as np
import numpy.matlib as mat
import pickle, json

from src.plot_experiment import plotExperiment


def euc_dist(X,Y) :
    return np.sqrt(np.sum(np.square(X-Y)))

class UpdateNetwork:
    """
    Class representing the hierarchical neural network (MLP) for updating the network.

    Attributes:
        params (dict): Parameters for the network.
        salientFeatures (dict): The array that keeps track of the accumulator - i.e. the salient features.
        time (int): The current time step.
        thres (list): The threshold values for each layer.
        pictures (list): Arrays that keep track of the pictures of the salient frames.
        distances (dict): Arrays that keep track of the distances between the salient frames.
        last (dict): Dictionary to store the last features for each layer.
        prev (dict): Dictionary to store the previous features for each layer.
        last_t (dict): Dictionary to store the last time step for each layer.
        accumulator (dict): Variable that accumulates how many salient features have been recorded.
        buff_dist (dict): Buffer of last distances to be normalized and used for the feat. threshold.
        real_time (int): The real time value.
        last_estimation (list): The last estimation values.
        states (dict): The network states.
        net_labels (list): The network labels.

    Methods:
        run(image, features, output_prob, net_states, net_labels):
            An iteration of the trial.
        plot(net_states, net_labels, real_time, last_estimation):
            Plot the experiment results.
    """

    def __init__(self, params):
        """
        Initialize the UpdateNetwork class.

        Args:
            params (dict): Parameters for the network.
        """
        self.params = params
        self.salientFeatures = {0: [], 1: [], 2: [], 3: []}
        self.time = 0
        self.thres = list(self.params['Tmax']['d2'])
        self.pictures = []
        self.distances = {0: [], 1: [], 2: [], 3: [], 'T0': [], 'T1': [], 'T2': [], 'T3': []}
        self.last = dict()
        self.prev = dict()
        self.last_t = dict()
        self.accumulator = {0: 0, 1: 0, 2: 0, 3: 0}
        self.buff_dist = {0: [], 1: [], 2: [], 3: []}
        self.real_time = 0
        self.last_estimation = []
        self.states = dict()
        self.net_labels = []

    def run(self, image, features, output_prob, net_states, net_labels):
        """
        An iteration of the trial.

        Args:
            image: The input image.
            features: The features extracted from the image.
            output_prob: The output probabilities.
            net_states: The network states.
            net_labels: The network labels.
        """
        self.string = " - Time:" + str(self.time)
        self.string += ", Feature:" + str(self.accumulator[3])

        for i in range(4):
            if self.time == 0:
                self.dist = 0.0
                self.last_t[i] = 0
                self.last[i] = mat.zeros(len(features[i]))
                self.prev[i] = mat.zeros(len(features[i]))

            if self.params['type'] == 'prev' and self.time > 0:
                self.dist = euc_dist(self.prev[i], features[i])
            elif self.params['type'] == 'last' and self.time > 0:
                self.dist = euc_dist(self.last[i], features[i])

            self.D = np.abs(self.time - self.last_t[i])
            self.Tmin = self.params['Tmin']['d2'][i]
            self.Tmax = self.params['Tmax']['d2'][i]

            self.thres[i] = self.thres[i]
            self.thres[i] -= ((self.Tmax - self.Tmin) / self.params['Ttau'][i]) * \
                             np.exp(-self.D / self.params['Ttau'][i])
            self.thres[i] += np.random.normal(0, (self.Tmax - self.Tmin) / 50.0)

            if self.time == 0 or self.dist >= self.thres[i]:
                self.last_t[i] = self.time
                self.last[i] = features[i]
                self.thres[i] = self.Tmax
                self.accumulator[i] += 1

                if i == 3:
                    self.string += "   NEW FEATURE NOW!!"
                    if self.params['visuals']:
                        self.pictures.append(image)

            self.prev[i] = features[i]
            self.distances[i].append(self.dist)
            self.distances['T' + str(i)].append(self.thres[i])

            self.salientFeatures[i].append(self.accumulator[i])

        self.now_xx = [self.accumulator[oo] for oo in self.accumulator.keys()]

        print(self.string, self.accumulator)

        self.time += 1

    def plot(self, net_states, net_labels, real_time, last_estimation):
        """
        Plot the experiment results.

        Args:
            net_states: The network states.
            net_labels: The network labels.
            real_time: The real time value.
            last_estimation: The last estimation values.

        Returns:
            fig: The plotted figure.
        """
        print(self.params['C'])
        self.last_estimation = last_estimation
        self.states = net_states
        self.net_labels = net_labels
        fig = plotExperiment(self.time, self.salientFeatures,
                             last_estimation, self.pictures,
                             self.distances, self.params, True, net_states,
                             net_labels, real_time)
        return fig

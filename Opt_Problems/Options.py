from enum import Enum

class ExtendedOptions(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c, cls))

    @classmethod
    def list_values(cls):
        return list(map(lambda c: c.value, cls))

class Datasets(ExtendedOptions):

    '''Binary Classification Datasets'''
    Australian = "australian"
    Mushroom = "mushroom"
    Phishing = "phishing"
    Sonar = "sonar"
    Gisette = "gisette"
    A9a = "a9a"
    W8a = "w8a"
    Ijcnn = "ijcnn"
    RealSim = "real-sim"

    '''Multiclass Classification Datasets'''
    MNIST = "mnist"
    CIFAR10 = "cifar10"


class StochasticApproximationType(ExtendedOptions):

    FullBatch = "Full Batch Approximation"
    MiniBatch = "Mini Batch Approximation"
    SpecifiedIndices = "Specified Indices"

class CUTESTNoiseOptions(ExtendedOptions):

    NormalNoisyGradient = "Add Normal Noise to Gradient"
    UniformNoiseInitialPoint = "Uniform Noise in Initial Point Distance"
    UniformNoiseOptimalPoint = "Uniform Noise in Optimal Point Distance"
    UniformNoiseInitialPointScaled = "Uniform Noise in Initial Point Distance Scaled"

class MachineLearningLossFunctions(ExtendedOptions):

    CrossEntropy = "Cross Entropy"
    HuberLoss = "Huber Loss"
    MSE = "Mean Squared Error"


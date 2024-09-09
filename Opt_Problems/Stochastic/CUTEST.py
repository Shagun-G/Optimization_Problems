import numpy as np
from Opt_Problems.Base import Problem
from Opt_Problems.Deterministic.CUTEST import CUTEST
from Opt_Problems.Utilities import create_rng
from Opt_Problems.Options import StochasticApproximationType, CUTESTNoiseOptions

class CUTESTStochastic(CUTEST):

    def __init__(self, name: str, noise_option : CUTESTNoiseOptions, noise_level : float = 1, optimal_point: np.ndarray = None) -> None:

        super().__init__(name = name)
        self._number_of_datapoints = float('inf')
        self.noise_option = noise_option
        self.noise_level = noise_level

        if self.noise_option in [CUTESTNoiseOptions.UniformNoiseInitialPointScaled, CUTESTNoiseOptions.UniformNoiseOptimalPoint]:
            if optimal_point is None:
                raise ValueError(f"Optimal point should be provided for {self.noise_option.value}")
            self._optimal_point = optimal_point

    def objective(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        if type is StochasticApproximationType.FullBatch:
            return super().objective(x)

        noise_term = self._noise_term_objecive(x, type, batch_size, seed, data_indices)
        return super().objective(x) + noise_term
    
    def gradient(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        if type is StochasticApproximationType.FullBatch:
            return super().gradient(x)

        noise_term = self._noise_term_gradient(x, type, batch_size, seed, data_indices)
        return super().gradient(x) + noise_term

    def _noise_term_objecive(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        noise_factor = self._generate_noise_factor(type = type, batch_size= batch_size, seed=seed, data_indices=data_indices)

        if self.noise_option is CUTESTNoiseOptions.UniformNoiseInitialPoint:
            return noise_factor * np.linalg.norm(x - self.initial_point())**2
        if self.noise_option is CUTESTNoiseOptions.UniformNoiseInitialPointScaled:
            return noise_factor * np.linalg.norm(x - self.initial_point())**2 / np.linalg.norm(self.initial_point() - self._optimal_point)**2
        if self.noise_option is CUTESTNoiseOptions.UniformNoiseOptimalPoint:
            return noise_factor * np.linalg.norm(x - self._optimal_point)**2

        raise ValueError(f"Invalid noise option: {self.noise_option}")

    def _noise_term_gradient(self, x: np.ndarray, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        noise_factor = self._generate_noise_factor(type = type, batch_size= batch_size, seed=seed, data_indices=data_indices)

        if self.noise_option is CUTESTNoiseOptions.UniformNoiseInitialPoint:
            return 2 * noise_factor * (x - self.initial_point())
        if self.noise_option is CUTESTNoiseOptions.UniformNoiseInitialPointScaled:
            return 2 * noise_factor * (x - self.initial_point()) / np.linalg.norm(self.initial_point() - self._optimal_point) ** 2
        if self.noise_option is CUTESTNoiseOptions.UniformNoiseOptimalPoint:
            return 2 * noise_factor * (x - self._optimal_point)
        if self.noise_option is CUTESTNoiseOptions.NormalNoisyGradient:
            return noise_factor

        raise ValueError(f"Invalid noise option: {self.noise_option}")

    def _generate_noise_factor(self, type: StochasticApproximationType, batch_size: int = None, seed: int = None, data_indices: list = None) -> float:

        seed_list = super().generate_stochastic_batch(type = type, batch_size= batch_size, seed=seed, data_indices=data_indices)
        noise_factor = 0
        for generate_seed in seed_list:
            rng = create_rng(generate_seed)
            if self.noise_option in [CUTESTNoiseOptions.UniformNoiseInitialPoint, CUTESTNoiseOptions.UniformNoiseInitialPointScaled, CUTESTNoiseOptions.UniformNoiseOptimalPoint]:
                noise_factor += rng.uniform(-self.noise_level, self.noise_level)
            elif self.noise_option is CUTESTNoiseOptions.NormalNoisyGradient:
                noise_factor += rng.normal(0, self.noise_level, size=(self.d, 1))

        return noise_factor / len(seed_list)
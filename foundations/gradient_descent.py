class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minimizer = init

        for _ in range(iterations):
            fx = 2 * minimizer
            minimizer = minimizer - learning_rate*fx

        return round(minimizer, 5)
        

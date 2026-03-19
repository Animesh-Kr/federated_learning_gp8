import numpy as np

class CarbonGridSimulator:
    def __init__(self):
        self.base_intensities = [100, 350, 600]
    def get_carbon_intensity(self, zone_id, current_round):
        base = self.base_intensities[zone_id % 3]
        return max(20, base + (50 * np.sin(current_round / 10.0)) + np.random.normal(0, 10))

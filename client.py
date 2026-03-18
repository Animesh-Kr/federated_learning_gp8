import flwr as fl
import torch
import random
from model import Net

ALPHA = 0.000001
BETA = 0.0000005


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainloader, device_profile):
        self.model = Net()
        self.trainloader = trainloader
        self.profile = device_profile

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        # ------------------ DROPOUT ------------------
        if random.random() < self.profile["dropout"]:
            return self.get_parameters({}), 0, {
                "compute_energy": 0,
                "communication_energy": 0,
                "total_energy": 0,
                "battery": self.profile["battery"],
                "cpu": self.profile["cpu_factor"],
                "compression": self.profile["compression"],
                "data": 0,
                "dropout": self.profile["dropout"]
            }

        self.set_parameters(parameters)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        self.model.train()

        training_steps = 0

        # ------------------ ENERGY-AWARE TRAINING ------------------
        max_steps = int(len(self.trainloader) * self.profile["cpu_factor"])

        for i, (data, target) in enumerate(self.trainloader):

            if i >= max_steps:
                break

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            training_steps += 1

        # ------------------ ENERGY CALCULATION ------------------
        cpu_cycles = training_steps * self.profile["cpu_factor"] * 1000
        compute_energy = cpu_cycles * ALPHA

        model_size = sum(p.numel() for p in self.model.parameters())
        transmitted_params = model_size * self.profile["compression"]
        communication_energy = transmitted_params * BETA

        total_energy = compute_energy + communication_energy

        metrics = {
            "compute_energy": compute_energy,
            "communication_energy": communication_energy,
            "total_energy": total_energy,
            "battery": self.profile["battery"],
            "cpu": self.profile["cpu_factor"],
            "data": len(self.trainloader.dataset),
            "dropout": self.profile["dropout"]
        }

        return self.get_parameters({}), len(self.trainloader.dataset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return 0.0, len(self.trainloader.dataset), {}
"""retinopathy_fl: strategy factory.

Maps the human-readable `strategy` name from the run config to a configured
Flower `Strategy` instance.
"""

from flwr.serverapp.strategy import FedAdam, FedAvg, FedProx, FedYogi, Strategy


def get_strategy(name: str, fraction_train: float) -> Strategy:
    """Build a Flower server-side strategy by name.

    Args:
        name: One of "FedAvg", "FedAdam", "FedYogi", "FedProx".
        fraction_train: Fraction of available clients sampled for training
            each round (forwarded to every strategy).
    """
    if name == "FedAvg":
        return FedAvg(fraction_train=fraction_train)

    if name == "FedAdam":
        return FedAdam(
            fraction_train=fraction_train,
            eta=0.001,
            eta_l=1.0,
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-9,
        )

    if name == "FedYogi":
        return FedYogi(
            fraction_train=fraction_train,
            eta=0.001,
            eta_l=1.0,
            beta_1=0.9,
            beta_2=0.99,
            tau=1e-3,
        )

    if name == "FedProx":
        return FedProx(fraction_train=fraction_train, proximal_mu=0.01)

    raise ValueError(
        f"Unknown strategy: {name!r}. "
        "Expected one of: FedAvg, FedAdam, FedYogi, FedProx."
    )

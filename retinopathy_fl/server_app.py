"""retinopathy_fl: ServerApp.

Defines the server-side federated-learning logic: builds the global model,
selects a strategy, runs the federated rounds and evaluates the resulting
global model on the centralized test set.
"""

from functools import partial

import torch
import torch.nn as nn

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from retinopathy_fl.strategy import get_strategy
from retinopathy_fl.task import filter_state_dict, get_model_type, load_centralized_dataset, test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read run config.
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    model_type: str = context.run_config["model-type"]
    strategy_name: str = context.run_config["strategy"]
    fraction_train: float = context.run_config["fraction-train"]

    print(f"Starting training for {num_rounds} rounds with initial lr={lr}")

    # Build the initial global model and the strategy.
    global_model = get_model_type(model_type)
    strategy = get_strategy(strategy_name, fraction_train)

    arrays = ArrayRecord(filter_state_dict(global_model.state_dict()))

    # Run the federated strategy for `num_rounds` rounds.
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=partial(global_evaluate, model_type=model_type),
    )

    # Save the final model to disk.
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def global_evaluate(
    server_round: int, arrays: ArrayRecord, model_type: str
) -> MetricRecord:
    """Evaluate the (aggregated) global model on the centralized test set."""
    print("=" * 10 + " GLOBAL EVALUATE " + "=" * 10)

    model = get_model_type(model_type)
    model.load_state_dict(filter_state_dict(arrays.to_torch_state_dict()), strict=False)
    model.to(DEVICE)

    # BN running stats are not aggregated across clients, so switch BN layers to
    # train mode to use batch statistics during evaluation instead of stale defaults.
    if model_type == "bn":
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.train()

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, DEVICE)

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})

"""retinopathy_fl: ServerApp.

Defines the server-side federated-learning logic: builds the global model,
selects a strategy, runs the federated rounds and evaluates the resulting
global model on the centralized test set.
"""

from functools import partial
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from retinopathy_fl.strategy import get_strategy
from retinopathy_fl.task import filter_state_dict, get_model, load_centralized_dataset, test

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    num_rounds     = context.run_config["num-server-rounds"]
    lr             = context.run_config["lr"]
    model_name     = context.run_config["model-name"]
    strategy_name  = context.run_config["strategy"]
    fraction_train = context.run_config["fraction-train"]

    print(f"Starting {num_rounds} rounds | model={model_name} | "
          f"strategy={strategy_name} | lr={lr}")

    global_model = get_model(model_name)
    strategy     = get_strategy(strategy_name, fraction_train)
    arrays       = ArrayRecord(filter_state_dict(global_model.state_dict()))

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=partial(global_evaluate, model_name=model_name),
    )

    print("\nSaving final model...")
    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    print("✅ Saved: final_model.pt")


def global_evaluate(
    server_round: int, arrays: ArrayRecord, model_name: str
) -> MetricRecord:
    print("=" * 10 + f" GLOBAL EVALUATE round {server_round} " + "=" * 10)

    model = get_model(model_name)
    model.load_state_dict(filter_state_dict(arrays.to_torch_state_dict()), strict=False)
    model.to(DEVICE)

    testloader = load_centralized_dataset()
    loss, acc, qwk = test(model, testloader, DEVICE)

    print(f"  loss={loss:.4f}  acc={acc:.4f}  QWK={qwk:.4f}")
    return MetricRecord({"loss": loss, "accuracy": acc, "qwk": qwk})

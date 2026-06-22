"""retinopathy_fl: ClientApp.

Defines the federated-learning client logic: local training and local
evaluation of the diabetic-retinopathy classifier on a client's data shard.
"""

import torch

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from retinopathy_fl.task import get_model_type, load_data, test, train_local_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = ClientApp()


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on this client's local data partition."""
    model_type = context.run_config["model-type"]
    local_epochs = context.run_config["local-epochs"]
    lr = msg.content["config"]["lr"]

    # Build model and load the global parameters received from the server.
    model = get_model_type(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    model.to(DEVICE)

    # Load this client's data partition.
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    print(f"[Client {partition_id}] training on {DEVICE} "
          f"(cuda available: {torch.cuda.is_available()})")

    # Train locally.
    avg_loss = train_local_model(
        model, trainloader, epochs=local_epochs, lr=lr, device=DEVICE
    )

    # Move parameters back to CPU before sending them over the wire.
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

    content = RecordDict(
        {
            "arrays": ArrayRecord(state_dict),
            "metrics": MetricRecord(
                {
                    "train_loss": float(avg_loss),
                    "num-examples": len(trainloader.dataset),
                }
            ),
        }
    )
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global model on this client's local validation data."""
    print("=" * 10 + " LOCAL EVALUATE " + "=" * 10)

    model_type = context.run_config["model-type"]

    model = get_model_type(model_type)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    model.to(DEVICE)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    eval_loss, eval_acc = test(model, valloader, DEVICE)

    metrics = {
        "eval_loss": float(eval_loss),
        "eval_acc": float(eval_acc),
        "num-examples": len(valloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)

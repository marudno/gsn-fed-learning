"""retinopathy_fl: ClientApp.

Defines the federated-learning client logic: local training and local
evaluation of the diabetic-retinopathy classifier on a client's data shard.
"""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from retinopathy_fl.task import get_model, load_data, test, train_local_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = ClientApp()

@app.train()
def train(msg: Message, context: Context) -> Message:
    model_name      = context.run_config["model-name"]
    local_epochs    = context.run_config["local-epochs"]
    partition_type  = context.run_config.get("partition-type", "iid")
    alpha           = float(context.run_config.get("dirichlet-alpha", 0.5))
    lr              = msg.content["config"]["lr"]

    model = get_model(model_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    model.to(DEVICE)

    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions,
                               partition_type=partition_type, alpha=alpha)

    print(f"[Client {partition_id}] training on {DEVICE}, "
          f"partition={partition_type}, alpha={alpha}")

    avg_loss = train_local_model(
        model, trainloader, epochs=local_epochs, lr=lr, device=DEVICE
    )

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    content = RecordDict({
        "arrays":  ArrayRecord(state_dict),
        "metrics": MetricRecord({
            "train_loss":   float(avg_loss),
            "num-examples": len(trainloader.dataset),
        }),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    print("=" * 10 + " LOCAL EVALUATE " + "=" * 10)
    model_name      = context.run_config["model-name"]
    partition_type  = context.run_config.get("partition-type", "iid")
    alpha           = float(context.run_config.get("dirichlet-alpha", 0.5))

    model = get_model(model_name)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    model.to(DEVICE)

    partition_id   = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader   = load_data(partition_id, num_partitions,
                               partition_type=partition_type, alpha=alpha)

    eval_loss, eval_acc, eval_qwk = test(model, valloader, DEVICE)

    content = RecordDict({"metrics": MetricRecord({
        "eval_loss":    float(eval_loss),
        "eval_acc":     float(eval_acc),
        "eval_qwk":     float(eval_qwk),
        "num-examples": len(valloader.dataset),
    })})
    return Message(content=content, reply_to=msg)

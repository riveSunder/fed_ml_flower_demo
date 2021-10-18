import argparse

import flwr as fl

from client import get_data, MLPClient

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Use the last 5k training examples as a validation set
    # The `evaluate` function will be called after every round
    def evaluate(weights: fl.common.Weights): 


        loss, _, accuracy_dict = model.evaluate(weights)

        return loss, accuracy_dict

    return evaluate


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--rounds", type=int, default=3,\
            help="number of rounds to train")

    args = parser.parse_args()

    model = MLPClient(split="val")
    strategy = fl.server.strategy.FedAvg(
        eval_fn=get_eval_fn(model),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", strategy=strategy, config={"num_rounds": args.rounds})


    #fl.server.start_server("localhost:8080",config={"num_rounds": 30})


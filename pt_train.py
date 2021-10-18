import argparse

from pt_client import PTMLPClient
import torch

import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-r", "--rounds", type=int, default=3,\
            help="number of rounds to train")

    args = parser.parse_args()

    torch.random.manual_seed(42)
    model = PTMLPClient(split="all") 

    t0 = time.time()
    print("start")
    for decadal_epochs in range(args.rounds):

        loss, _, accuracy = model.evaluate(model.get_parameters())
        print(f" {10*decadal_epochs} epochs, time: {time.time() - t0}")
        print(f"loss: {loss:.3f}, accuracy: {accuracy['accuracy']}")
        model.fit(model.get_parameters(), epochs=10)

    loss, _, accuracy = model.evaluate(model.get_parameters())
    print(f" {10*decadal_epochs} epochs, time: {time.time() - t0}")
    print(f"loss: {loss:.3f}, accuracy: {accuracy['accuracy']}")

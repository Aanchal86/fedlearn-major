
import argparse
import json

from torch.utils.data import Dataset

from federated.networking.dataset_provider import get_client_dataset
from federated.networking.http_client import FederatedHTTPClient
from federated.networking.worker import RemoteClientWorker


class FederatedClientRunner:
    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        server_url: str = "http://127.0.0.1:8000",
        seed: int = 42,
    ):
        self.client = FederatedHTTPClient(
            client_id=client_id,
            server_url=server_url,
        )

        self.worker = RemoteClientWorker(
            client_id=client_id,
            dataset=dataset,
            seed=seed,
        )

    def run(
        self,
        experiment_id: int,
        round_number: int,
        register: bool = True,
        max_wait_seconds: float = 60.0,
    ) -> dict:
        if experiment_id <= 0:
            raise ValueError("Experiment ID must be positive")

        if round_number <= 0:
            raise ValueError("Round number must be positive")

        if register:
            self.client.register()

        assignment = self.client.wait_for_assignment(
            experiment_id=experiment_id,
            round_number=round_number,
            max_wait_seconds=max_wait_seconds,
        )

        if assignment.experiment_id != experiment_id:
            raise ValueError("Experiment ID mismatch")

        if assignment.round_number != round_number:
            raise ValueError("Round number mismatch")

        update = self.worker.execute(assignment)

        if update.experiment_id != experiment_id:
            raise ValueError("Update experiment ID mismatch")

        if update.round_number != round_number:
            raise ValueError("Update round number mismatch")

        return self.client.submit_update(update)

    def run_rounds(
        self,
        experiment_id: int,
        num_rounds: int,
        start_round: int = 1,
        register: bool = True,
        max_wait_seconds: float = 60.0,
    ) -> list[dict]:
        if experiment_id <= 0:
            raise ValueError("Experiment ID must be positive")

        if num_rounds <= 0:
            raise ValueError("Number of rounds must be positive")

        if start_round <= 0:
            raise ValueError("Starting round must be positive")

        if max_wait_seconds <= 0:
            raise ValueError("Maximum wait must be positive")

        if register:
            self.client.register()

        results = []

        for round_number in range(
            start_round,
            start_round + num_rounds,
        ):
            print(
                f"Client {self.client.client_id}: "
                f"starting round {round_number}"
            )

            result = self.run(
                experiment_id=experiment_id,
                round_number=round_number,
                register=False,
                max_wait_seconds=max_wait_seconds,
            )

            results.append({
                "round_number": round_number,
                "response": result,
            })

            print(
                f"Client {self.client.client_id}: "
                f"completed round {round_number}"
            )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="FedLearn standalone federated client"
    )

    parser.add_argument("--client-id", required=True)
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--round", type=int, default=1)

    parser.add_argument(
        "--num-rounds",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--dataset",
        choices=["fashion_mnist", "pneumonia_mnist"],
        default="pneumonia_mnist",
    )

    parser.add_argument("--num-clients", type=int, default=5)

    parser.add_argument(
        "--partition-type",
        choices=["iid", "dirichlet"],
        default="iid",
    )

    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.5,
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-rate", type=float, default=0.0)
    parser.add_argument("--noisy-client-id", type=int)
    parser.add_argument("--max-wait-seconds", type=float, default=60.0)
    parser.add_argument("--skip-registration", action="store_true")

    args = parser.parse_args()

    if args.num_rounds <= 0:
        parser.error("--num-rounds must be positive")

    if args.round <= 0:
        parser.error("--round must be positive")

    dataset = get_client_dataset(
        dataset_name=args.dataset,
        client_id=args.client_id,
        num_clients=args.num_clients,
        partition_type=args.partition_type,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed,
        noise_rate=args.noise_rate,
        noisy_client_id=args.noisy_client_id,
    )

    print(
        f"Client {args.client_id}: "
        f"{len(dataset)} local training samples"
    )

    runner = FederatedClientRunner(
        client_id=args.client_id,
        dataset=dataset,
        server_url=args.server_url,
        seed=args.seed + int(args.client_id.removeprefix("client_")) - 1,
    )

    if args.num_rounds == 1:
        result = runner.run(
            experiment_id=args.experiment_id,
            round_number=args.round,
            register=not args.skip_registration,
            max_wait_seconds=args.max_wait_seconds,
        )
    else:
        result = runner.run_rounds(
            experiment_id=args.experiment_id,
            num_rounds=args.num_rounds,
            start_round=args.round,
            register=not args.skip_registration,
            max_wait_seconds=args.max_wait_seconds,
        )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

import dataclasses
import logging
import pathlib
import sys

import env as _env
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import saver as _saver
from tqdm import tqdm
import tyro

# Detect if we're in a server/Docker environment (no TTY)
tqdm_is_server = not sys.stderr.isatty()

# Custom tqdm class that prints progress bars in Docker/non-TTY environments
class ServerTqdm(tqdm):
    def update(self, n=1):
        result = super().update(n)
        if result and tqdm_is_server:
            print(f"{self}\n", flush=True)
        return result

    def display(self, msg=None, pos=None):
        if not tqdm_is_server:
            return super().display(msg, pos)
        return True


@dataclasses.dataclass
class Args:
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")

    task: str = "gym_aloha/AlohaTransferCube-v0"
    seed: int = 0

    action_horizon: int = 10

    host: str = "0.0.0.0"
    port: int = 8000

    display: bool = False


def main(args: Args) -> None:
    runtime = _runtime.Runtime(
        environment=_env.AlohaSimEnvironment(
            task=args.task,
            seed=args.seed,
        ),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=_websocket_client_policy.WebsocketClientPolicy(
                    host=args.host,
                    port=args.port,
                ),
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[
            _saver.VideoSaver(args.out_dir),
        ],
        max_hz=50,
    )

    runtime.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)

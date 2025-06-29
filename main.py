import json
import logging
import time
from argparse import ArgumentParser
from pathlib import Path

from src.batcher import ContinousBatcher, SynchronousBatcher
from src.config import BenchmarkConfig
from src.utils import get_dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main(config_path: Path, output: Path):
    config = BenchmarkConfig.model_validate_json(config_path.read_text())
    if config.continous_batching:
        batcher = ContinousBatcher(config)
    else:
        batcher = SynchronousBatcher(config)

    dataset = get_dataset(config.dataset_size, batcher.tokenizer)
    generations = batcher(dataset)

    stats = batcher.stats.model_dump_json()
    if output is not None:
        stats["generations"] = generations
        output.write_text(json.dumps(batcher.stats))
    batcher.stats.print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=Path, help="Path to config file", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file to write stats to",
        required=False,
    )
    args = parser.parse_args()
    main(args.config, args.output)

import json
import sys

from tabulate import tabulate

from src.stats import Stats


def main():
    if len(sys.argv) != 3:
        print(
            f"Expected command: `python {sys.argv[0]} path/to/stats1.json path/to/stats2.json`"
        )
        print(f"Got command: `python {' '.join(sys.argv)}`")
        return
    wrong = 0
    correct = 0
    with open(sys.argv[1]) as sync_fin, open(sys.argv[2]) as cont_fin:
        results1 = json.load(sync_fin)
        results2 = json.load(cont_fin)

    assert len(results1["generations"]) == len(results2["generations"])
    n = len(results1["generations"])
    for s, c in zip(results1.pop("generations"), results2.pop("generations")):
        if s != c:
            wrong += 1
        else:
            correct += 1
    print("Correct %s Wrong %s" % (correct, wrong))

    stats1 = Stats.model_validate_json(json.dumps(results1))
    stats2 = Stats.model_validate_json(json.dumps(results2))
    latencies1 = [
        e - s for e, s in zip(stats1.sample_end_times, stats1.sample_start_times)
    ]
    latencies2 = [
        e - s for e, s in zip(stats2.sample_end_times, stats2.sample_start_times)
    ]
    data = [
        ["Run time (s)", stats1.run_time, stats2.run_time],
        ["Prefill tokens (tok)", stats1.prefill_tokens, stats2.prefill_tokens],
        ["Generated tokens (tok)", stats1.generated_tokens, stats2.generated_tokens],
        [
            "Generation speed (tok/s)",
            stats1.generated_tokens / stats1.run_time,
            stats2.generated_tokens / stats2.run_time,
        ],
        [
            "Latency (global) (s)",
            sum(stats1.sample_end_times) / n - stats1.start_time,
            sum(stats2.sample_end_times) / n - stats2.start_time,
        ],
        ["Latency (sample) (s)", sum(latencies1) / n, sum(latencies2) / n],
    ]

    # Per sample latency from global start: {sum(self.sample_end_times) / n - self.start_time} s
    # Per sample latency from sample start: {sum([e - s for e, s in zip(self.sample_end_times, self.sample_start_times)]) / n}
    headers = ["Metric", sys.argv[1], sys.argv[2]]
    print(tabulate(data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    main()

# Add the positional arguments

# Parse the arguments
# break

import sys
import json
from pathlib import Path


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
        sync_results = json.load(sync_fin)
        cont_results = json.load(cont_fin)

    for s, c in zip(sync_results, cont_results):
        assert s["prefix"] == c["prefix"]
        if s["generation"] != c["generation"]:
            wrong += 1
        else:
            correct += 1
    print("Correct %s Wrong %s" % (correct, wrong))


if __name__ == "__main__":
    main()

# Add the positional arguments

# Parse the arguments
# break

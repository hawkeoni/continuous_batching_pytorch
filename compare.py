import sys
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
    for s, c in zip(sync_results, cont_results):
        assert s["prefix"] == c["prefix"]
        if s["generation"] != c["generation"]:
            # print(f"""`{s["generation"]}`""")
            # print(f"""`{c["generation"]}`""")
            # print("------")
            wrong += 1
        else:
            correct += 1
    print("Correct %s Wrong %s" % (correct, wrong))


if __name__ == "__main__":
    main()

# Add the positional arguments

# Parse the arguments
# break

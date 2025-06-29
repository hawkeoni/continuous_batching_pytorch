import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Process some integers.")

# Add the positional arguments
parser.add_argument("first_argument", type=str, help="The first argument")
parser.add_argument("second_argument", type=str, help="The second argument")

# Parse the arguments
args = parser.parse_args()

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
        # break
print("Correct %s Wrong %s" % (correct, wrong))

import random


def preprocess(path, shuffle=True):
    with open(path) as f:
        lines = "".join([l for l in f.readlines()]).strip()
    examples = lines.split("\n\n")
    if shuffle:
        random.shuffle(examples)
    x = []
    y = []
    for ex in examples:
        features = ex.split("\n")
        x.append(features[:-1])
        y.append(1 if features[-1] == "H" else 0)
    # print(y)
    return x, y


if __name__ == "__main__":
    # execute only if run as a script
    FILE = "data/train.txt"
    preprocess(FILE)

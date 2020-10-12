from deephyper.benchmark import HpProblem

Problem = HpProblem()

sp = {}

# Problem.add_dim("lr", (0.0009, 0.1))


for i in range(1, 18):
    Problem.add_dim(f"u{i}", (32, 128))
    Problem.add_dim(f"a{i}", [None, "relu", "sigmoid", "tanh"])

    sp[f"u{i}"] = 64
    sp[f"a{i}"] = "relu"
Problem.add_starting_point(**sp)
Problem.add_starting_point(
    **{
        "a1": None,
        "a10": "relu",
        "a11": "relu",
        "a12": "tanh",
        "a13": "tanh",
        "a14": "tanh",
        "a15": "relu",
        "a16": "relu",
        "a17": "tanh",
        "a2": "sigmoid",
        "a3": None,
        "a4": "sigmoid",
        "a5": "tanh",
        "a6": None,
        "a7": "tanh",
        "a8": "relu",
        "a9": None,
        "u1": 56,
        "u10": 102,
        "u11": 126,
        "u12": 47,
        "u13": 48,
        "u14": 57,
        "u15": 51,
        "u16": 91,
        "u17": 63,
        "u2": 59,
        "u3": 66,
        "u4": 53,
        "u5": 65,
        "u6": 33,
        "u7": 126,
        "u8": 117,
        "u9": 45,
    }
)

if __name__ == "__main__":
    print(Problem)

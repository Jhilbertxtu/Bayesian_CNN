import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import re
import numpy as np
plt.rc('font', family='serif', size=16)
plt.rcParams.update({'xtick.labelsize': 16, 'ytick.labelsize': 16, 'axes.labelsize': 16})

# change for given number of tasks
task = "1"


def load_data():
    os.chdir("/home/felix/Dropbox/publications/Bayesian_CNN_MCVI/results/")
    with open("diagnostics_{}.txt".format(task), 'r') as file:
        acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    print(acc)

    train = acc[1::2]
    valid = acc[0::2]
    return np.array(train).astype(np.float32), np.array(valid).astype(np.float32)


f = plt.figure(figsize=(10, 8))

train, valid = load_data()

print(valid)

plt.plot(valid, "--", label=r"Validation, prior: $U(a, b)$", color='maroon')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
x_ticks = range(len(valid))
plt.xticks(x_ticks[19::20], map(lambda x: x+1, x_ticks[19::20]))

f.suptitle("Accuracy after training for 200 epochs")
plt.legend()

plt.savefig("results_{}.png".format(task))


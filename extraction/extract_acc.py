import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import re
import numpy as np

# change for given number of tasks
task = "LeNet"


def load_data():
    os.chdir("/home/felix/Dropbox/publications/Bayesian_CNN_MCVI/results/")
    with open("diagnostics_{}.txt".format(task), 'r') as file:
        acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
    #acc = list(map(lambda x: x.split(" ")[-1], re.findall(r"(acc: \d.\d+)", file)))

    #print(re.findall(r"(acc: \d.\d+)", file))
    print(acc)

    train = acc[0::3]
    valid = acc[1::3]
    return np.array(train).astype(np.float32), np.array(valid).astype(np.float32)


f = plt.figure(figsize=(10, 8))
name_ext = task

train, valid = load_data()

print(valid)

plt.plot(valid, "--", label=r"Validation, prior: $\mathcal{U}(a, b)$", color="#34d536")
#plt.plot(i, valid, "--", label=r"Validation, prior: $q(w \mid \theta)$", color="#d534d3")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
x_ticks = range(len(valid))
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))
#for label in f.ax.xaxis.get_ticklabels()[::5]:
#    label.set_visible(True)
f.suptitle("Accuracy after training for 100 epochs")
plt.legend()

plt.savefig("results_{}.png".format(task))


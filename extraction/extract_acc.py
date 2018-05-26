import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import re
import numpy as np

# change for given net
test_type = "LeNet"


def load_data(basename):
    file = open("diagnostics_{}.txt".format(test_type)).read()
    acc = list(map(lambda x: x.split(" ")[-1], re.findall(r"(acc: \d.\d+)", file)))

    print(re.findall(r"(acc: \d.\d+)", file))

    train = acc[0::3]
    valid = acc[1::3]
    return np.array(train).astype(np.float32), np.array(valid).astype(np.float32)


f = plt.figure(figsize=(10, 8))
name_ext = test_type

train, valid = load_data("/home/felix/Dropbox/publications/Bayesian_CNN_MCVI/results/")

print(valid)

#plt.plot(i, train, label=r"Train, prior: $q(w \mid \theta)$", color="#9c209b")
#plt.plot(i, valid, "--", label=r"Validation, prior: $q(w \mid \theta)$", color="#d534d3")
#plt.plot(i, MAP, "--", label=r"MAP, prior: $q(w \mid \theta)$", color="#e273e1")

#train, valid, MAP = load_data("domain", i, name_ext)

#plt.plot(train, label=r"Train, prior: $\mathcal{U}(a, b)$", color="#209c22")
plt.plot(valid, "--", label=r"Validation, prior: $\mathcal{U}(a, b)$", color="#34d536")
#plt.plot(i, MAP, "--", label=r"MAP, prior: $\mathcal{U}(a, b)$", color="#73e275")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.xticks(range(len(valid)), map(lambda x: x+1, range(len(train))))
#for label in f.ax.xaxis.get_ticklabels()[::5]:
#    label.set_visible(True)
f.suptitle("Accuracy after training for 100 epochs")
plt.legend()


plt.savefig("figures/" + name_ext + "result.pdf")


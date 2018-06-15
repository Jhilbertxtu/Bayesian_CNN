import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")
import re
import numpy as np
plt.rc('font', family='serif', size=32)
plt.rcParams.update({'xtick.labelsize': 32, 'ytick.labelsize': 32, 'axes.labelsize': 32})

# change for given number of tasks
os.chdir("/home/felix/Dropbox/publications/Bayesian_CNN_MCVI/results/")

with open("diagnostics_1.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
print(acc)

train_1 = acc[0::2]
valid_1 = acc[1::2]

train_1 = np.array(train_1).astype(np.float32)
valid_1 = np.array(valid_1).astype(np.float32)

with open("diagnostics_2.txt", 'r') as file:
    acc = re.findall(r"'acc':\s+tensor\((.*?)\)", file.read())
print(acc)

train_2 = acc[0::2]
valid_2 = acc[1::2]

train_2 = np.array(train_2).astype(np.float32)
valid_2 = np.array(valid_2).astype(np.float32)


f = plt.figure(figsize=(20, 16))


print(valid_1)
print(valid_2)

plt.plot(valid_1, label=r"Validation CIFAR-10, prior: $U(a, b)$", color='maroon')
plt.plot(valid_2, label=r"Validation MNIST, prior: $U(a, b)$", color='darkblue')


plt.xlabel("Epochs")
plt.ylabel("Accuracy")
x_ticks = range(len(valid_1))
plt.xticks(x_ticks[9::10], map(lambda x: x+1, x_ticks[9::10]))

plt.legend(fontsize=28)

plt.savefig("results_CNN.png")


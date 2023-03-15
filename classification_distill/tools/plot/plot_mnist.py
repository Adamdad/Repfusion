import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

with open('cls/tools/plot/mnist_S.npy','rb') as f:
    Ss = np.load(f)
with open('cls/tools/plot/mnist_rank.npy','rb') as f:
    ranks = np.load(f)

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "Times New Roman"
ts= [0,1,2, 5, 8, 10,20,50,300,500, 900,999]
plt.figure(figsize=(8,6))
for S, t in zip(Ss,ts):
    if t in [0, 2,10, 300,500, 900,999]:
        plt.plot(S, label=f"t={t}")
plt.yscale('symlog')
plt.legend(fontsize=20)
plt.xlabel("Singular Values Index",fontsize=32)
plt.ylabel("Singular Values",fontsize=32)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
# plt.show()
plt.tight_layout()
plt.savefig("Singular Value_mnist.pdf")


plt.figure(figsize=(8,6))
# for ranks, t in zip(Ss,[0,10,50,300,500, 900]):
# plt.plot(ts[:6],ranks[:6], 'o-',linewidth=3)
# ts = np.array(ts)+0.0000001
plt.plot(ts,ranks, 'o-')

# plt.legend(fontsize=16)
plt.xlabel("TimeStep",fontsize=32)
plt.ylabel("Effective rank",fontsize=32)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
plt.xlim((-0.1,1100))
# plt.show()
plt.xscale("symlog")
plt.tight_layout()
plt.savefig("Effective rank_mnist.pdf")
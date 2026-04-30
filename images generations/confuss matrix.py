import matplotlib.pyplot as plt
import numpy as np

cm = np.array([
    [850,20,10,15,5],
    [30,900,20,10,5],
    [25,30,820,40,15],
    [10,15,30,760,20],
    [5,10,20,25,700]
])
labels = ["Normal","DoS","Fuzzy","RPM","Gear"]
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(cm)
ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels)
ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i,j], ha='center', va='center')
fig.colorbar(im)
ax.set_title("Figure 6: Confusion matrix (example)")
plt.savefig("figure6_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()
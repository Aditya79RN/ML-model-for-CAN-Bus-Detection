import matplotlib.pyplot as plt
models = ["Random Forest","SVM","Gradient Boosting"]
precision = [0.66,0.63,0.69]
recall = [0.67,0.62,0.68]
x = range(len(models))
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, precision, marker='o', label='Precision')
ax.plot(x, recall, marker='s', label='Recall')
ax.set_xticks(x); ax.set_xticklabels(models)
ax.set_ylim(0,1); ax.set_ylabel("Score")
ax.legend(); ax.set_title("Precision vs Recall")
plt.savefig("precision_vs_recall.png", dpi=300, bbox_inches='tight')
plt.close()
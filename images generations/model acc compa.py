import matplotlib.pyplot as plt
models = ["Random Forest","SVM","Gradient Boosting"]
acc = [0.68, 0.64, 0.70]
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(models, acc)
ax.set_ylim(0,1)
ax.set_ylabel("Accuracy")
ax.set_title("Figure 5: Model accuracy comparison")
plt.savefig("figure5_model_accuracy.png", dpi=300, bbox_inches='tight')
plt.close()
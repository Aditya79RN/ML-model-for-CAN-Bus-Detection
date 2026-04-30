import matplotlib.pyplot as plt
classes = ["DoS","Fuzzy","Gear","RPM","Normal"]
values = [3665771, 3838860, 4443142, 4621702, 988987]
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(classes, values)
ax.set_ylabel("Number of CAN frames")
ax.set_title("Figure 4: Dataset distribution")
plt.savefig("figure4_dataset_distribution.png", dpi=300, bbox_inches='tight')
plt.close()
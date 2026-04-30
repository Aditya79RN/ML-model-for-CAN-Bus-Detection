import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(9,5))
ax.axis('off')
boxes = [
    ("Vehicle CAN Network", 0.02, 0.72),
    ("CAN Message\nCapture", 0.02, 0.48),
    ("Preprocessing\n(hex→int, cleaning)", 0.02, 0.24),
    ("Feature Engineering\n(payload→features)", 0.46, 0.48),
    ("ML Model\n(training & testing)", 0.46, 0.24),
    ("Intrusion Detection\n(Alert & Classification)", 0.78, 0.24)
]
for text, x, y in boxes:
    ax.add_patch(Rectangle((x,y),0.42,0.18,fill=False,linewidth=1.2))
    ax.text(x+0.02, y+0.09, text, fontsize=10, va='center')

arrows = [((0.23,0.72),(0.23,0.66)), ((0.23,0.56),(0.23,0.46)),
          ((0.44,0.36),(0.46,0.56)), ((0.68,0.36),(0.78,0.36))]
for (x1,y1),(x2,y2) in arrows:
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='->', mutation_scale=14))

ax.set_title("Figure 1: System architecture of the ML-based CAN intrusion detection system")
plt.savefig("figure1_system_architecture.png", dpi=300, bbox_inches='tight')
plt.close()
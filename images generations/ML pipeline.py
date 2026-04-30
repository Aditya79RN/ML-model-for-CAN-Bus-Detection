import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Create figure
fig, ax = plt.subplots(figsize=(6,10))

# Function to draw pipeline box
def draw_box(text, x, y):
    box = FancyBboxPatch(
        (x, y),
        3, 0.8,
        boxstyle="round,pad=0.3",
        edgecolor="black",
        linewidth=1.5,
        facecolor="#E8F0FE"
    )
    ax.add_patch(box)
    ax.text(x+1.5, y+0.4, text, ha='center', va='center', fontsize=11)

# Draw ML pipeline boxes (top to bottom)
draw_box("Raw CAN Bus Data\n(CSV / Log Files)", 1.5, 8)
draw_box("Data Cleaning\nRemove Missing / Corrupted Frames", 1.5, 6.5)
draw_box("Feature Extraction\nID, DLC, Timestamp, Payload", 1.5, 5)
draw_box("Train/Test Split\n80% Training / 20% Testing", 1.5, 3.5)
draw_box("Model Training\nRandom Forest Classifier", 1.5, 2)
draw_box("Prediction & Detection\nNormal vs Attack", 1.5, 0.5)

# Draw arrows
for y in [7.3, 5.8, 4.3, 2.8, 1.3]:
    ax.arrow(
        3, y,
        0, -0.6,
        head_width=0.2,
        head_length=0.2,
        fc="black",
        ec="black"
    )

# Remove axes
ax.axis('off')

# Save figure
plt.savefig("figure3_ml_pipeline.png", dpi=300, bbox_inches='tight')

# Show plot
plt.show()
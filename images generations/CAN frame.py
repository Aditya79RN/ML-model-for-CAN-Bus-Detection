import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(10,1.8))
ax.axis('off')
parts = ["Start","Arbitration ID","Control","DLC","Data Payload (0-8 bytes)","CRC","ACK","End"]
x = 0.01
w = 0.12
for p in parts:
    ax.add_patch(Rectangle((x,0.05), w, 0.9, fill=False))
    ax.text(x+0.005,0.5,p, fontsize=9)
    x += w + 0.01
ax.set_title("Figure 2: CAN data frame structure (simplified)")
plt.savefig("figure2_can_frame.png", dpi=300, bbox_inches='tight')
plt.close()

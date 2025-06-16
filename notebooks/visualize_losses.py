import json
import matplotlib.pyplot as plt
import numpy as np

# Load the checkpoint data
with open('models/checkpoint_state.json', 'r') as f:
    data = json.load(f)

train_losses = data['train_losses']
val_losses = data['val_losses']

# Create figure with appropriate size
plt.figure(figsize=(12, 6))

# Create x-axis values (steps)
steps = np.arange(len(train_losses)) * 1000  # Assuming 1000 steps between checkpoints

# Plot both loss curves
plt.plot(steps, train_losses, label='Training Loss', linewidth=2, color='blue', alpha=0.8)
plt.plot(steps, val_losses, label='Validation Loss', linewidth=2, color='orange', alpha=0.8)

# Customize the plot
plt.xlabel('Training Steps', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss Over Time', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3, linestyle='--')

# Add some statistics
final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]
min_train_loss = min(train_losses)
min_val_loss = min(val_losses)

# Add text box with final losses
textstr = f'Final Train Loss: {final_train_loss:.4f}\nFinal Val Loss: {final_val_loss:.4f}\n' \
          f'Min Train Loss: {min_train_loss:.4f}\nMin Val Loss: {min_val_loss:.4f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('models/loss_curves.png', dpi=300, bbox_inches='tight')
print("Chart saved to models/loss_curves.png")

# Also display the plot
plt.show() 
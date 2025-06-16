import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the checkpoint data
with open('models/checkpoint_state.json', 'r') as f:
    data = json.load(f)

train_losses = data['train_losses']
val_losses = data['val_losses']

# Create x-axis values (steps)
steps = list(range(0, len(train_losses) * 1000, 1000))

# Create the interactive plot
fig = go.Figure()

# Add training loss trace
fig.add_trace(go.Scatter(
    x=steps,
    y=train_losses,
    mode='lines',
    name='Training Loss',
    line=dict(color='blue', width=2),
    hovertemplate='Step: %{x}<br>Training Loss: %{y:.4f}<extra></extra>'
))

# Add validation loss trace
fig.add_trace(go.Scatter(
    x=steps,
    y=val_losses,
    mode='lines',
    name='Validation Loss',
    line=dict(color='orange', width=2),
    hovertemplate='Step: %{x}<br>Validation Loss: %{y:.4f}<extra></extra>'
))

# Calculate statistics
final_train_loss = train_losses[-1]
final_val_loss = val_losses[-1]
min_train_loss = min(train_losses)
min_val_loss = min(val_losses)

# Update layout
fig.update_layout(
    title=dict(
        text='Training and Validation Loss Over Time',
        font=dict(size=20)
    ),
    xaxis=dict(
        title=dict(text='Training Steps', font=dict(size=14)),
        gridcolor='rgba(128, 128, 128, 0.2)'
    ),
    yaxis=dict(
        title=dict(text='Loss', font=dict(size=14)),
        gridcolor='rgba(128, 128, 128, 0.2)'
    ),
    hovermode='x',
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        bgcolor='rgba(255, 255, 255, 0.8)',
        bordercolor='rgba(0, 0, 0, 0.2)',
        borderwidth=1
    ),
    plot_bgcolor='white',
    width=1200,
    height=600
)

# Add annotations with statistics
fig.add_annotation(
    xref="paper", yref="paper",
    x=0.02, y=0.98,
    xanchor='left', yanchor='top',
    text=f"Final Train Loss: {final_train_loss:.4f}<br>" +
         f"Final Val Loss: {final_val_loss:.4f}<br>" +
         f"Min Train Loss: {min_train_loss:.4f}<br>" +
         f"Min Val Loss: {min_val_loss:.4f}",
    showarrow=False,
    font=dict(size=12),
    bgcolor='rgba(255, 228, 181, 0.8)',
    bordercolor='rgba(0, 0, 0, 0.2)',
    borderwidth=1,
    borderpad=10
)

# Save as HTML
fig.write_html('models/loss_curves_interactive.html')
print("Interactive chart saved to models/loss_curves_interactive.html")

# Also save as static image if plotly-kaleido is installed
try:
    fig.write_image('models/loss_curves_plotly.png', width=1200, height=600, scale=2)
    print("Static Plotly chart saved to models/loss_curves_plotly.png")
except:
    print("Note: Install plotly-kaleido to save static images from Plotly")

# Show the plot
fig.show() 
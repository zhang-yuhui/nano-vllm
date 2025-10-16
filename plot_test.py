import matplotlib.pyplot as plt


def plot_time_chart(data: list[list[float]], fig_name: str = "timeline"):
    fig, ax = plt.subplots(figsize=(10, 6))
    assert len(data[0]) == 3, "len is" + str(len(data[0]))
    # Define segments
    color = ("blue", "green")
    data = [[[a[0], a[1]-a[0], color[0]], [a[1], a[2]-a[1], color[1]]] for a in data ]

    # Plot each bar
    for i, bars in enumerate(data):
        for bar in bars:
            ax.barh(i, width=bar[1], left=bar[0], height=1, color=bar[2], edgecolor='black')

    ax.set_yticks(range(len(data)))
    ax.set_ylabel("Request")
    ax.set_xlabel('Time')
    ax.set_title('Segmented Horizontal Bar Chart')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save instead of show
    plt.savefig(fig_name + '.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
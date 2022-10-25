# Create the CircosPlot object: c
import networkx as nx
from nxviz import CircosPlot
import matplotlib.pyplot as plt
c = CircosPlot(G, node_color = 'bipartite', node_grouping = 'bipartite', node_order = 'centrality')

# Draw c to the screen
c.draw()
# Display the plot
plt.show()

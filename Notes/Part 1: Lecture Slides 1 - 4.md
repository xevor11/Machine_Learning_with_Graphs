## Introduction to Graph Theory

The lecture gives an overview of basic terms and definitions used in graph theory. 
Why should you learn graphs? The impact! The impact of graphs application is proven for Social networking, Drug design, AI reasoning.

One of the applications of Social Networks is the Facebook social graph which shows that all users have only 4-degrees of separation


Another interesting example of application: predict side-effects of drugs (when patients take multiple drugs at the same time, what are the potential side effects? 
It represents a link prediction task).

Analyzing networks through tasks:

1. Node classification (Predict the type/color of a given node)
2. Link prediction (Predict whether two nodes are linked)
3. Identify densely linked clusters of nodes (Community detection)
4. Network similarity (Measure similarity of two nodes/networks)

Key Definitions:

* A network is a collection of objects (nodes) where some pairs of objects are connected by links (edges). 
* Types of networks: Directed vs. undirected, weighted vs. unweighted, connected vs. disconnected
* Types of networks: Directed vs. undirected, weighted vs. unweighted, connected vs. disconnected
* The maximum number of edges on N nodes (for undirected graph) is
  - $E_{max}$ $=$ $\frac{N(N - 1)}{2}$
* Complete graph is an undirected graph with the number of edges $E$ $=$ $E_{max}$, and its average degree is $N-1$.
* Bipartite graph is a graph whose nodes can be divided into two disjoint sets U and V such that every link connects a node in $U$ to one in $V$ (independent sets)
* Ways for presenting a graph: visual, adjacency matrix, adjacency list.

## Properties of Networks vs Random Graphs

Metrics to measure network:

* Degree distribution, $P(k)$ : Probability that a randomly chosen node has degree $k$.
  - $N_{k}$ = # nodes with degree k.
* Path length, h: a length of the sequence of nodes in which each node is linked to the next one (path can intersect itself).
* Distance (shortest path, geodesic) between a pair of nodes is defined as the number of edges along the shortest path connecting the nodes. If the two nodes are not connected, the distance is usually defined as infinite (or zero).
* Diameter: maximum (shortest path) distance between any pair of nodes in a graph.
* Clustering coefficient, $C_{i}$: describes how connected are i’s neighbors to each other.
  - $C_{i}$ $\epsilon$ $[0,1],$ $C_{i}$ $=$ $\frac{2_{e_{i}}} {k_{i}{(k_{i} - 1)}}$
where e_{i} is the number of edges between neighbors of node i.

* Average clustering coefficient:
  
  $C$ $=$ $\frac{1}{N} {\sum_{i}^{N}} {C_{i}}$
  
* Connected components, s: a subgraph in which each pair of nodes is connected with each other via a path (for real-life graphs, they usually calculate metrics for the largest connected components disregarding unconnected nodes).

## Structural Roles in Networks

Subnetworks, or subgraphs, are the building blocks of networks, they have the power to characterize and discriminate networks.

Network motifs are: 

| Types      | Description |
| ----------- | ----------- |
| Recurring   | found many times, i.e., with high frequency|
| Significant | Small induced subgraph: Induced subgraph of graph G is a graph, formed from a subset X of the vertices of graph G and all of the edges connecting pairs of vertices in subset X.|
| Patterns of interconnections | More frequent than expected. Subgraphs that occur in a real network much more often than in a random network have functional significance.|

Motifs help us understand how networks work, predict operation and reaction of the network in a given situation.

Motifs can overlap each other in a network.

Significance of motifs: Motifs are overrepresented in a network when compared to randomized networks. Significance is defined as:

$SP_{i}$ $=$ $\frac {Z_{i}}{\sqrt{\sum_{j}}{\zeta_{j}^{2}}}$ $and$ $Z_{i}$ $=$ $\frac {(N_{i}^{real} - \nu_{i}^{rand})} {std(N_{i}{rand})}$

where ${N_{i}^{real}}$ is # of subgraphs of type i in network ${G^{real}}$ and ${N_{i}^{real}}$ is # of subgraphs of type i in network ${G^{rand}}$.

* ${Z_{i}}$ captures statistical significance of motif i.
* Generally, larger networks display higher Z-scores.
* To calculate ${Z_{i}}$, one counts subgraphs i in ${G^{real}}$, then counts subgraphs i in ${G^{rand}}$ with a configuration model which has the same #(nodes), #(edges) and #(degree distribution) as ${G^{real}}$.

## Graphlets

Graphlets are connected non-isomorphic subgraphs (induced subgraphs of any frequency). 
Graphlets differ from network motifs, since they must be induced subgraphs, whereas motifs are partial subgraphs. 
An induced subgraph must contain all edges between its nodes that are present in the large network, while a partial subgraph may contain only some of these edges. 
Moreover, graphlets do not need to be over-represented in the data when compared with randomized networks, while motifs do.

Node-level subgraph metrics:

* Graphlet degree vector counts #(graphlets) that a node touches.
* Graphlet Degree Vector (GDV) counts #(graphlets) that a node touches.

Graphlet degree vector provides a measure of a node’s local network topology: comparing vectors of two nodes provides a highly constraining measure of local topological similarity between them.

Finding size-k motifs/graphlets requires solving two challenges:

* Enumerating all size-k connected subgraphs
* Counting # of occurrences of each subgraph type via graph isomorphisms test.

Just knowing if a certain subgraph exists in a graph is a hard computational problem! 

* Subgraph isomorphism is NP-complete.
* Computation time grows exponentially as the size of the motif/graphlet increases.
* Feasible motif size is usually small (3 to 8).

## Structural roles in networks

Role is a collection of nodes which have similar positions in a network

* Roles are based on the similarity of ties between subsets of nodes.
* In contrast to groups/communities, nodes with the same role need not be in direct, or even indirect interaction with each other. 
* Examples: Roles of species in ecosystems, roles of individuals in companies.
* Structural equivalence: Nodes are structurally equivalent if they have the same relationships to all other nodes.

## Community Structure in Networks

The lecture shows methods for community detection problems (nodes clustering).

Background example from Granovetter work, 1960s: people often find out about new jobs through acquaintances rather than close friends. 
This is surprising: one would expect your friends to help you out more than casual acquaintances.

Why is it that acquaintances are most helpful?

* Long-range edges (socially weak, spanning different parts of the network) allow you to gather information from different parts of the network and get a job.
* Structurally embedded edges (socially strong) are heavily redundant in terms of information access.

## Network Communities

Network communities are sets of nodes with lots of internal connections and few external ones (to the rest of the network).

Modularity is a measure of how well a network is partitioned into communities: the fraction of the edges that fall within the given groups minus the expected fraction if edges were distributed at random.

## Louvian algorithm

Louvian algorithm is a greedy algorithm for community detection with O(n log(n)) run time.

* Supports weighted graphs.
* Provides hierarchical communities.
* Widely utilized to study large networks because it is fast, has rapid convergence and high modularity output (i.e., “better communities”).

Each pass of Louvain algorithm is made of 2 phases: 

Phase 1 – Partitioning:

* Modularity is optimized by allowing only local changes to node-communities memberships;
* For each node i, compute the modularity delta (∆Q) when putting node i into the community of some neighbor j; 
* Move i to a community of node j that yields the largest gain in ∆Q.

Phase 2 – Restructuring: 

* identified communities are aggregated into super-nodes to build a new network;
* communities obtained in the first phase are contracted into super-nodes, and the network is created accordingly: super-nodes are connected if there is at least one edge between the nodes of the corresponding communities;
* the weight of the edge between the two supernodes is the sum of the weights from all edges between their corresponding communities;
* goto Phase 1 to run it on the super-node network.

The passes are repeated iteratively until no increase of modularity is possible.

## Detecting overlapping communities: BigCLAM

BigCLAM is a model-based community detection algorithm that can detect densely overlapping, hierarchically nested as well as non-overlapping communities in massive networks

Step 1: Define a generative model for graphs that is based on node community affiliations –  Community Affiliation Graph Model (AGM). Model parameters are Nodes V, Communities C, Memberships M. Each community c has a single probability pc.

Step 2: Given graph G, make the assumption that G was generated by AGM. Find the best AGM that could have generated G. This way we discover communities.


## Spectral Clustering

Three basic stages of spectral clustering:

1. Pre-processing: construct a matrix representation of the graph.
2. Decomposition: compute eigenvalues and eigenvectors of the matrix; map each point to a lower-dimensional representation based on one or more eigenvectors.
3. Grouping: assign points to two or more clusters, based on the new representation.

### Graph partitioning

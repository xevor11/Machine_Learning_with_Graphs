Lecture 1 -- Intro to graph theory
=================================

*[Slides](http://snap.stanford.edu/class/cs224w-2019/slides/01-intro.pdf), [Video](https://youtu.be/mL7D311K1kg)*

The lecture gives an overview of basic terms and definitions used in graph theory. Why should you learn graphs? The impact! The impact of graphs application is proven for Social networking, Drug design, AI reasoning.

My favourite example of applications is the Facebook social graph which shows that all users have only 4-degrees of separation:

![](https://lh6.googleusercontent.com/-78lCVp5PYiEMZ7AnWVJrCnu6Svm_kNmnTtEvYYhn4cJ1mvbEhySeCf5u2RmeqsPYqwsaE6cAz8tISpjMVVLBGe1Z-JfHpvUhhHlbzt4w5_BPmAEM_JT77Yu3-lF738fZoNJi70E)

*[Backstrom-Boldi-Rosa-Ugander-Vigna, 2011]*

Another interesting example of application: predict side-effects of drugs (when patients take multiple drugs at the same time, what are the potential side effects? It represents a link prediction task).

What are the ways to analyze networks:

1.  Node classification (Predict the type/color of a given node)
2.  Link prediction (Predict whether two nodes are linked)
3.  Identify densely linked clusters of nodes (Community detection)
4.  Network similarity (Measure similarity of two nodes/networks)

Key definitions:

-   A network is a collection of objects (nodes) where some pairs of objects are connected by links (edges). 
-   Types of networks: Directed vs. undirected, weighted vs. unweighted, connected vs. disconnected
-   Degree of node *i* is the number of edges adjacent to node *i.*
-   The maximum number of edges on N nodes (for undirected graph) is

![E_{max}=\frac{N(N-1)}{2}](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-9294df641e197f8261c2bad7744992b8_l3.svg "Rendered by QuickLaTeX.com")

-   Complete graph is an undirected graph with the number of edges *E = Emax*, and its average degree is *N-1*.
-   Bipartite graph is a graph whose nodes can be divided into two disjoint sets* U *and *V* such that every link connects a node in *U* to one in *V*; that is, *U* and *V* are independent sets (restaurants-eaters for food delivery, riders-drivers for ride-hailing).
-   Ways for presenting a graph: visual, adjacency matrix, adjacency list.

* * * * *

Lecture 2 -- Properties of networks. Random Graphs
=================================================

*[Slides](http://snap.stanford.edu/class/cs224w-2019/slides/02-gnp-smallworld.pdf), [Video](https://youtu.be/BEyzpx8ko7s)*

Metrics to measure network:

-   *Degree distribution, P(k)*: Probability that a randomly chosen node has degree *k*. 

Nk = # nodes with degree *k*.

-   *Path length, h*: a length of the sequence of nodes in which each node is linked to the next one (path can intersect itself).
-   *Distance* (shortest path, geodesic) between a pair of nodes is defined as the number of edges along the shortest path connecting the nodes. If the two nodes are not connected, the distance is usually defined as infinite (or zero).
-   *Diameter*: maximum (shortest path) distance between any pair of nodes in a graph.
-   *Clustering coefficient, Ci*: describes how connected are i's neighbors to each other.

![C_{i}\in [0,1], C_{i}=\frac{2e_{i}}{k_{i}(k_{i}-1)}](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-b31dcfa47f42999d8d86f4c99dd3d98f_l3.svg "Rendered by QuickLaTeX.com")

where *ei* is the number of edges between neighbors of node *i.*

-   *Average clustering coefficient:*

![C=\frac{1}{N}\sum_{i}^{N}C_{i} ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-269aa8ce9692c9fb2d94c762f434ddcf_l3.svg "Rendered by QuickLaTeX.com")

-   *Connected components, s*: a subgraph in which each pair of nodes is connected with each other via a path (for real-life graphs, they usually calculate metrics for the largest connected components disregarding unconnected nodes).

###### Erdös-Renyi Random Graphs:

-   *Gnp*: undirected graph on n nodes where each edge *(u,v)* appears i.i.d. with probability *p*.
-   Degree distribution of *Gnp* is binomial.
-   Average path length: O(log n).
-   Average clustering coefficient: *k / n*.
-   Largest connected component: exists if *k > 1*.

Comparing metrics of random graph and real life graphs, real networks are not random. While *Gnp* is wrong, it will turn out to be extremely useful!

Real graphs have high clustering but diameter is also high. How can we have both in a model? Answer in Small-World Model* [Watts-Strogatz '98]*.

Small-World Model introduces randomness ("shortcuts"): first, add/remove edges to create shortcuts to join remote parts of the lattice; then for each edge, with probability *p*, move the other endpoint to a random node.

![](https://lh4.googleusercontent.com/HBAY4DL8pZBzebHo_E_k00oZ6X6S4L8IvX9bOmdTYdsVCTv5YR944uzkNgqKQLNonEQklwCOR5ne6mzmcXAx3R1q5Nm1Jpej0JFVptAKJSgsC0pYQAuscraYxOr2-YK-Xy9jBnB3)

*Difference between regular random and small world networks*

Kronecker graphs: a recursive model of network structure obtained by growing a sequence of graphs by iterating the Kronecker product over the initiator matrix. It may be illustrated in the following way:

![](https://lh6.googleusercontent.com/a9wnNlkCnpcDshd_Ut2dT0gT8y7ABBBemz__4Kla_aQUcv1RCCOV4UW012oXpsYPR4UoeiTo5IAM-SiVs-oWm7j4O8PAR-BOmB8n6VxXZSoxWD8Y0bPydBQ9C1PbxGC6unXAAl_J)

Random graphs are far away from real ones, but real and Kronecker networks are very close.

* * * * *

Recitation 1 -- Introduction to snap.py
======================================

[*Slides*](http://snap.stanford.edu/class/cs224w-2019/slides/CS224W-snappy-tutorial.pdf)

Stanford Network Analysis Platform (SNAP) is a general purpose, high-performance system for analysis and manipulation of large networks ([http://snap.stanford.edu](http://snap.stanford.edu/)). SNAP software includes Snap.py for Python, SNAP C++ and SNAP datasets (over 70 network datasets can be found at <https://snap.stanford.edu/data/>). 

Useful resources for plotting graph properties:

-   Gnuplot: [http://www.gnuplot.info](http://www.gnuplot.info/)
-   Visualizing graphs with Graphviz: [http://www.graphviz.org](http://www.graphviz.org/)
-   Other options are in Matplotlib: [http://www.matplotlib.org](http://www.matplotlib.org/) 

I will not go into details here, better to get hands on right away with the original snappy tutorial at <https://snap.stanford.edu/snappy/doc/tutorial/index-tut.html>.

* * * * *

Lecture 3 -- Motifs and Structural Roles in Networks 
====================================================

*[Slides](http://snap.stanford.edu/class/cs224w-2019/slides/03-motifs.pdf), [Video](https://youtu.be/pn8jba-rnWY)*

Subnetworks, or subgraphs, are the building blocks of networks, they have the power to characterize and discriminate networks.

Network motifs are: 

| Recurring | found many times, i.e., with high frequency |
| Significant | Small induced subgraph: Induced subgraph of graph G is a graph, formed from a subset X of the vertices of graph G and all of the edges connecting pairs of vertices in subset X. |
| Patterns of interconnections | More frequent than expected. Subgraphs that occur in a real network much more often than in a random network have functional significance. |

![](https://lh3.googleusercontent.com/dgpn38f92r3CKjj67WJ_ldKg-PEqG3wo-RziCc-9D6KOa10yiuhHKINTMW8LT55jsjf-B9G5gj8e96HSMXuCalZpiKiAkQcW_NeUch0IMNuoMi0UUe0WcIZiROQyqgaZqhb3DpnU)

*How to find a motif [drawing is mine]*

Motifs help us understand how networks work, predict operation and reaction of the network in a given situation.

Motifs can overlap each other in a network.

Examples of motifs: 

![](https://lh4.googleusercontent.com/5OrWIS4UCtDNa4VDafovRLKjpQSFrCdXfDy2l7YYENo071Akuyw4FTzz_hJCRw6t3HGyScFGliGzMRcnURNQ2IPDyi7tvyPtLHZb-pxcuStEDpeCr91Xf8pSPkKfd7oULOiu717U)

*[Drawing is mine]*\
*Significance of motifs*: Motifs are overrepresented in a network when compared to randomized networks. Significance is defined as:

![SP_{i}=Z_{i}/\sqrt{\sum_{j}^{}Z_{j}^{2}}\: \: and \, \, \; \! Z_{i}=(N_{i}^{real}-\bar{N_{i}^{rand}})/std(N_{i}^{rand})](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-01be7939beaa6804ebbc8688e78a1bc5_l3.svg "Rendered by QuickLaTeX.com")

where *Nireal *is # of subgraphs of type *i* in network* Greal* and *Nireal* is # of subgraphs of type *i *in network *Grand*.

-   *Zi* captures statistical significance of motif *i.*
-   Generally, larger networks display higher Z-scores.
-   To calculate *Zi*, one counts subgraphs *i* in *Greal*, then counts subgraphs *i* in *Grand* with a configuration model which has the same #(nodes), #(edges) and #(degree distribution) as *Greal*.

![](https://lh3.googleusercontent.com/62Hke1RIe_gzXj7NpiEh_Fgg0Ps9oYG3eEnG_fvUsMtp95gIiVAlqT0q3gnTSR4-kh22V-28p6a3-htJesFUiZxoYHP78FXjnhBGNbK4QUtoN5aHBY-a4OwZYFw9JX71jmYhgcKT)

*How to build configuration model*

###### Graphlets

Graphlets are connected non-isomorphic subgraphs (induced subgraphs of any frequency). Graphlets differ from network motifs, since they must be induced subgraphs, whereas motifs are partial subgraphs. An induced subgraph must contain all edges between its nodes that are present in the large network, while a partial subgraph may contain only some of these edges. Moreover, graphlets do not need to be over-represented in the data when compared with randomized networks, while motifs do.

Node-level subgraph metrics:

-   Graphlet degree vector counts #(graphlets) that a node touches.
-   Graphlet Degree Vector (GDV) counts #(graphlets) that a node touches.

Graphlet degree vector provides a measure of a node's local network topology: comparing vectors of two nodes provides a highly constraining measure of local topological similarity between them.

Finding size-k motifs/graphlets requires solving two challenges: 

-   Enumerating all size-k connected subgraphs
-   Counting # of occurrences of each subgraph type via graph isomorphisms test.

Just knowing if a certain subgraph exists in a graph is a hard computational problem! 

-   Subgraph isomorphism is NP-complete.
-   Computation time grows exponentially as the size of the motif/graphlet increases.
-   Feasible motif size is usually small (3 to 8).

Algorithms for counting subgaphs: 

-   Exact subgraph enumeration (ESU) *[Wernicke 2006]* (explained in the lecture, I will not cover it here).
-   Kavosh *[Kashani et al. 2009]*.
-   Subgraph sampling *[Kashtan et al. 2004]*.

Structural roles in networks

Role is a collection of nodes which have similar positions in a network

-   Roles are based on the similarity of ties between subsets of nodes. 
-   In contrast to groups/communities, nodes with the same role need not be in direct, or even indirect interaction with each other. 
-   Examples: Roles of species in ecosystems, roles of individuals in companies.
-   Structural equivalence: Nodes are structurally equivalent if they have the same relationships to all other nodes.

![](https://lh6.googleusercontent.com/tmIuRqPpM_lBYzVBICwNDrB5WRtw0hzNUPEwHRa3kJHSs9TWpvqJnuDB3_8ZhWDcKxf25OPzqOxSr-DsOM2WcCKHZyqJoKdXjHR9FvhS9rKegSuwHbmLeIXn6stoh0DtBOzxj666)

*Nodes 3 and 4 are structurally equivalent*

Examples when we need roles in networks:

-   Role query: identify individuals with similar behavior to a known target. 
-   Role outliers: identify individuals with unusual behavior. 
-   Role dynamics: identify unusual changes in behavior.

Structural role discovery method RoIX:

-   Input is adjacency matrix.
-   Turn network connectivity into structural features with Recursive feature Extraction *[Henderson, et al. 2011a]*. Main idea is to aggregate features of a node (degree, mean weight, # of edges in egonet, mean clustering coefficient of neighbors, etc) and use them to generate new recursive features (e.g., mean value of "unweighted degree" feature between all neighbors of a node). Recursive features show what kinds of nodes the node itself is connected to.
-   Cluster nodes based on extracted features. RolX uses non negative matrix factorization for clustering, MDL for model selection, and KL divergence to measure likelihood.

![](https://lh3.googleusercontent.com/joQejOKfr66_yr1s0IdZ3JOEJvtoQK6ls3sJr9az0Tl9qHDbSbT81rSPR_g2B0YuK-1H-ANKbJI74zyyeZwgrppaetDTZXDqFHbQP345frTAyRvtmsBh-7AnbZVAB1mbupin1xbJ)

*Structural role discovery method RoIX*

* * * * *

Lecture 4 -- Community Structure in Networks
===========================================

*[Slides](http://snap.stanford.edu/class/cs224w-2019/slides/04-communities.pdf), [Video](https://youtu.be/JEqAMcVr4jw)*

*The lecture shows methods for community detection problems (nodes clustering).*

Background example from Granovetter work, 1960s: people often find out about new jobs through acquaintances rather than close friends. This is surprising: one would expect your friends to help you out more than casual acquaintances.

Why is it that acquaintances are most helpful?

-   Long-range edges (socially weak, spanning different parts of the network) allow you to gather information from different parts of the network and get a job.
-   Structurally embedded edges (socially strong) are heavily redundant in terms of information access.

![](https://lh4.googleusercontent.com/e47So0YL6FaGp3AEGS2BFiqOJfNBVe4rFQujqDZyg4jGgPR3hrY0w2bUsLYilUs2lK5kTjRfuyCz05iyF78zwPEzW70Kxp_cuaoCfmPdd7Emu4dqWS3NMh16Fk_rT1IqExsXvRyP)

*Visualisation of strong and weak relations in a network*

From this observation we find one type of structural role --  Triadic Closure: if two people in a network have a friend in common, then there is an increased likelihood they will become friends themselves.

Triadic closure has high clustering coefficient. Reasons: if A and B have a friend C in common, then: 

-   A is more likely to meet B (since they both spend time with C). 
-   A and B trust each other (since they have a friend in common).
-   C has incentive to bring A and B together (since it is hard for C to maintain two disjoint relationships).

Granovetter's theory suggests that networks are composed of tightly connected sets of nodes and leads to the following conceptual picture of networks:

![](https://lh5.googleusercontent.com/8SIl1QurIM_Yj991YYklDBng3P7s2wZvwbzRSQzVnT_8UJjeW4OFR-wHysYS5YPj_wC1L7KOsW0AJmbOzLobeLauXmXRZTG-Qs2IzkIM4RksPxuWAZRZgYCRZuKAdnJs-VbAJfv1)

###### Network communities

Network communities are sets of nodes with lots of internal connections and few external ones (to the rest of the network).

Modularity is a measure of how well a network is partitioned into communities: the fraction of the edges that fall within the given groups minus the expected fraction if edges were distributed at random.

Given a partitioning of the network into groups disjoint *s*:

![s\in S: \enspace Q\propto \sum_{s\in S}^{}[(number \; of \; edges \; within \; group \; s)-(expected \; number \; of \; edges \; within \; group \; s)] ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-6cf61ed80d005d79717b9c48e6750059_l3.svg "Rendered by QuickLaTeX.com")

Modularity values take range [-1,1]. It is positive if the number of edges within groups exceeds the expected number. Values greater than 0.3-0.7 means significant community structure.

To define the expected number of edges, we need a Null model. After derivations (can be found in slides), modularity is defined as:

![Q(G, S)=\frac{1}{2m}\sum_{s \in S}^{}\sum_{i \in s}^{}\sum_{j \in s}^{}(A_{ij}-\frac{k_{i}k_{j}}{2m}) ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-c27ab37210bb46894be20ad09910281f_l3.svg "Rendered by QuickLaTeX.com")

-   Aij represents the edge weight between nodes *i* and *j*;
-   *ki* and *kj* are the sum of the weights of the edges attached to nodes *i* and *j*;
-   *2m* is the sum of all the edges weights in the graph.

###### Louvian algorithm

Louvian algorithm is a greedy algorithm for community detection with *O(n log(n))* run time.

-   Supports weighted graphs.
-   Provides hierarchical communities.
-   Widely utilized to study large networks because it is fast, has rapid convergence and high modularity output (i.e., "better communities").

Each pass of Louvain algorithm is made of 2 phases: 

*Phase 1 -- Partitioning: *

-   Modularity is optimized by allowing only local changes to node-communities memberships;
-   For each node *i*, compute the modularity delta (*∆Q*) when putting node *i* into the community of some neighbor *j*; 
-   Move *i* to a community of node *j* that yields the largest gain in *∆Q*.

*Phase 2 -- Restructuring: *

-   identified communities are aggregated into super-nodes to build a new network;
-   communities obtained in the first phase are contracted into super-nodes, and the network is created accordingly: super-nodes are connected if there is at least one edge between the nodes of the corresponding communities; 
-   the weight of the edge between the two supernodes is the sum of the weights from all edges between their corresponding communities;
-   goto Phase 1 to run it on the super-node network.

The passes are repeated iteratively until no increase of modularity is possible.

![](https://lh6.googleusercontent.com/mkDQ-ZPfc44jeHmenlTemO02aJ8jKmk2eu6SPnpQfLDrFInMj16ytXi0TrYsqpwsKdIiY6-raAC2xCd2IYuL5tj_4CHMMynS0fP8FbpJJX4m365AiFkmahj2p6zaPazawTnPVsdI)

*Louvain algorithm*

###### Detecting overlapping communities: BigCLAM

BigCLAM is a model-based community detection algorithm that can detect densely overlapping, hierarchically nested as well as non-overlapping communities in massive networks (more details and all proofs are in Yang, Leskovec "[Overlapping Community Detection at Scale: A Nonnegative Matrix Factorization Approach](http://infolab.stanford.edu/~crucis/pubs/paper-nmfagm.pdf)", 2013).

In BigCLAM communities arise due to shared community affiliations of nodes. It explicitly models the affiliation strength of each node to each community. Model assigns each node-community pair a nonnegative latent factor which represents the degree of membership of a node to the community. It then models the probability of an edge between a pair of nodes in the network as a function of the shared community affiliations. 

*Step 1*: Define a generative model for graphs that is based on node community affiliations --  Community Affiliation Graph Model (AGM). Model parameters are Nodes *V*, Communities *C*, Memberships *M*. Each community *c* has a single probability *p**c**.*

*Step 2*: Given graph *G*, make the assumption that *G* was generated by AGM. Find the best AGM that could have generated G. This way we discover communities.

![](https://lh3.googleusercontent.com/Q30rcZhzmyp2mv7_xoLE61sOXbuumq3w2jLNejziyP2CwUUbuPbRRg5SP-WkxBuKJT0B7PMa29mN1WW-EyVlAt0leHQZLJgOm8WnBLOC9VxaB3vlebkGqosjIwyjUF8343m_gujp)
===============================================================================================================================================================================

* * * * *

Lecture 5 -- Spectral Clustering
===============================

*[Slides](http://snap.stanford.edu/class/cs224w-2019/slides/05-spectral.pdf), [Video](https://youtu.be/OL2-FhGCohE)*

*NB: these notes skip lots of mathematical proofs, all derivations can be found in slides themselves.*

Three basic stages of spectral clustering:

1.  Pre-processing: construct a matrix representation of the graph.
2.  Decomposition: compute eigenvalues and eigenvectors of the matrix; map each point to a lower-dimensional representation based on one or more eigenvectors.
3.  Grouping: assign points to two or more clusters, based on the new representation.

Lecture covers them one by one starting with graph partitioning.

###### Graph partitioning

Bi-partitioning task is a division of vertices into two disjoint groups A and B. Good partitioning will maximize the number of within-group connections and minimize the number of between-group connections. To achieve this, define edge cut: a set of edges with one endpoint in each group:

![Cut(A,B)=\sum_{i \in A, j \in B}^{}w_{ij} ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-431983726cfa497553508d1c963a8e7e_l3.svg "Rendered by QuickLaTeX.com")

![](https://lh3.googleusercontent.com/kmjxrOR4FFk6dnMR2g2A1QWwuwzndtI5NPdnADxCWvRKTHJ6P7rpBLwP6teb9Ig0Ye6jaLhfAtMM7gknszVSaHhdmP8zw90kT0vdv88JYDdjegbQ4N1DMhQO8RDT6s4AbhHKJ_i6)

Graph cut criterion:

-   *Minimum-cut*: minimize weight of connections between groups. Problems: it only considers external cluster connections and does not consider internal cluster connectivity.
-   *Conductance*: connectivity between groups relative to the density of each group. It produces more balanced partitions, but computing the best conductance cut is NP-hard.

*Conductance* * φ(A,B) = cut(A, B) / min(total weighted degree of the nodes in A or in B)*

To efficiently find good partitioning, they use Spectral Graph Theory.

Spectral Graph Theory analyze the "spectrum" of matrix representing G: Eigenvectors x(i) of a graph, ordered by the magnitude (strength) of their corresponding eigenvalues *λi*.

Λ = {*λ1*,*λ2*,...,*λn}, λ1≤ λ2*≤ ...≤ *λn* (*λi* are sorted in ascending (not descending) order).

If *A* is adjacency matrix of undirected *G* (aij=1 if (i,j) is an edge, else 0), then *Ax = λx.*

What is the meaning of *Ax*? It is a sum of labels *xj* of neighbors of *i*. It is better seen if represented in the form as below:

![](https://lh4.googleusercontent.com/yMbrTQXeWbv8WLrSohtnPF_QXG889H9uiJH4Z5wlfXv4xW9YsYP-rzVa29I3xWBZjlmrsbmgGIVYIdOsjO0CuF06BVrz40G_VoXOMOOBJnOJeYxRIxa805DNXSCkyIh0iwhuc3LL)

*Examples of eigenvalues for different graphs:*

For d-regular graph (where all nodes have degree *d*) if try *x=(1,1,1,...1)*:

*Ax = (d,d,..,d) = *λ*x -> *λ** *= d*. This *d* is the largest eigenvalue of d-regular graph (proof is in slides).

An *N x N* matrix can have up to *n* eigenpairs.

*Matrix Representation:*

-   Adjacency matrix (*A): A = [aij], *aij* = 1* if edge exists between node *i* and *j* (0-s on diagonal). It is a symmetric matrix, has n real eigenvalues; eigenvectors are real-valued and orthogonal.
-   Degree matrix (D): D=[*dij*],dijis a degree of node *i*.
-   Laplacian matrix *(L): L = D-A*. Trivial eigenpair is with *x=(1,...,1) -> Lx = 0* and so* λ = λ1 = 0.*

So far, we found *λ *and* λ1*. From theorems, they derive

![\lambda _{2}=min \frac {\sum_{(i,j) \in E} (x_{i}-x_{j})^{2}}{\sum_{i}x_{i}^{2}} ](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-6230246da2422119308ea1f0003c052c_l3.svg "Rendered by QuickLaTeX.com")

(all labelings of node *i* so that *sum(xi) = 0*, meaning to assign values *xi* to nodes *i* such that few edges cross 0  (*xi* and *xj* to subtract each other as shown on the picture below).

![](https://lh5.googleusercontent.com/rRNdzN43WN1a84G4IO1fCbcrcSGz-GTMoz5rqvbcf2IbMa-PVcet35h62gbFc7qjp4CAdhlfe9rF57B4TTPH9NaQF2nf6FkKk-vB7Zf34PKjw9vURFQHwJ7ol902fKPYxu9yQ0JO)

From theorems and profs, we go to the Cheeger inequality:

![\frac {\beta ^{2}}{2k_{max}}\leqslant \lambda _{2} \leqslant 2 \beta](https://elizavetalebedeva.com/wp-content/ql-cache/quicklatex.com-83c48b3e7363f5f58894181bc0b36923_l3.svg "Rendered by QuickLaTeX.com")

where *kmax* is the maximum node degree in the graph and *β *is conductance of the *cut (A, B)*.

*β = (# edges from A to B) / |A|*. This is the approximation guarantee of the spectral clustering: spectral finds a cut that has at most twice the conductance as the optimal one of conductance *β*.

To summarize 3 steps of spectral clustering:

1.  Pre-processing: build Laplacian matrix L of the graph.
2.  Decomposition: find eigenvalues *λ* and eigenvectors *x* of the matrix L and map vertices to corresponding components of x2.
3.  Grouping: sort components of reduced 1-dimensional vector and identify clusters by splitting the sorted vector in two.

![](https://lh3.googleusercontent.com/6aKqqxZ0dr6SK7McfIqPI9TQMQOw7XcvZEbme8ahTvHKQTRQXuucj5rK0yKRt4yd3-EUaTRAPGGBFk-dMPr9KIt3WVArPJM73ggSQagN9S9gbW-DyyK1UXlcy_5ftdz4ZULDKSvL)

How to choose a splitting point? 

-   Naïve approaches: split at 0 or median value.
-   More expensive approaches: attempt to minimize normalized cut in 1-dimension (sweep over ordering of nodes induced by the eigenvector).

###### K-way spectral clustering:

Recursive bi-partitioning *[Hagen et al., '92]* is the method of recursively applying bi-partitioning algorithm in a hierarchical divisive manner. It has disadvantages: inefficient and unstable.

More preferred way: 

Cluster multiple eigenvectors *[Shi-Malik, '00]*: build a reduced space from multiple eigenvectors. Each node is now represented by *k* numbers; then we cluster (apply k-means) the nodes based on their *k*-dimensional representation.

To select *k*, use eigengap: the difference between two consecutive eigenvalues (the most stable clustering is generally given by the value *k* that maximizes eigengap).

![](https://lh4.googleusercontent.com/HrXGB29jU-obtjIKb8SEbgILknUUIi8E4YiGJo3mSgJKSfguHnUsSGWsD1xojhjzQ_TPmmxQ7jjhi1uH1lqQnjE-A6wEWpqpTXUWxI3xcLzzf6nYWhXXijj3SNw9zCu_1QsOvJUZ)

*K-way spectral clustering: choose k=2 as it has the highest eigengap.*

###### Motif-based spectral clustering

Different motifs will give different modular structures. Spectral clustering for nodes generalizes for motifs as well.

Similarly to edge cuts, volume and conductance, for motifs:

![](https://lh6.googleusercontent.com/Ai9yKbqd5gYAzWoKBEHGtmj-_j7Gc58vpPKwUu5T0EVXQGnIG0i6q1g-ngRopNCGKYAZokk1OXbT3PPMIT5ZnWI6Qm2eGRxMThLuRgmAqM_rbjiepZMw-oNWL1hckdqhnYyaF67U)

*volM(S) = #(motif endpoints in S), φ(S) = #(motifs cut) / volM(S)*.

Motif Spectral Clustering works as follows:

-   Input: Graph G and motif M.
-   Using G form a new weighted graph W(M).
-   Apply spectral clustering on W(M).
-   Output the clusters.

Resulting clusters will obtain near optimal motif conductance.

The same 3 steps as before:

1.  Preprocessing: given motif M. Form weighted graph W(M).

*Wij(M)=# times edge (i,j) participates in the motif M.*

1.  Apply spectral clustering: Compute Fiedler vector *x* associated with λ2 of the Laplacian of L(M).

*L(M) = D(M) -- W(M) *where degree matrix *Dij(M) = sum(Wij(M))*. Compute L(M)x = λ2x. Use *x* to identify communities.

1.  Sort nodes by their values in *x: x1, x2, ...xn*. Let *Sr = {x1, x2, ...xn}* and compute the motif conductance of each *Sr*.

At the end of the lecture, there are two examples of how motif-based clustering is used for food webs and gene regulatory networks.

There exist other methods for clustering:

-   [METIS](http://glaros.dtc.umn.edu/gkhome/views/metis): heuristic but works really well in practice.
-   [Graclus](http://www.cs.utexas.edu/users/dml/Software/graclus.html): based on kernel k-means. 
-   [Louvain](http://perso.uclouvain.be/vincent.blondel/research/louvain.html): based on Modularity optimization.
-   [Clique percolation method](http://angel.elte.hu/cfinder/): for finding overlapping clusters.

# MolecularGeometricDL
A repo of the most seminal applications of geometric deep learning in molecular tasks.

The most comprehensive professionally curated resource on Geometric Deep learning applied in molecular tasks including
the best tutorials, videos, books, papers, articles, courses, websites, conferences and open-source libraries.

I am creating this resource, while conducting my PhD work at Helmholtz Centre of Environmental Research under the supervision 
of Dr. Jana Schor.
The papers will be listed by time order, noting the advancements along the way.

Disclaimer: All the images are sourced from the resources I linked.

# Table of Contents 
- [What is a graph?](#What-is-a-graph?)
- [Graph Neural Networks](#Graph-Neural-Networks)
- [Graph Convolutions](#Graph-Convolutions)
- [Tutorials](#Tutorials)
- [Papers](#Papers)
- [Articles/Blogs](#Articles)
- [Repositories](#Repositories)
- [Videos](#Videos)
- [Tools](#Tools)



## What is a graph?
A graph G is a set of nodes and vertices between them G(V,E). Molecules can be intuitively seen as graphs where the nodes are the atoms and the edges are the bonds between them.


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/graphs.png?raw=true)


How we  use the graphs as input though?

The graph can be represented essentially by three matrices:
- The adjacency matrix, which shows how the nodes(atoms) are connected
- The node features matrix, which encodes information about every node(atom)
- The edge features matrix, whoch encodes information about the edge(bond)


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/graphmatrices.png?raw=true)

## Graph Neural Networks

GNNs are a type of networks that operate on these graphs. 

There are two ways to develop GNNs, spectrally and spatially.

Both tried to generalize the mathematical concept of convolution to graphs.
The spectral methods stuck to the strict mathematical notions resorted to the frequency domain(Laplacian eigenvectors).
Being computationally expensive and not applicable to inductive scenarios, they finally died out.
Spatial ones form the ones, now known as graph convolutions and are the ones that we are going to analyse more.
If you still want to get a basic understanding of spectral methods you can advise the links below.

Oops, I mentioned inductive without even explaining. The image speaks for itself.

__Inductive learning:__
This type of learning is like the usual supervised learning. The model has not seen the nodes/graphs that will
later classify. This applies to graph-classification tasks which are our main interst for molecular proprerty
prediction.

__Transductive learning:__
In transductive learning, the model has seen the nodes without their labels and/or some features but gets an
understanding of how they are connected within the graph. That is useful mainly for node-classification tasks.

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/inductive.png?raw=true)



## Graph Convolutions

__Normal Convolutions__

A typical feed forward network does a forward pass with the following equation:

- __Y = &#963;(W*X + &#946;)__,

where σ is a non-linear function(ReLU,tanh), W is the weight associated with each feature, X the features and β is the bias.

In convolutional neural networks, the input usually is an image(i.e a tensor height* width*channels). 

- An RGB image has three channels whereas a greyscale only one. 

- In CNNs, the W is called a filter or kernel and is usually a matrix(2x2, 3x3 etc.) which is the same passed acrossed the image to extract features(patterns) from every part of the image.

- That is called weight sharing. That is done because a pattern is interesting wherever it is in the image(translational invariance)


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/cnns.gif)

The question became how we can generalize the convolutions to graphs?

There are some significant differences between images and graphs.
- Images are positioned in a Euclidean space, and thus have a notion of locality.Pixels that are close to each other will be much more strongly related than distant ones. Graphs on the other hand do not as information about the distance between nodes is not encoded.
- Pixels follow an order while graph nodes do not.
So, the locality is achieved in graphs based on neighborhoods.
Also, we adopt the weight sharing from the normal convolutions.

__Invariance__

The order invariance is achieved by applying functions that are order invariant
Permutation matrix, P is a matrix that only changes the order of another matrix.
So for every P, the following equation should be obeyed.
- f(PX)=f(X)

__Equivariance__

But if we wanted information on node-level the invariant function would not suffice. Instead, we need a permutation equivariant function that do not change the node order and follow the following equation.
- f(PX)=Pf(X)

We can think of these functions f that transform the x<sub>i</sub> features of a node to a latent vector h<sub>i</sub>.


__h<sub>i</sub> = f(x<sub>i</sub>)__

Stacking these will result in 
__H = f(X)__

How we can use these latent vectors?

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/graphtasks.png?raw=true)

But hold on...

How we incorporate the adjacency matrix A into this equation?

A simple update rule: 

__H<sub>k+1</sub> = &#963;(A*W*H<sub>k</sub>)__, 
where A is the adjacency matrix, k is the number of itearations and we dropped β for simplicity reasons.

Hopefully the similarities with the classical equation are obvious.

Node-wise the equation is written:

__h<sub>i</sub> = Σ (W*h<sub>j</sub>)__,

where j is every neighbor of the node i.

Let's see it in practice:


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/adjacency.png?raw=true)

Considering this adjacency matrix, when we update the state of the node v<sub>1</sub>, we will take into account its neighbor states.
That although would be wrong as we'll be entirely dropping the previous state of node v<sub>1</sub>. 
So, we need to make a correction to the adjacency matrix A by adding the identity matrix and creating the matrix Ã.
That would add 1s across the diagonal making each node a neighbor of itself, i.e we add self-loops.

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/adjacencycorrected.jpg?raw=true)

Each latent vector of a node is a sum of the vectors of its neighbors. So, if the degree of a node( degree shows to how many neighbors a node has) is really high the scale of the latent vector would be entirely different and we'll face vanishing or exploding gradients.
- So, we should normalize based on the degree of the node.
Firstly we calculate degree matrix, D by summing up row-wise the adjacency matrix, Ã.


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/degree.jpg?raw=true)

Then, we inverse it and thus the equation takes the form.


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/inversedegree.jpg?raw=true)

__H<sub>k+1</sub> = &#963;(ÃD<sup>-1</sup>W<sub>k</sub>H<sub>k</sub>)__

__WE DID IT!__
We now have the first equation upon we can build our different variants of graph convolutions.

This equation essentially describes a simple averaging of the neighbors' vectors.
This update of the state of the vectors happens for i number of steps.
On each step or a neighborhood hop you aggregate the vectors fo the neighbors.
Once we have all the latent vectors for each node after k number of steps, we can use these for node classification or in our case
we can aggregate them and reach a unique embedding for every graph.
# GCN Variants
- GCN

The most known variation of graph convolutions was set by [Kipf & Welling](https://arxiv.org/abs/1609.02907)
in 2017. 
They introduced a renormalization trick which is more than just a mere average of the neighbors.
They normalize by  1&#247;&#8730;(d<sub>i</sub> * d<sub>j</sub>)


__H<sub>k+1</sub> = &#963;(D<sup>-1/2</sup>ÃD<sup>-1/2</sup>W<sub>k</sub>H<sub>k</sub>)__

From now on, we'll refer to it as the __GCN__.

- GAT(Graph Attention Networks)

Petar Velickovic had another [idea](https://arxiv.org/abs/1710.10903). Instead, of giving an equal weight to every neighbor that will be added explicitly, 
a concept called attention. So, the node-wise equation now became:

__h<sub>i</sub> = σ(Σ(a<sub>ij</sub>h<sub>j</sub>))__
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/GAT.png?raw=true)

The a<sub>ij</sub> comes from applying a softmax to e<sub>ij</sub> = a(h<sub>i</sub>,h<sub>j</sub>)
which are non-normalized coefficients across pairs of nodes

Influenced by the results of [Vaswani et al.](https://arxiv.org/abs/1706.03762)
they included multi head attention mechanisms which is essentially a K 
number of replicates which are then concatenated or aggregated. The following figure from the paper makes it abundantly 
clear.
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/GAT-MULTI.png?raw=true)


# Message Passing Neural Nets

The term message-passing arised in 2017 and  is really intuitive way to see graph neural nets.
The two main points evolve around the two functions that happen in a GNN
- The Update function, q
- The Aggregate function, U

From this [youtube video](https://www.youtube.com/watch?v=zCEYiCxrL_0) we can sum them up by the figure.

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/MPNN.png?raw=true)

Essentially we concatenate the vector of the node-in-focus of the previous step with the edges K and its neighbors.
The resulting vector passed through an update function f and then aggregated by the function U. 
Finally they are passed through a non-linear function to get new updated representation.

The previously described GCN and GAT, following a similar [formalism](https://towardsdatascience.com/a-unified-view-of-graph-neural-networks-12b40e8fdac5)
can be described in the following figures.
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/image/GCNGATMP.png?raw=true)


This [article](https://distill.pub/2021/understanding-gnns/) includes an interactive session to play aroung with graphs and the most essential GNN variants.




## Tutorials

- [Intro to Graph Neural Networks](https://www.youtube.com/watch?v=8owQBFAHw7E)
- [UvA DL Notebooks](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial7/GNN_overview.html)
## Papers
| Paper | Author | Year | Github | Comments |
| ---------- | ----- | --- | --- | -------------|
| [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292) | Duvenaud et al. | 2015 | None | Vanilla |
| [Molecular Graph Convolutions: Moving Beyond Fingerprints](https://arxiv.org/abs/1603.00856?context=stat) | Kearnes et al. | 2016 | None | salmad |
| [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | Kipf and Welling | 2016 | None | not application on molecules but the most influential GCN |
| [Convolutional Embedding of Attributed Molecular Graphs for Physical Property Prediction](https://pubs.acs.org/doi/10.1021/acs.jcim.6b00601) | Coley et al.| 2017 | [Github](https://github.com/connorcoley/conv_qsar_fast) | Atom features in graphs like ECFP. Not significant improvement over |    
| [Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a.html)| Gilmer et al. | 2017 |None| 🔥🔥🔥|
|[Graph Attention Networks](https://arxiv.org/abs/1710.10903)| Velickovic et al. | 2017| none| 🔥🔥🔥|
|[Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)| Hamilton et al.| 2017|none|com|
|[Learning Graph-Level Representation for Drug Discovery](https://arxiv.org/abs/1709.03741) | Li et al. | 2017|none|com|
|[Low Data Drug Discovery with One-Shot Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5408335/) | Tran | 2017|none|com|
|[MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973) | De Cao and Kiprf | 2018|none|com|
|[PotentialNet for Molecular Property Prediction](https://pubs.acs.org/doi/full/10.1021/acscentsci.8b00507) | Feinberg et al. | 2018|none|com|
|[Adaptive Graph Convolutional Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/11691) | Li et al. | 2018|none|com|
|[Junction Tree Variational Autoencoder for Molecular Graph Generation](https://proceedings.mlr.press/v80/jin18a.html) | Jin et al. | 2018|none|com|
|[Molecule Property Prediction Based on Spatial Graph Embedding](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00410) | Wang et al. | 2018|none|com|
|[Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation](https://proceedings.neurips.cc/paper/2018/file/d60678e8f2ba9c540798ebbde31177e8-Paper.pdf) | You et al. | 2018|none|com|
|[Hierarchical Graph Representation Learning with Differentiable Pooling](https://proceedings.neurips.cc/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html)| Ying | 2018|none|com|
|[Graph classification using structural attention](https://dl.acm.org/doi/pdf/10.1145/3219819.3219980) | Lee et al. | 2018|none|com|
|[Chemi-Net: A Molecular Graph Convolutional Network for Accurate Drug Property Prediction](https://www.mdpi.com/1422-0067/20/14/3389/htm) | Liu et al. | 2019|none|com|
|[Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237) | Yang et al. | 2019|none|comfire|
|[Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) | Hu et al. | 2019|none|com🔥🔥🔥|
|[CensNet: Convolution with Edge-Node Switching in Graph Neural Networks](https://www.ijcai.org/Proceedings/2019/0369.pdf) | Jiang | 2019
|[Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://pubmed.ncbi.nlm.nih.gov/31408336/) | Xiong et al. | 2020
|[Geom-GCN: Geometric Graph Convolutional Networks](https://arxiv.org/abs/2002.05287) | Pei et al. | 2020|none|com|
|[Multi-View Graph Neural Networks for Molecular Property Prediction](https://arxiv.org/abs/2005.13607) | Ma et al. | 2020|none|com|
|[ASGN: An Active Semi-supervised Graph Neural Network for Molecular Property Prediction](https://dl.acm.org/doi/pdf/10.1145/3394486.3403117)| Hao et al. | 2020|none|com|
|[MoFlow: An Invertible Flow Model for Generating Molecular Graphs](https://dl.acm.org/doi/abs/10.1145/3394486.3403104)| Zang et al. | 2020|none|com|
|[GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation](https://arxiv.org/abs/2001.09382) | Shi et al. | 2021|none|com|
|[Communicative Representation Learning on Attributed Molecular Graphs](https://www.ijcai.org/proceedings/2020/0392.pdf) | Song et al. | 2021|none|com🔥🔥🔥|
| __Variational Graph Autoencoders__| | | | |
|[Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) | Kipf amd Welling | 2016|noen|com|
|[Constrained Graph Variational Autoencoders for Molecule Design](https://arxiv.org/abs/1805.09076) | Liu et al. | 2018 🔥🔥🔥|none|com|
|[Constrained Generation of Semantically Valid Graphs via Regularizing Variational Autoencoders](https://proceedings.neurips.cc/paper/2018/file/1458e7509aa5f47ecfb92536e7dd1dc7-Paper.pdf) | Ma et al. | 2018|none|com|
|__Graph Unet__| | | | |
|[Graph U-Nets](https://arxiv.org/pdf/1905.05178.pdf)| Gao and Li |2018|none|com|
|__Graph Contrastive learning__| | | | |
|[MoCL: Data-driven Molecular Fingerprint via Knowledge-aware Contrastive Learning from Molecular Graph](https://arxiv.org/abs/2106.04509) | Sun et al. | 2021|none|com|
|[FragNet, a Contrastive Learning-Based Transformer Model for Clustering, Interpreting, Visualizing, and Navigating Chemical Space](https://www.mdpi.com/1420-3049/26/7/2065) | Shrivastava | 2021|none|com|
|[Molecular contrastive learning of representations via graph neural networks](https://www.nature.com/articles/s42256-022-00447-x) | Wang et al. \| 2022 🔥🔥🔥|none|com|
|__Graph Transformers__| | | | |
|[Graph Transformer Networks](https://arxiv.org/abs/1911.06455) | Jun et al. | 2020|none|com|
|__Reviews__ | | | | |
|[Graph convolutional networks: a comprehensive review](https://computationalsocialnetworks.springeropen.com/articles/10.1186/s40649-019-0069-y?ref=https://githubhelp.com) | Zhang et al. | 2019|none|com|
|[How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) | Xu et al. | 2019|none|com|
|[Graph convolutional networks for computational drug development and discovery](https://academic.oup.com/bib/article/21/3/919/5498046?login=true) | Sun et al. | 2020|none|com|
|[Graph neural networks: A review of methods and applications](https://www.sciencedirect.com/science/article/pii/S2666651021000012) | Zhou et al. | 2020|none|com|
|[A compact review of molecular property prediction with graph neural networks](https://www.sciencedirect.com/science/article/pii/S1740674920300305) | Wieder et al. | 2020 🔥🔥🔥|none|com|
|[Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00479-8) | Jiang et al. | 2021 🔥🔥|none|com|



## Articles
- [Understanding GNNs](https://distill.pub/2021/understanding-gnns/) :fire: :fire: :fire:
- [Introduction to Graph Neural Networks](https://theaisummer.com/graph-convolutional-networks/)
- [Graph Convolutional Networks](https://tkipf.github.io/graph-convolutional-networks/)
## Repositories
- [Chemprop: A library with MPNN and D-MPNN applications on molecular datasets](https://github.com/chemprop/chemprop)🔥

## Videos
-[ An Introduction to Graph Neural Networks: Models and Applications](https://www.youtube.com/watch?v=zCEYiCxrL_0) by Miltos ALlamanis(Microsoft Research)🔥🔥🔥
- [Intro to graph neural networks (ML Tech Talks)](https://www.youtube.com/watch?v=8owQBFAHw7E) by Petar Velickovic(DeepMind)🔥🔥🔥
- [Understanding Graph Neural Networks](https://www.youtube.com/watch?v=fOctJB4kVlM) by DeepFindr
- [The AI EPiphany](https://www.youtube.com/c/TheAIEpiphany/playlists) by Gordic Aleksa


## Tools
- [RDKit](https://www.rdkit.org/): A cheminformatics library for generating/calculating molecular descriptors and fingerprints and handling molecules
- [MoleculeNet](https://moleculenet.org/): A library for benchmarking ML models across different molecular tasks
- [DeepChem](https://github.com/deepchem/deepchem): A toolkit which includes a lot of different models and datasets with relevant tutotials for gentle introduction into molecular ML. 
- [DGL-lifesci](https://github.com/awslabs/dgl-lifesci):  A package for various applications in life science with graph neural networks.🔥🔥🔥

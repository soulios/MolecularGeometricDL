# MolecularGeometricDL
A repo of the most seminal applications of geometric deep learning in molecular tasks.

The most comprehensive professionally curated resource on Geometric Deep learning applied in molecular tasks including the best tutorials, videos, books, papers, articles, courses, websites, conferences and open-source libraries.
I am creating this resource, while conducting my PhD work at Helmholtz Centre of Environmental Research under the supervision of Dr. Jana Schor.
The papers will be listed by time order, noting the advancements along the way.

# Table of Contents 
- [Molecular representations](#Molecular-representations)
- [What is a graph?](#What-is-a-graph?)
- [Graph Convolutions](#Graph-Convolutions)
- [Tutorials](#Tutorials)
- [Papers](#Papers)
- [Articles/Blogs](#Articles)
- [Repositories](#Repositories)
- [Videos](#Videos)
- [Tools](#Tools)



## Molecular representations
A molecule can be represented in a lot of ways. As input to a machine learning model, some represntations are more popular.
Although, these representations have resulted in useful ML models for different molecular tasks, the plateau has not yet been reached.
Due to the rise of graph neural networks in the last five years, several applications involve molecular tasks.

# Molecular descriptors
"The molecular descriptor is the final result of a logic and mathematical procedure which transforms chemical information encoded within a symbolic representation of a molecule into a useful number or the result of some standardized experiment." [Handbook of molecular descriptors](https://onlinelibrary.wiley.com/doi/book/10.1002/9783527613106)

There are several open-source and proprietary tools and packages that calculate a number of descriptors and of course there is a discrepancy between them. That does not allow for uniform representations of molecules and leads to non reproducible results.
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/descriptors.png?raw=true)



# Molecular Fingerprints
Molecular fingerprints represent the molecule as a sequence of bits. The most common types of fingerprints are the substructure based, the topological and the circular ones.
- Substructure-based : The bit string depends on the presence in the compound of certain substructures or features from a given list of structural keys(MACCS,PubChem). 

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/SUBFP.jpg?raw=true)

- Topological : They work by analyzing all the fragments of the molecule following a (usually linear) path up to a certain number of bonds, and then hashing every one of these paths to create the fingerprint. 


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/TOPOLOGICALFP.jpg?raw=true)

- Circular : They are topological fingerprints but instead of looking for paths in the molecule, the environment of each atom up to a determined radius is recorded(ECFP,FCFP).


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/CIRCULARFP.jpg?raw=true)

# Smiles
The simplified molecular-input line-entry system (SMILES) is a specification in the form of a line notation for describing the structure of chemical species using short ASCII strings. SMILES strings can be imported by most molecule editors for conversion back into two-dimensional drawings or three-dimensional models of the molecules. 
Unfortunately, a molecule can be represented by several SMILES strings and SMILES do not encode 3D information about the molecule
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/SMILES.png?raw=true)



## What is a graph?
A graph G is a set of nodes and vertices between them G(V,E). Molecules can be intuitively seen as graphs where the nodes are the atoms and the edges are the bonds between them.


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/graphs.png?raw=true)


How we  use the graphs as input though?

The graph can be represented essentially by three matrices:
- The adjacency matrix, which shows how the nodes(atoms) are connected
- The node features matrix, which encodes information about every node(atom)
- The edge features matrix, whoch encodes information about the edge(bond)


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/graphmatrices.png?raw=true)



## Graph Convolutions

__Normal Convolutions__

A typical feed forward network does a forward pass with the following equation:

- __Y = &#963;(W*X + &#946;)__,

where σ is a non-linear function(ReLU,tanh), W is the weight associated with each feature, X the features and β is the bias.

In convolutional neural networks, the input usually is an image(i.e a tensor height* width*channels). 

- An RGB image has three channels whereas a greyscale only one. 

- In CNNs, the W is called a filter or kernel and is usually a matrix(2x2, 3x3 etc.) which is the same passed acrossed the image to extract features(patterns) from every part of the image.

- That is called weight sharing. That is done because a pattern is interesting wherever it is in the image(translational invariance)


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/cnns.gif)

The question became how we can generalize the convolutions to graphs?

There are some significant differences between images and graphs.
- Images are positioned in a Euclidean space, and thus have a notion of locality.Pixels that are close to each other will be much more strongly related than distant ones. Graphs on the other hand do not as information about the distance between nodes is not encoded.
- Pixels follow an order while graph nodes do not.
So, the locality is achieved in graphs based on neighborhoods.
Also, we adopt the weight sharing from the normal convolutions.
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/graphmatrices.png?raw=true)

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

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/graphtasks.png?raw=true)

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


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/adjacency.png?raw=true)

Considering this adjacency matrix, when we update the state of the node v<sub>1</sub>, we will take into account its neighbor states.
That although would be wrong as we'll be entirely dropping the previous state of node v<sub>1</sub>. 
So, we need to make a correction to the adjacency matrix A by adding the identity matrix and creating the matrix Ã.
That would add 1s across the diagonal making each node a neighbor of itself, i.e we add self-loops.

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/adjacencycorrected.jpg?raw=true)

Each latent vector of a node is a sum of the vectors of its neighbors. So, if the degree of a node( degree shows to how many neighbors a node has) is really high the scale of the latent vector would be entirely different and we'll face vanishing or exploding gradients.
- [So, we should normalize based on the degree of the node].
Firstly we calculate degree matrix, D by summing up row-wise the adjacency matrix, Ã.


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/degree.jpg?raw=true)

Then, we inverse it and thus the equation takes the form.


![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/inversedegree.jpg?raw=true)

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

The most known variation of graph convolutions was set by Kipf & Welling in 2017. 
They introduced a renormalization trick which is more than just a mere average of the neighbors.
They normalize by  1&#247;&#8730;(d<sub>i</sub> * d<sub>j</sub>)


__H<sub>k+1</sub> = &#963;(D<sup>-1/2</sup>ÃD<sup>-1/2</sup>W<sub>k</sub>H<sub>k</sub>)__

From now on, we'll refer to it as the __GCN__.

- GAT(Graph Attention Networks)

Petar Velickovic had another idea. Instead, of giving an equal weight to every neighbor that will be added explicitly, 
a concept called attention. So, the node-wise equation now became:

__h<sub>i</sub> = σ(Σ(a<sub>ij</sub>h<sub>j</sub>))__
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/GAT.png?raw=true)

The a<sub>ij</sub> comes from applying a softmax to e<sub>ij</sub> = a(h<sub>i</sub>,h<sub>j</sub>)
which are non-normalized coefficients across pairs of nodes

Influenced by the results of Vaswani et al. they included multi head attention mechanisms which is essentially a K 
number of replicates which are then concatenated or aggregated. The following figure from the paper makes it abundantly 
clear.
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/GAT-MULTI.png?raw=true)


- Gated GNN

In 2016, Li et al. introduced Gated Graph Neural Networks.
They can be summed up by the following equation.

# Message Passing Neural Nets

The term message-passing arised in 2017 and  is really intuitive way to see graph neural nets.
The two main points evolve around the two functions that happen in a GNN
- The Update function, q
- The Aggregate function, U
From this [youtube video](https://www.youtube.com/watch?v=zCEYiCxrL_0) we can sum them up by the following equation.

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/MPNN.png?raw=true)

and we can see the clearly the variants that arise from this.

![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/GCNMP.png?raw=true)
![alt text](https://github.com/soulios/MolecularGeometricDL/blob/main/GGNN.png?raw=true)




## Tutorials


## Papers

- [Convolutional Networks on Graphs for Learning Molecular Fingerprints](https://arxiv.org/abs/1509.09292) by Duvenaud et al. in 2015.
- [Molecular Graph Convolutions: Moving Beyond Fingerprints](https://arxiv.org/abs/1603.00856?context=stat) by Kearnes et al. in 2016
-  [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) by Kipf and Welling in 2016, not application on molecules but the most influential GCN 🔥🔥🔥
- [Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a.html) by Gilmer et al. in 2017🔥🔥🔥
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) by Velickovic et al. in 2017🔥🔥🔥
- [Inductive Representation Learning on Large Graphs](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf) by Hamilton in 2017
- [Learning Graph-Level Representation for Drug Discovery](https://arxiv.org/abs/1709.03741) by Li et al. in 2017
- [Low Data Drug Discovery with One-Shot Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5408335/) by Tran in 2017
- [MolGAN: An implicit generative model for small molecular graphs](https://arxiv.org/abs/1805.11973) by De Cao and Kiprf in 2018
- [PotentialNet for Molecular Property Prediction](https://pubs.acs.org/doi/full/10.1021/acscentsci.8b00507) by Feinberg et al. in 2018
- [Adaptive Graph Convolutional Neural Networks](https://ojs.aaai.org/index.php/AAAI/article/view/11691) by Li et al. in 2018
- [Junction Tree Variational Autoencoder for Molecular Graph Generation](https://proceedings.mlr.press/v80/jin18a.html) by Jin et al. in 2018
- [Molecule Property Prediction Based on Spatial Graph Embedding](https://pubs.acs.org/doi/10.1021/acs.jcim.9b00410) by Wang et al. in 2018
- [Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation](https://proceedings.neurips.cc/paper/2018/file/d60678e8f2ba9c540798ebbde31177e8-Paper.pdf) by You et al. in 2018
- [Hierarchical Graph Representation Learning with Differentiable Pooling](https://proceedings.neurips.cc/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html)by Ying in 2018
- [Graph classification using structural attention](https://dl.acm.org/doi/pdf/10.1145/3219819.3219980) by Lee et al. in 2018
- [Chemi-Net: A Molecular Graph Convolutional Network for Accurate Drug Property Prediction](https://www.mdpi.com/1422-0067/20/14/3389/htm) by Liu et al. in 2019
- [Analyzing Learned Molecular Representations for Property Prediction](https://pubs.acs.org/doi/full/10.1021/acs.jcim.9b00237) by Yang et al. in 2019🔥🔥🔥
- [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) by Hu et al. in 2019🔥🔥🔥
- [CensNet: Convolution with Edge-Node Switching in Graph Neural Networks](https://www.ijcai.org/Proceedings/2019/0369.pdf) by Jiang in 2019
- [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://pubmed.ncbi.nlm.nih.gov/31408336/) by Xiong et al. in 2020
- [Geom-GCN: Geometric Graph Convolutional Networks](https://arxiv.org/abs/2002.05287) by Pei et al. in 2020 
- [Multi-View Graph Neural Networks for Molecular Property Prediction](https://arxiv.org/abs/2005.13607) by Ma et al. in 2020
- [ASGN: An Active Semi-supervised Graph Neural Network for Molecular Property Prediction](https://dl.acm.org/doi/pdf/10.1145/3394486.3403117)by Hao et al. in 2020
- [MoFlow: An Invertible Flow Model for Generating Molecular Graphs](https://dl.acm.org/doi/abs/10.1145/3394486.3403104)by Zang et al. in 2020
- [GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation](https://arxiv.org/abs/2001.09382) by Shi et al. in 2021
- [Communicative Representation Learning on Attributed Molecular Graphs](https://www.ijcai.org/proceedings/2020/0392.pdf) by Song et al. in 2021🔥🔥🔥

# Variational Graph Autoencoders
- [Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) by Kipf amd Welling in 2016
- [Constrained Graph Variational Autoencoders for Molecule Design](https://arxiv.org/abs/1805.09076) by Liu et al. in 2018 🔥🔥🔥
- [Constrained Generation of Semantically Valid
Graphs via Regularizing Variational Autoencoders](https://proceedings.neurips.cc/paper/2018/file/1458e7509aa5f47ecfb92536e7dd1dc7-Paper.pdf) by Ma et al. in 2018
# Graph Unet
-  [Graph U-Nets](https://arxiv.org/pdf/1905.05178.pdf) by Gao and Li in 2018

# Graph Contrastive learning
- [MoCL: Data-driven Molecular Fingerprint via Knowledge-aware Contrastive Learning from Molecular Graph](https://arxiv.org/abs/2106.04509) by Sun et al. in 2021
- [FragNet, a Contrastive Learning-Based Transformer Model for Clustering, Interpreting, Visualizing, and Navigating Chemical Space](https://www.mdpi.com/1420-3049/26/7/2065) by Shrivastava in 2021
- [Molecular contrastive learning of representations via graph neural networks](https://www.nature.com/articles/s42256-022-00447-x) by Wang et al. in 2022 🔥🔥🔥
# Graph Transformers
- [Graph Transformer Networks](https://arxiv.org/abs/1911.06455) by Jun et al. in 2020

# Reviews 
- [Graph convolutional networks: a comprehensive review](https://computationalsocialnetworks.springeropen.com/articles/10.1186/s40649-019-0069-y?ref=https://githubhelp.com) by Zhang et al. in 2019
- [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) by Xu et al. in 2019
- [Graph convolutional networks for computational drug development and discovery](https://academic.oup.com/bib/article/21/3/919/5498046?login=true) by Sun et al. in 2020
- [Graph neural networks: A review of methods and applications](https://www.sciencedirect.com/science/article/pii/S2666651021000012) by Zhou et al. in 2020
- [A compact review of molecular property prediction with graph neural networks](https://www.sciencedirect.com/science/article/pii/S1740674920300305) by Wieder et al. in 2020 🔥🔥🔥
- [Could graph neural networks learn better molecular representation for drug discovery? A comparison study of descriptor-based and graph-based models](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00479-8) by Jiang et al. in 2021 🔥🔥



## Articles

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

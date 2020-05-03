---
title: "Paper Summary: Stiffness, A New Perspective on Generalization in Neural Networks"
date: 2019-03-29T00:00:55+03:00
draft: False
layout: post
description: Measuring how groupped the data is during training.
---

This is a summary of [Stiffness: A New Perspective on Generalization in Neural Networks](https://arxiv.org/abs/1901.09491). TLDR below.


| Author | Organization | Previous work |
| ---    |  ----        | ------         |
| [Stanislav Fort](http://stanford.edu/~sfort1/) | Google AI Resident, Google AI Zurich | [The Goldilocks zone: Towards better understanding of neural network loss landscapes](https://arxiv.org/pdf/1807.02581.pdf) |
| Paweł Krzysztof Nowak | Google AI Resident| |
| [Srini Narayanan](https://ai.google/research/people/SriniNarayanan) | Google AI Resident| [Points, Paths, and Playscapes: Large-scale Spatial Language Understanding Tasks Set in the Real World](https://ai.google/research/pubs/pub47017) |

---


### Stiffness?

This paper aims at improving our understanding of how neural networks generalize from the point of view of *stiffness*. The intuition behind stiffness is how a gradient update on one point affects another:

> [it] characterizes the amount of correlation between changes in loss on the two due to the application of a gradient update based on one of them. (4.1, Results and discussion)

Stiffness is expressed as the expected sign of the gradients `g`:

![Formula 5, page 2]({{ site.baseurl }}/images/stiffness/formula_stiffness.png "Formula 5 - arxiv.org/abs/1901.09491")

A weight update that improves the loss for **X_1** and **X_2** is stiff and characterized as anti-stiff if the loss beneficiate for one of the points and doesn't help the other.

![Figure 1, page 3]({{ site.baseurl }}/images/stiffness/figure1_stiffness_overview.png "Figure 1 - arxiv.org/abs/1901.09491" )

The question is now how do we choose **X_1** and **X_2**. Authors explore two ways: by class membership or by distance.

### Stiffness based on class membership

We can look at how a gradient update on a point in class A will affect another point's loss belonging to class B. In the paper they craft a *class stiffness matrix*, which is the average of stiffness between each point grouped by class:

![Formula 6, page 3]({{ site.baseurl }}/images/stiffness/formula06-class-membership.png "Formula 6 - arxiv.org/abs/1901.09491")

The diagonal of this matrix represent the model's within class generalization capability. You can find an example of stiffness class matrix at different steps of the training stage:

![Figure 6, page 5]({{ site.baseurl }}/images/stiffness/figure06-page5.png "Figure 6 - arxiv.org/abs/1901.09491")

At early stages, the stiffness is high between members of the same classes (hence the red diagonal). The majority of the cells raises their stiffness until reaching the point of overfitting: stiffness reaches 0.

### Stiffness as a function distance and learning rate
Stiffness is then studied through the distance lens, they distinguish two kinds of distance: pixel-wise (in the input space) and layerwise (in the representational space).

![Figure 9, page 6]({{ site.baseurl }}/images/stiffness/figure9_depending_on_distance.png "Formula 9 - arxiv.org/abs/1901.09491")

> The general pattern visible in Figure 9 is that there exists a critical distance within which input data points tend to move together under gradient updates, i.e. have positive stiffness. This holds true for all layers in the network, with the tendency of deeper layers to have smaller stiff domain sizes.

Authors define stiff regions as "regions of the data space that move together when a gradient update is applied".

![Figure 10, page 7]({{ site.baseurl }}/images/stiffness/figure10-stiffdomain-learningrate.png "Formula 10 - arxiv.org/abs/1901.09491")

We can see that a higher learning rate increase the size of the stiff regions which suggests that higher learning rates help generalization.

---

### TLDR

- Stiffness quantify how much gradient update on one group of point affects another
- Stiffness is tightly linked to generalization
- Stiffness tends to 0 when the system overfit
- Higher learning rate increases the area under which points are moving together

---

Complementary resources:

- Manifold Mixup: Better Representations by Interpolating Hidden States - https://arxiv.org/abs/1806.05236 (cited in the article)

---
title: "Overview of One Year of Lottery Ticket Research"
date: 2020-02-13T22:39:28+02:00
draft: false
layout: post
---

Winning tickets were [discovered in March 2018](https://arxiv.org/abs/1803.03635) and presented at ICRL the same year. It drawed a lot of attention. It sheds light on yet unknown underlying properties of neural networks and seems to be one of the keys for faster training and smaller models. Overall flipping on the head how we approach neural net architecture design. 

Winning tickets in deep learning were mentioned as one of the most important topics of 2019 by Lex Fridman's in his [Deep Learning State of the Art 2020](https://youtu.be/0VH1Lim8gL8?t=2761) (awesome) lecture. This article aims at summarizing what I understood after reading about it. Hope you'll enjoy it.

# Pruning

It is known that DL models have generally heavy computational requirements and can be blocking in particular settings. ResNet, for instance, requires 50M operations for one single inference. They've been efforts to reduce the number of parameters with [quantization](https://pytorch.org/docs/stable/quantization.html),
[knowledge distillation](https://towardsdatascience.com/what-is-knowledge-distillation-41bb77a3c6d8) and [pruning](https://github.com/he-y/Awesome-Pruning).

Pruning removes the *least important* weights or channels. Least important can mean the one with the smallest magnitudes or other heuristics. Such a technique is working well and can reduce up to 90% of the weights in a network while preserving most of the original accuracy. While pruning can help to reduce the model's size, it won't help training it faster. It is generally a post-processing step, after training. Retraining a pruned model won't yield the same results as if you prune after training. If it were possible to be able to train the pruned model directly, train faster without sacrificing performances.

But in their paper [Jonathan Frankle](https://arxiv.org/search/cs?searchtype=author&query=Frankle%2C+J), [Michael Carbin](https://arxiv.org/search/cs?searchtype=author&query=Carbin%2C+M) experimentally found that instead of training large networks and then reduce their size we might be able to train smaller networks upfront:

> dense, randomly-initialized, feed-forward networks contain subnetworks ("winning tickets") that - when trained in isolation - reach test accuracy comparable to the original network in a similar number of iterations.

-------

# Winning Tickets

In order to find winning tickets, initialization seems to be the key:

> When their parameters are randomly reinitialized [...], our winning tickets no longer match the performance of the original network, offering evidence that these smaller networks do not train effectively unless they are appropriately initialized.

They found that we can train a pruned model again after re-initializing the weights with the original model's parameters. This gives systematically better results than re-initializing randomly. Doing this process multiple times is called iterative pruning (with no re-init):

```
1. Randomly initialize a neural network [with weights θ0]
2. Train the network for j iterations, arriving at parameters θj 
3. Prune [by magnitude] p% of the parameters in θj , creating a mask m
4. Reset the remaining parameters to their values in θ0
5. Goto 2
```

If the subnetwork produced by this technique matches the original network's performances, it is called a winning ticket. The following graph represents averaged results of five runs on a LeNet (fully dense) network on the MNIST dataset. This model was pruned in different ways:

- [blue] Done with the recipe above
- [orange] Same as blue but replace step 4 by "Randomly initialize the remaining parameters"
- [red] Same as the orange line without step 5
- [green] Same as blue without step 5

![Figure 4-b of 803.03635]({{ site.baseurl }}/images/winning-ticket/figure4-b.png)

 We can see that step 4 is the key as the green and blue lines are consistently performing better and are trained faster than randomly re-initialized networks. They also found similar results with convolutional networks like VGG and ResNet on MNIST and CIFAR10 (there are *many* more details in the [original paper](https://arxiv.org/abs/1803.03635)).

----

# Pruning Early

But the method above seems to struggle against deeper networks. In a [follow-up paper](https://arxiv.org/abs/1903.01611) (March 2019), the authors changed slightly the way the remaining parameters are reset (step 4):

> Rather than set the weights of a winning ticket to their original initializations, we set them to the weights obtained after a small number of training iterations (*late resetting*). Using late resetting, we identify the first winning tickets for Resnet-50 on Imagenet.

The graph below plot performances against different levels of sparsity of deep models rewound (iteration at which we reset the weights) with different values. We can see that rewinding at iteration 0 does not perform better than the original network whereas rewinding at higher iteration does:

![]({{ site.baseurl }}/images/winning-ticket/figure8-followup.png)

Those deeper models were resisting the winning ticket recipe above but found something interesting after looking at their *stability*:

- *Stability to pruning*: "the distance between the weights of a
subnetwork trained in isolation and the weights of the same subnetwork when trained within the larger network". Which captures "a subnetwork’s ability to train in isolation and still reach the same destination as the larger network". If a neuron is stable it won't be much affected by its neighbors disappearing through masking.
- *Stability to data order*: "the distance between the weights of two copies of a subnetwork trained with different data orders". Which captures " a subnetwork’s intrinsic ability to consistently reach the same destination despite the gradient noise of SGD".

The table below shows stability for different networks. *Warmup* means that the learning rate is scheduled to increase slowly during training, possibly reducing the noise of the optimizer. *IMP* is the original recipe to generate winning tickets:

![]({{ site.baseurl }}/images/winning-ticket/figure3-followup.png)

We can see that IMP fails at fiding winning tickets in deeper networks without changing the learning rate. We can also see that there's a link between performances and the stabilities measures. "Winning tickets are more stable than the random subnetworks".

---

# What about other domains?

So far winning tickets have been tested on the same datasets and on computer vision tasks. One can ask if this isn't just drastic overfitting or if the winning tickets transfer at all. 

Facebook [published a paper](https://arxiv.org/abs/1906.02773) (June 2019) tested the winning ticket evaluation and transfer across six visual datasets. For instance, testing generating winning tickets on ImageNet and testing it others (like CIFAR-100):

![figure 4-e of 1906.02773]({{ site.baseurl }}/images/winning-ticket/figure4-e.png)

They observed that winning tickets generalize across all datasets with (at least) close performances than the original one. And that winning tickets generated on larger datasets generalized better than the other ones, probably due to the number of classes in the original model. Finally, this paper also tested the transfer successfully across different optimizers successfully.

What about other tasks than image classification? Facebook [published in parallel a paper](https://arxiv.org/abs/1906.02768) (June 2019) that test the winning ticket in reinforcement learning and NLP tasks.

> For NLP, we found that winning ticket initializations beat random tickets both for recurrent LSTM models trained on language modeling and Transformer models trained on machine translation. [...] For RL, we found that winning ticket initializations substantially outperformed random tickets on classic control problems and for many, but not all, Atari games.

----

# TLDR
- Many neural networks are over-parameterized
- Frankle & Carbin found simple algorithm to find smaller network within larger ones
- Those sub-networks are trainable from scratch and can perform at least as well and often better 
- What makes winning tickets special is still unclear but seems to be a critical step toward a deeper understanding of the underlying properties of neural nets

-----

## Sources
- [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
- [Stabilizing the Lottery Ticket Hypothesis](https://arxiv.org/abs/1903.01611)
- [One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers](https://arxiv.org/abs/1906.02773)
- [Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP](https://arxiv.org/abs/1906.02768)
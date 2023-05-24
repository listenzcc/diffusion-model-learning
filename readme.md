
# 扩散模型入门（一）

扩散模型是近年来比较热门的神经网络模型，我认为所谓扩散模型就是对扩散过程进行数学建模，并且能够逆转扩散过程的数学方法。本文将开始从一个初学者的视角尝试理解它的思想和应用。

本文使用的 Toy demo 可见我的在线笔记本

[Diffusion of the curve](https://observablehq.com/@listenzcc/diffusion-of-the-curve)

---


## 何为扩散

所谓扩散就是布朗运动，体现在数学概念上，我们可以将它理解成数据样本点的分布向标准正态分布不断靠拢的过程。这个过程如下图所示，左图中的曲线代表一组高维数据，它拥有连续维度，不同维度的采样值曲线上蓝色点所示，它们的分布如右侧红色点所示。而经过“扩散”之后，采样值的分布逐渐符合标准正态分布，如右图所示。

![Untitled](%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8%EF%BC%88%E4%B8%80%EF%BC%89%209bf4745620354c4a819dc19e85867a83/Untitled.png)

![Untitled](%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8%EF%BC%88%E4%B8%80%EF%BC%89%209bf4745620354c4a819dc19e85867a83/Untitled%201.png)

## 何为扩散模型

我认为所谓扩散模型就是对上述扩散过程进行数学建模，并且能够逆转扩散过程的数学方法。我觉得这方面介绍得比较清楚的论文是以下这篇，虽然题目中没有扩散的字眼，但“非平衡的热动力学”这个名词非常写意，甚至有些浪漫。

![Untitled](%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8%EF%BC%88%E4%B8%80%EF%BC%89%209bf4745620354c4a819dc19e85867a83/Untitled%202.png)

[Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585)

热力学定律告诫我们，扩散无法避免、熵增无法避免，也就是说宇宙总是向着更加混乱和无序发展的，而维持有序则需要外界不断对某个系统施加能量。那么这个无序的状态可以看作是变量服从正态分布，而有序的低熵状态则代表上述的曲线。它上面的采样值服从未知但确定的分布。

从例子中可以想见，一条有序的曲线经过若干次简单的计算后即可“退回”到正态分布的混沌状态，且这个过程是连续的，或局部可微的。因此，我们同样有理由相信这个过程是“可逆的”，只需要“学会”每一步扩散的微分步骤，就可以在数学上把这个物理上难以维持的过程给反转过来。

为了这个目标所做的一切努力，就是扩散模型，它的优势有四点：（深度网络）模型结构不限、能够实现采样、易与其他概率密度结合、易与样本状态结合。

> 1. extreme flexibility in model structure,
2. exact sampling,
3. easy multiplication with other distributions, e.g. in order to compute a posterior, and
4. the model log likelihood, and the probability of individual states, to be cheaply evaluated.
> 

## 附录：扩散过程的简要数学说明

为了表达方便，我们将采样值（曲线）的分布表示为

$$
x_0 \sim \Psi(x)
$$

它代表一个确定但未知的分布。而我们认为经过一段时间的随机扩散后，新分布属于正态分布

$$
x_T \sim \mathcal{N}(\mu, \Sigma)
$$

为了简单起见，通常假设它服从标准正态分布

$$
\begin{cases}
\mu = 0 \\
\Sigma = I
\end{cases}
$$

因此，扩散过程可以表示成变量连续变化的过程

$$
x_0, x_1, x_2, ..., x_T
$$

直觉上来讲，它可以是一条 Markov 链，其转移概率为

$$
p(x_t) = p(x_{t-1}) \cdot p(x_t|x_{t-1})
$$

连续变化过程如下图所示，希望它看上去有一些动态的感觉。

![Untitled](%E6%89%A9%E6%95%A3%E6%A8%A1%E5%9E%8B%E5%85%A5%E9%97%A8%EF%BC%88%E4%B8%80%EF%BC%89%209bf4745620354c4a819dc19e85867a83/Untitled%203.png)
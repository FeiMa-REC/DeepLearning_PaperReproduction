# ResNet

## 创新点：
- 提出 Residual 结构（残差结构），并搭建超深的网络结构（可突破1000层）
- 使用 Batch Normalization 加速训练（丢弃dropout）

### 为什么使用Residual

在ResNet网络提出之前，传统的卷积神经网络都是通过将一系列卷积层与池化层进行堆叠得到的。一般我们会觉得网络越深，特征信息越丰富，模型效果应该越好。但是实验证明，当网络堆叠到一定深度时，会出现两个问题：

* 1、**梯度消失或梯度爆炸**：若每一层的误差梯度小于1，反向传播时，网络越深，梯度越趋近于0
反之，若每一层的误差梯度大于1，反向传播时，网路越深，梯度越来越大。
* 2、**退化问题(degradation problem)**：在解决了梯度消失、爆炸问题后，仍然存在深层网络的效果可能比浅层网络差的现象。

如下图所示，当单纯的通过堆叠卷积层和池化层加深网络，我们可以发现其效果没有浅层网络好。

![](./NoteImgs/1.png)

而对于上诉两个问题，ResNet论文中给出了两个解决方法：
>对于梯度消失或梯度爆炸问题，ResNet论文提出通过数据的预处理以及在网络中使用 BN（Batch Normalization）层来解决。

>对于退化问题，ResNet论文提出了 residual结构（残差结构）来减轻退化问题，下图是使用residual结构的卷积网络，可以看到随着网络的不断加深，效果并没有变差，而是变的更好了。（虚线是train error，实线是test error）

![](./NoteImgs/2.png)

### 残差结构

对于不同深度的ResNet对于使用两种不同的残差结构。

下图中左侧残差结构称为 BasicBlock(适用于浅层网络)，右侧残差结构称为 Bottleneck(适用于深层网络)：

![](./NoteImgs/3.png)

**对于深层的 Bottleneck，1×1的卷积核起到降维和升维（特征矩阵深度）的作用，同时可以大大减少网络参数。**

![](./NoteImgs/4.png)

观察下图的 ResNet18层网络，可以发现有些残差块的 short cut 是实线的，而有些则是虚线的。

这些虚线的 short cut 上通过1×1的卷积核进行了维度处理（特征矩阵在长宽方向降采样，深度方向调整成下一层残差结构所需要的channel）。

![](./NoteImgs/5.png)

下图是原论文给出的不同深度的ResNet网络结构配置，注意表中的残差结构给出了主分支上卷积核的大小与卷积核个数，表中 残差块×N 表示将该残差结构重复N次。

![](./NoteImgs/6.png)

原文的表注中已说明，**conv3_x, conv4_x, conv5_x所对应的一系列残差结构的第一层残差结构都是虚线残差结构**。因为这一系列残差结构的第一层都有调整输入特征矩阵shape的使命（将特征矩阵的高和宽缩减为原来的一半，将深度channel调整成下一层残差结构所需要的channel）

需要注意的是，**对于ResNet50/101/152，其实conv2_x所对应的一系列残差结构的第一层也是虚线残差结构**，因为它需要调整输入特征矩阵的channel。根据表格可知通过3x3的max pool之后输出的特征矩阵shape应该是[56, 56, 64]，但conv2_x所对应的一系列残差结构中的实线残差结构它们期望的输入特征矩阵shape是[56, 56, 256]（因为这样才能保证输入输出特征矩阵shape相同，才能将捷径分支的输出与主分支的输出进行相加）。所以第一层残差结构需要将shape从[56, 56, 64] --> [56, 56, 256]。注意，这里只调整channel维度，高和宽不变（而conv3_x, conv4_x, conv5_x所对应的一系列残差结构的第一层虚线残差结构不仅要调整channel还要将高和宽缩减为原来的一半）。


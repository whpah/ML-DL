# 机器学习~2

## 1-1神经网络

在生物学中，一个神经元可以接受来自多个神经元的电信号，在处理后向其它神经元输出，作为别的神经元的输入。
因此：
![image-20220712171529380](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220712171529380.png)

每一层都可以用一个向量来表示，在中间的隐藏层中输出的时激活向量，最后得出一个值。之前我们讲过，在预估房子的价格时，特征有x1，x2对应长和宽，但这样并不是最合适的，我们可以令x3=x1*x2得到面积进行预估，此时我们是手动进行特征的选择的。而神经网络则是自行选择最适合的特征进行后续的运算。
神经网络结构的问题就是，要有多少个隐藏层，每个隐藏层要有多少个神经元。

这种多层神经网络被称为多层感知器![image-20220712172938160](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220712172938160.png)

## 例子：图像识别

![image-20220717164054489](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220717164054489.png)

一张图片会以每一像素的亮度形成一个向量，将此向量输入，第一层的神经元会先识别图像中是否存在线条之类的特征，若存在，将其聚合在一起，作为向量的一个新的元素，第二层的神经元则是检测更大范围的特征是否存在，若存在则将其聚合起来，形成向量的一个新的元素，最后则是对比与原图的差距，得出原图是什么。即输入不同的数据，神经网络就会自动学习检测不同的特征，从而完成图像识别，或者判断输入是否有某个特定事物的预测任务。

## 1-2神经网络中的层

![image-20220717170533633](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220717170533633.png)

w2^[1] 表示第1层的第二个神经元的逻辑回归函数的参数，所有神经元计算出的结果组成向量作为下一层神经元的输入，即向量a^[1]表示第一层的输出，这样循环下去直至最后一层进行阈值判断，从而得出结论。!!最后一层得出的a是一个值，并不是一个向量。![image-20220717171309868](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220717171309868.png)

![image-20220717173110060](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220717173110060.png)

a[i]^[j]表示第j层的第i个神经元，进行计算后得出的结果，而向量a[j]表示第j层输出的向量

## TensorFlow实现推理模型

![image-20220721120953331](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721120953331.png)

x定义为一个包含了样本特征的向量，layer_1表示1层，dense表示这是个稠密层，units表示此层有15个神经元，activation表示激活函数的类型为逻辑回归，a2=表示将向量a1进行算法运算后得出的新的向量

![image-20220721134250912](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721134250912.png)

输出a1得到tf.tensor（...） 这表示a1是一个一行三列的矩阵，数据类型是浮点型
当然，你也可以将tensor形式转换为numpy类型即通过a1.numpy()

## 建立一个神经网络（TensorFlow）

## 如何只利用Numpy以及Python实现向前传播

![image-20220721142735406](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721142735406.png)

x是一个numpy数组，存放一个样本。接着我们分别求出每一个神经元的代价函数参数w1_1,w1_2,w1_3，之后计算出逻辑函数<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721142927710.png" alt="image-20220721142927710" style="zoom: 33%;" />的z函数值，再将z带入回归函数中得出最后的的a1_1,a1_2,a1_3。将a1_1,a1_2,a1_3组合起来形成一个新的numpy数组，作为下一层的输入，持续上述操作直至得出一个结果，再进行阈值处理，得出结论。

![image-20220721144946839](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721144946839.png)

a.shape[0]  表示矩阵的行数
a.shape[1]  表示矩阵的列数
[ : , j ] 表示输出W矩阵的第 j+1 列

大写字母表示矩阵，小写字母表示向量或者标量，dense函数定义的是如何输出一层的结果

向前传播向量化表示：

![image-20220721151317331](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721151317331.png)



![image-20220721153238928](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721153238928.png)

步骤一指定模型，它告诉TensorFlow如何计算推断
步骤二编译模型，用一个特定的损失函数，此处使用的是<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721153014944.png" alt="image-20220721153014944" style="zoom: 50%;" />和二元交叉熵基本相同
第三步训练模型，基于样本以及梯度下降的次数进行训练,调用函数去最小化神经网络的代价函数

## 模型训练对比以及具体步骤

![image-20220721164442811](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721164442811.png) 

step1:设计模型函数--f_w,b(x)
step2:设计代价函数--J(w,b)
step3:梯度下降求最适模型参数 w=w-...    b=b-...

![image-20220721165111188](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721165111188.png)

step1：指定了神经网络的整体架构，即该网络具有几层，每一层有几个神经元，每个神经元的激活函数又选择什么（f_w,b(x)）
step2:选择合适的代价函数<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721193337897.png" alt="image-20220721193337897" style="zoom:50%;" />这个式子在TensorFlow中被称为**二元交叉熵**

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721194248338.png" alt="image-20220721194248338" style="zoom:50%;" />

step3:通过fit（）函数实现反向传播即求导的链式法则，再进行迭代一百次之后就得出了一个很小的代价函数<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721195031377.png" alt="image-20220721195031377" style="zoom:50%;" />这个等价于

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220721195120056.png" alt="image-20220721195120056" style="zoom:50%;" />

## 新的激活函数 ReLU

![img](https://img-blog.csdnimg.cn/20190417150129745.png)

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220722094534094.png" alt="image-20220722094534094" style="zoom: 50%;" />

## 如何选择激活函数

根据问题的类型进行选择，二分问题就用逻辑回归，线性问题就用RuLU

## 为什么我们需要激活函数？

1、对于y=ax+b 这样的函数，当x的输入很大时，y的输出也是无限大小的，经过多层网络叠加后，值更加膨胀的没边了，这显然不符合我们的预期，很多情况下我们希望的输出是一个概率。
2、线性的表达能力太有限了，即使经过多层网络的叠加，y=ax+b无论叠加多少层最后仍然是线性的，增加网络的深度根本没有意义。

##### 因此我们需要对线性输出作非线性变化：

激活函数的作用:激活函数是用来加入非线性因素的，因为线性模型的表达能力不够。这个变换要满足什么条件？
1、非线性，这样增加网络的深度才有意义
2、可导的，不然怎么做梯度下降
3、易于计算的
4、输出空间最好是有限的，这条好像也不是必须的，Relu就没有遵循这条

**不要在神经网络的隐藏层中使用线性激活函数**，这相当于是啥都没干（详情如图），建议使用ReLU，它可以做的更好![image-20220722102100352](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220722102100352.png)
计算至最后向量a2仍被看作是一个线性函数，所以多层神经网络并不能提升计算复杂特征的能力，不能学习任何比线性函数更复杂的东西，还不如直接使用线性回归呢。![image-20220722102432571](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220722102432571.png)

前三层用线性激活函数，最后一层用逻辑回归激活函数，仍然相当于什么都没用，不如直接逻辑回归方便。综上所述，我们在隐藏层不能全部使用线性激活函数。

## 多分类问题

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220722104439851.png" alt="image-20220722104439851" style="zoom:50%;" />

y有多种可能，不再是二分问题

## Softmax回归算法

![image-20220722105748774](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220722105748774.png)

Softmax的损失函数![image-20220722114004395](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220722114004395.png)

## 可以进行多分类的神经网络 

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220722115640884.png" alt="image-20220722115640884" style="zoom:50%;" />

softmax的激活函数，他不像是ReLU或者是sigmod仅仅与zi相关，softmax是与zi-zj都有关系，比如说它存在着10个类别那就是需要计算出z1-z10，然后才能计算出ai的值

在tensorflow中实现softmax的分类操作：（这是个相对复杂的版本 实际开发中不要使用 后面会讲更简单的版本）![image-20220722120515162](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220722120515162.png)

更简单的版本，但是我看不懂（第二部分p32）![image-20220723142025804](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220723142025804.png)

## 多标签分类

**首先明白，多标签分类和多类分类是两个概念。**
**二分类**：表示分类任务中有两个类别，比如我们想识别一幅图片是不是猫。也就是说，训练一个分类器，输入一幅图片，用特征向量x表示，输出是不是猫，用y=0或1表示。二类分类是假设每个样本都被设置了一个且仅有一个标签 0 或者 1。

**多类分类(Multiclass classification)**: 表示分类任务中有多个类别, 比如对一堆水果图片分类, 它们可能是橘子、苹果、梨等. 多类分类是假设每个样本都被设置了一个且仅有一个标签: 一个水果可以是苹果或者梨, 但是同时不可能是两者。

**多标签分类(Multilabel classification)**: 给你一张图片，问你这张图片中有没有狗，人，猫。即三个标签，输出的结果为[有狗（1），有猫（1），没有人（0）]

## 比梯度下降更优秀的优化算法Adam算法

梯度下降的目的是寻找更适合模型的参数w，b<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220723144544150.png" alt="image-20220723144544150" style="zoom:25%;" />

Adam算法可以自动调节学习率，当学习率小了会将其自动变大，大了会将其自动变小。

## 卷积层

卷积层的每个神经元仅关注输入图像的一个区域，如果在神经网络中存在多个卷积层即被成为卷积神经网络（CNN）
![image-20220724133017487](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220724133017487.png)

## 通过诊断方法来评估算法的性能

取一个数据集，把它分为训练集和一个单独的测试集，可以系统评估你的学习成果。通过计算J_test和J_train，你可以衡量模型在测试集和训练集上的表现。

![image-20220724140631254](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220724140631254.png)

即将一个样本集分为两部分，七三开或者八二开都可以，得出的结果越小说明，模型越准确。同时我们也可以通过以下操作来进一步提高诊断的速率与效果：
![image-20220724140857169](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220724140857169.png)

即设置一个阈值0.5，计算出概率，若该概率>0.5则为1，反之为0，然后对比预测值与真实值是否相等，选取不相等的继续存放在对应的测试集和训练集中。因为预测值和真实值相等的话进行训练和测试并没有太大的意义。

## 如何修改和测试程序以进行模型选择？（更好的诊断算法）

较好的方式将样本分为交叉验证集、测试集、训练集

![image-20220724144245075](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220724144245075.png)

选择不同参数d，即模型的阶数，通过训练集来训练模型得到每一个模型的拟合参数wi，bi  ，然后用交叉验证集对每一个模型进行运算，寻找Jcv（w,b）最小的那个模型，最后通过测试集来对这个模型泛化误差进行公平估计。在模型决策之前不要看测试集，因为这确保测试集是公平的，而不会过于乐观地估计你模型的泛化能力。

## 方差和偏差

https://blog.csdn.net/weixin_42327752/article/details/121428875

方差（Variance）：指预测值之间的离散程度
偏差（Bias）：指预测值和真实值之间的误差

<img src="https://img-blog.csdnimg.cn/e2ff3ce37c64413ba7a4cdb91a8d8435.png#pic_center" alt="在这里插入图片描述" style="zoom:25%;" />

对于样本数据，如果选择的模型过于简单，学不到很多信息，此时模型的预测值和真实值误差很大，也就是偏差很大，随着模型的复杂度提升，学到的信息也越来越多，使得偏差逐渐降低。
同样的，随着模型复杂的提升，数据相对模型而言变得简单，使得模型学到了更多的数据噪音，方差也就越来越大。

泛化误差 = 数据本身噪声 + 偏差 + 方差

如下图蓝线,所以需要在中间位置找到一个合适的模型复杂度，使得泛化误差尽可能地小。过于简单导致欠拟合，过于复杂导致过拟合。
这也就是我们常说的训练误差随着模型复杂度地提升而降低，而泛化误差会逐渐增大。训练误差更多和偏差相关，偏差越小，模型就越能拟合训练数据。

<img src="https://img-blog.csdnimg.cn/a728c064d5ce4c8b90fbf68c22e327de.png" alt="img" style="zoom:33%;" />

<img src="https://img-blog.csdnimg.cn/02c70649445b45adbebcfc967ea23f9c.png" alt="img" style="zoom: 67%;" />

![image-20220724154716521](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220724154716521.png)分别为高偏差，正常，高方差

高偏差意味着算法在训练集上表现不好 欠拟合
高方差意味着算法在交叉验证集和训练集上变现的更差 过拟合
当然也存在高偏差和高方差一起存在的情况，即在前一段过拟合，后一段欠拟合
![image-20220724154146873](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220724154146873.png)

## 通过正则化来对降低方差和偏差

通过正则化求得λ的值

![image-20220724173803672](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220724173803672.png)

选取不同的 λ ，在一个范围中，每一次*2 ，通过这些不同的正则化参数来拟合参数wi,bi，然后通过交叉验集（J_cv）来运算得到对应的值，最小的对应的λ即为最合适的。如果想得到算法的泛化误差，可以通过J_test

![image-20220724175923551](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220724175923551.png)

λ减小，J_train减小，方差减小，J_cv减小，偏差减小，在中间是相对较合适的，方差和偏差都比较正常。

![image-20220725100010339](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220725100010339.png)

## 学习曲线

训练集个数较小的时候，二次函数可以很好地拟合所有的训练样本，而交叉误差则会很大，因为我们只是拟合了少部分样本。当训练集的个数增加是，二次函数不能完美地拟合所有的训练样本 ，所以训练误差增加，而因为模型在此时可以更好地拟合大部分的训练样本了，所以J_cv也就随之下降。

![image-20220725104820004](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220725104820004.png)

如果我们有一个**基准线**作为参开标准，比如人类水平的表现，那么我们可以看到这个模型是高偏差还是高方差的

如果一个学习算法有很高的偏差，那么获得更多的训练数据本身不会有多大帮助。曲线只会保持水平，因为此时模型已经训练到一个其本身能够达到的较好的状态了。继续训练下去，只会保持水平。

![image-20220725103637391](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220725103637391.png)

如果一个算法有很高的方差，那么获得更多的训练数据可能会有所帮助。随着训练样本的增加，模型可以更好地拟合更多的样本，从而J_train上升，J_cv下降，直至他们接近了基准线

![image-20220725105723753](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220725105723753.png)

综上所述，加入我们有1000个样本，你可以用100个样本来训练一个模型，看看他的训练误差和交叉验证误差，然后再用200个样本来训练一个模型，然后画出J_train和J_cv，重复以上操作然后画出大致的学习曲线。即可观察学习曲线是高偏差还是高方差，从而决定下一步的操作。

但其实，使用不同大小子集训练多种不同模型是需要昂贵算力的，再实际中并不常见。

![](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220725114311118.png)

**获得更多样本：**减小方差，样本多了，模型就可以更好地拟合样本，模型变得更好了，平滑了，方差就小了
**减少不必要的特征：**特征没有存在的需求，会使模型变得很复杂，去除了之后，模型会变得简单，图像平滑，方差变小
**增加特征/多项式特征：**模型过于简单，无法拟合更多的样本
**减小λ：**会将注意力放在训练集的拟合上，从而降低了偏差
**增大λ：**因为把太多的注意力妨再训练集的拟合上，会牺牲泛化到新例子上的能力，增加了λ会迫使算法拟合更加平滑的函数。从而降低方差

## 方差、偏差与神经网络的结合

神经网络为我们提供了必须权衡偏差和方差的困境的方法。大型神经网络使用中小型的数据集时是低偏差的机器。

![image-20220725145104111](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220725145104111.png)

拥有一个更大的神经网络并不会有什么坏处，多数情况下，它甚至可以大大提升性能。大型神经网络一般都是低偏差网络。

## 迁移学习

首先在大型数据集上训练，然后在较小的数据集上进一步参数调优，这两个步骤被称为监督预训练。![image-20220728164411675](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220728164411675.png)

具体操作如下：你想要通过神经网络识别数字1-10，所以你先用一个具有很多图片的数据集进行识别训练，训练完了之后，将最后一层的各个神经元的参数重置，重新进行训练。因为通过学习识别猫狗牛人等，它已经为处理输入图像的前几层学习了一些合理的参数，然后通过这些参数迁移到新的网络，新的神经网络的初始值变得更合适了，这样我们只需要再让算法学习一点就可以达到预期的目标。

因此我们可以下载别人已经预先训练好的神经网络，然后再很具自己的数据集进一步训练或者微调网络

## 构建机器学习系统全过程

我们以语音识别为例来说明机器学习项目的整个周期。
![image-20220728171022162](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220728171022162.png)

# 决策树

## 熵定义

![image-20220729111337189](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220729111337189.png)

H(p1)为熵函数，表示样本集合不纯度，熵越小，集合不纯度越低；表示知识的不确定性，熵越小，不确定性越小。

![这里写图片描述](https://imgconvert.csdnimg.cn/aHR0cDovL2ltZy5ibG9nLmNzZG4ubmV0LzIwMTcwODI3MTExMzM2NzAw?x-oss-process=image/format,png)

在决策学习中熵的减少被称为信息增益

![image-20220804094646224](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220804094646224.png)

p1^left表示左子树中带有正标签的例子，在此正标签表示猫
w1^left表示为所有根节点到左边子分支的样本比例
p1^root表示根节点为正的例子的比例

![image-20220804094801877](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220804094801877.png)

信息增益表示进行分裂后熵的变化幅度，结果越大表示熵减的越多，因此我们可以借此来选择一个特性在一个节点上拆分。

![image-20220804101027497](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220804101027497.png)

one-hot-encoding

若一个特征中存在着多种可能，就将其拆开为不同的特征，例如猫耳朵形状，有尖的，圆的，扁的，那么就可以将其分为三个特征进行后续操作。
同时，对于其他特征，比如，是否有胡须，也可以用0/1来表示。![image-20220804102511920](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220804102511920.png)

上述五个特征额可以也投入新的网络或者逻辑回归来尝试训练猫分类器

![image-20220804104855968](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220804104855968.png)连续性特则尝试不同的阈值，根据选定的阈值划分特征，做对应的信息增益的计算，再与其他的特征进行比较，选择最大的信息增益作为分裂的特征。

回归树 给你一个训练集，模型会做的事沿着决策树往下走，直到到达叶节点，然后预测结果。在这里是给你猫狗的相关信息，模型预测他们的体重。![image-20220804110630051](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220804110630051.png)

回归树的节点选择：求得根节点和叶子节点两侧的方差，计算在这种特征分割条件下，方差减小的量，众所周知，方差越小，数字之间的差异度越小，因此我们选择
减小的方差最大点那个特征。

![image-20220804110343461](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220804110343461.png)

## 集成学习

决策树是非常敏感的，一个条件的改变可能就会导致分割节点的变化，因此我们通过多个树进行集成，通过投票从而可以更大几率地确定结果。你的整体算法也就会因此变得更加健壮。

为了建立一个集成模型，我们需要一个技术，叫做有放回抽样。这可以让你构建一个新的训练集。

## 随机森林算法

## XG Boost

给你一个具有B个样本的数据集，实用又放回采样创建一个大小为m的新训练集，然后在新的数据集上训练决策树。

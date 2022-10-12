# **机器学习~2**

## 1、K-Means算法

![image-20220808182546148](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220808182546148.png)

step1：随便选择两个位置作为簇心，计算每一个点到簇心的距离，将每一个点都分配到一个簇
step2：通过公式计算出每个簇的新簇心位置，重复计算每一个点到新簇心的距离，重新定义每一个点的归属
step3：重复上述操作，直至该算法收敛，不在变化。此时就完成了聚类操作。



![image-20220808184654625](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220808184654625.png)如果你有两个特征，那么μ就会有两个数字，如果有n个特征，那就有n个数字，每一个xi都是一个n维向量。
求簇心就是求向量的想同纬度的均值，均值组成的向量即为簇心。

有一种极端情况：随机分配到簇心太偏了，没有点属于这个簇，有两种解决办法：1、直接删除这个簇 2、重新随机找个簇心。不过在运行kmeans时，消除这个簇的做法实际上更为常见。

k-means其实也在优化一个特定的代价函数

![image-20220808190851032](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220808190851032.png)

![image-20220808191244199](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220808191244199.png)

ci表示训练样本xi所在的簇的索引（1到k）
代价函数表示所有样本点xi到其对应的簇心的距离的和的平均值

# 异常检测

最常用的办法是通过一种称为密度估计的技术，当你得到M个样本的训练集时，你要做的第一件事是为X的概率建立一个模型，算法会尝试找出具有高概率的特征X1和X2的值,以及在数据集中不太可能有/遇见概率小的值是什么。![image-20220809110605293](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809110605293.png)

之后我们就会学习如何根据训练集判断哪些区域的概率更高、哪些区域的概率更低。那些概率低的，更大概率存在差错，需要仔细查看。

![image-20220809160514168](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809160514168.png)

左边是μ，右边是方差$σ^2$  当我们只有一个特征时，我们可以计算出均值以及方差。通过正态分布公式就可以计算出，待检验的x的概率p（x），如果p（x）太小，即可能存在差错。

![image-20220809162316805](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809162316805.png)

如果我们有多个特征，比如引擎出现问题包含温度和振动频率两个方面，那么向量X即由两个特征组成，我们会先研究单个特征的p（x），再将其所有结果相乘，即得到飞机引擎出现问题的概率。

![image-20220809164333668](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809164333668.png)

## **如何选择阈值？ 如何检查异常检测系统是否正常运行？**

![image-20220809171219282](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809171219282.png)

已知10000个已知正常引擎的样本和20个已知存在异常的引擎，我们对其进行以下的划分：
训练集：6000个
交叉验证集：2000个正常引擎以及10个异常引擎
测试集：2000个正常引擎以及10个异常引擎
然后我们来看，交叉验证集和测试集中，该模型将多少个正常引擎判断为异常引擎，又有多少个将异常引擎判断为正常引擎，如果这种误判的情况出现的频率很小即表示这个异常检测系统是正常运行的。若存在误判情况，那么我们可以通过那些例子来对阈值进行修改，例如若将好的判断成坏的的情况出现次数多，就适当地增大阈值，坏的判断成好的出现的次数多就减小阈值。

![image-20220809171144881](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809171144881.png)

如果说已知存在异常的引擎个数很少，可以取消交叉验证及，直接通过测试集来进行评测。

## 异常检测和监督学习之间的比较：

当你的已知异常样本个数非常少时，异常检测往往更加合适![image-20220809173127935](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809173127935.png)

如果说你想要预测的东西是存在变化性的，比如诈骗，这个东西是更新迭代的，不是一成不变的，因此适合异常检测，因为差错检验是通过对已知的负样本（y=0 即正常的样本 我也不知道为什么这么拗口 但是是这样的）进行学习，然后来判断，即使你是新的，也可以进行评判。

如果你预测的是一成不变的，比如垃圾邮件，再怎么搞还是那几样，监督学习就是对比这封邮件是否存在垃圾邮件的特征，从而进行有效判断。

## 如何选择一个好的特征？

先训练模型，然后查看算法在交叉验证集中没能检测出的异常,然后查看这些样本，考虑是否有必要新建一个特征，从而使算法能够发现这些异常。异常的样本会在新特征上表现出有反常大或小的值。

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809191059526.png" alt="image-20220809191059526" style="zoom:50%;" />

![image-20220809191437524](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809191437524.png)

选择在异常情况下可能具有异常大值或小值的特性。比如电脑非常不正常，但是CPU负载和网络流量都正常。不正常的是CPU负载大，同时流量非常低。如果你正在传输视频，然后计算机可能存在的情况是高CPU高流量或者低CPU和低流量，所以我们将CPU和流量的比值看作为一个新的特征，这个新特征可以让异常检测算法在未来碰到异常的特定机器时可以正确地判定。

## 推荐系统：

![image-20220809203035138](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809203035138.png)

所有用户合在一起的代价函数：<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220809203115439.png" alt="image-20220809203115439" style="zoom:50%;" />这个是用户j的代价函数，然后我们将所有用户的代价函数相加，即得到上面黄色标记的式子，式中的wj 和 bj对应j用户的参数，xi表示电影i的特征。f（x）是预测的评分，y^（i，j）表示用户j对电影i的真实评分。 wk^j是j用户对电影k的参数。即所有用户对所有电影的参数的和。

![image-20220810155050494](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220810155050494.png)

在已知参数wj和bj的情况下，我们可以通过代价函数对特征的取值进行估计。wj和bj为已知的用户的参数，xi表示第i个电影的特征，上面提到的3个都是一个向量，就比如xi就可能包含两个特征，y^(i,j) 为用户j对电影i的真实评分，xk^i表示电影i的第k个特征为多少。J(x ^i)表示电影i的特征的的代价函数。J(x ^1,x ^2,....x ^n)表示所有电影特征的代价函数和。

![image-20220810161808997](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220810161808997.png)

将两个特征函数合在一起得到上述式子，此时为参数和电影特征的代价函数。我们可以通过梯度下降法对其求解。![image-20220810162556436](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220810162556436.png)

上述算法被称为协同过滤，指的是因为多个用户对同一部电影进行评分，有点协作的样子，给了你这部电影可能样子的感觉，这可以让你猜测这部电影的合适特征，这反过来可以让你预测其他还没有评价这部电影的用户可能在为未来的评价，所以协同过滤就是从多个用户哪里收集数据。

![image-20220812145308390](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220812145308390.png)

以上是可用于**二元标签**协同过滤的代价函数。即0/1

## 均值归一化

均值归一化可以让算法运行的更快一些，它可以给算法更合理的预测，![image-20220812153429922](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220812153429922.png)

当我们遇到一个人对所有电影都未评分时，他的w和b一般会是0，如果直接计算的话得出的结果也都是0 ，这是不合理的，因此我们对每一行进行均值归一化，再通过wx+b计算出预估评分后再加上μ作为最终的评分。

## Tensorflow实现协同过滤算法

#### 使用Tensorflow的GradientTape功能实现：

![image-20220814194454966](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220814194454966.png)

x=tf.variable(3.0) 获取参数w，并将其初始化为3，告诉Tensorflow w是一个变量，就是告诉它w是我们想要优化的参数。

iterations 表示进行三十次梯度下降

tf.GradientTape() as tape ：Tensorflow会自动记录计算代价J所需的步骤顺序

接下来TensorFlow会在执行GradientTape代码时保存操作序列，通过tape.gradient(costJ,[w])计算导数项

#### auto diff实现：

箭头所指的这行代码指明了优化器使用的是Keras. optimizer中的Adam.并在括号内明确了学习率。

```python
optimizer=keras.optimizer.Adam(learning_rate=1e-1)
```

## 基于内容过滤算法 

协同过滤采用的一般方法是，根据其他和你评分相近的用户的评分，向你推荐商品。

基于内容过滤的算法会根据用户和物品的特征，做好匹配后向你推荐物品,相比于协同过滤算法，基于内容过滤可以更好的匹配用户和物品。在基于内容的过滤中，我们将开发一种学习用户和电影匹配的算法。

![image-20220816161537903](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220816161537903.png)
x_u：用户对于电影的喜好向量
x_m：电影的类型分布向量
两个向量通过各自的神经网络得出两个不同的新向量，新的向量会更加精简，有效，新向量进行点积运算得到的结果即为预测结果。
代价函数包含了两个网络中的参数，通过梯度下降或者其他优化算法得到的也是两个网络中的参数。

许多大规模的推荐系统的实现可以分为两个步骤，分别被称为数据检索和排名
在检索步骤期间，这个想法将生成一大堆看似合理的候选项目列表，试图涵盖您向用户推荐的种种可能

















reshape()
squeeze()
unsqueeze()

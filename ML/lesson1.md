# 机器学习~1

## 1-1监督学习：有目的明确的训练方式，你知道得到的是什么

**定义**：对于数据集中的每一个样本（包含输入x和输出标签y），我们想要通过算法预测并得到正确的答案
**回归问题**：我们预测的目标是一个连续的输出值，连续指的是两个点之间可以进行细分 如100 100.1 100.2 101
**分类问题**：预测类别，类别不一定是数字，例如预测图片中是猫还是狗，肿瘤是良性的还是恶性的。也可以预测数字例如1、2、3 与回归问题不同的是，分类问题预测的是一个小的有限的可以产出类别 如0、1、2 并不是所有可能的数字之间 如100 100.1 100.2 101

## 1-2无监督学习：没有明确目的的训练方式，你无法提前知道结果是什么

**定义**：对于数据集中的每一个样本（包含输入x但不包含输出标签y），要求算法自己找到数据中的一些结构或者模式或者有趣的东西
**聚类算法**：在没有标签（比如说你的年龄、体重、身高等）的情况下个获取数据，并尝试将他们自动分组到集群（具有相同的数据类型）中
**异常检测**：被用来探测不寻常的事情
**降维**：将一个大数据集压缩到一个小得多的数据集，同时尽可能小的损失信息

## 2-1模型描述

**线性回归模型**：
满足：1、要是一个回归问题，预测的变量 y 输出集合是无限且连续，我们称之为回归。预测明天的降雨量多少，就是一个回归问题。2、要预测的变量 y 与自变量x的关系是**线性**的：即变量之间保持等比例的关系，从图像上来看，变量之间的形状为直线，斜率为常数。
**分类模型**：预测的变量 y 输出集合是有限的，预测值只能是有限集合内的一个，比如，天气预报预测明天是否下雨，是一个二分类问题。

m：训练样本的数量
x：输入变量的特征
y：输出变量（预测的目标变量）
(x , y)：一个训练样本
(x^i , y^i)：表示特定的训练样本，即第i个样本，i不是指数，只是训练集的一个索引	
y^:y-hat 表示对y的估计或预测

模型函数 f ：
$$
f(x)=wx+b
$$
<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220705112726712.png" alt="image-20220705112726712" style="zoom:25%;" />

​																							单变量线性回归函数

## 2-2代价函数

定义：所有预测值-实际值的差的平方，再相加，求平均值，最后值越小，说明找到的w，b越合适

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220707143944430.png" alt="image-20220707143944430" style="zoom:50%;" />

可视化代价函数：
1、设计模型函数为 f(x)=w\*x+b

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220707153112789.png" alt="image-20220707153112789" style="zoom: 25%;" />

2、根据模型函数绘画出对应的三维代价函数空间图像

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220707153149568.png" alt="image-20220707153149568" style="zoom:25%;" />

3、平面切割 作出对应的等高线图像 图像的椭圆的中心对应的w，b即为最佳值

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220707153038558.png" alt="image-20220707153038558" style="zoom: 33%;" />

## 3-1梯度下降

定义：
1、在存在模型函数f(x)的情况下、设置w、b初始值为零，寻找下降速度最快的路线，到了最低点即得到了局部最优的w、b。
2、更换初始w、b的值重复上一个步骤，从而寻得所有的局部最优点，再进行比较，从而寻找到全局最优解。

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220708104745204.png" alt="image-20220708104745204" style="zoom:25%;" />

## 3-2实现梯度下降

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220708115350671.png" alt="image-20220708115350671" style="zoom:33%;" />

1、α控制了在更新模型参数w、b时迈的步子大小，α的值如果太小，达到最低点需要很长时间。α的值如果太大，会导致第一次就会越过最低点，因为α太大了，导致之后的变换中，w的值左右摇摆，斜率会变得越来越大，即不会再寻找到最低点。

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220708123854775.png" alt="image-20220708123854775" style="zoom:25%;" />

2、偏导控制方向
3、w控制下一个点的位置

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220708124502065.png" alt="image-20220708124502065" style="zoom:33%;" />

正常情况会是这样，下降的速度逐渐减小直至导数为0 即 w=w-α*0 时结束

## 3-3线性回归中的梯度下降



<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220708125712926.png" alt="image-20220708125712926" style="zoom: 25%;" />

将模型函数和代价函数整合进入梯度下降函数得到梯度下降算法：重复操作直至w、b收敛

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220708125830084.png" alt="image-20220708125830084" style="zoom:50%;" />

## 4-1多类特征

![image-20220709101152298](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709101152298.png)

![image-20220709101852854](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709101852854.png)

如果存在影响模型的多个特征，我们会用向量来进行简略表示，通过向量的点积可以实现这一操作：

**点积**：两个向量a = [a1, a2,…, an]和b = [b1, b2,…, bn]的点积定义为：a·b=a1b1+a2b2+……+anbn。
使用矩阵乘法并把（纵列）向量当作n×1 矩阵，点积还可以写为：
$$
a·b=（a^T）*b
$$
这样可以用更少的字符写出更紧凑的模型 上面显示的即为**多元线性回归**

## 4-2向量化

Numpy：Python支持的一个最为常用的线性代数库
![image-20220709103924561](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709103924561.png)

```python
w=np.array([1.0,2.5.-3.3])
b=4
x=np.array([10,20,30]) #我们能够使用x[0] x[1] x[2]来表示向量中的个别项
```

![image-20220709104225886](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709104225886.png)

```python
f=w[0]*x[0]+
  w[1]*x[1]+
  w[2]*x[2]+b
```

显然这样很繁琐 当n的值变得很大 将无法表示 因此：![image-20220709104421779](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709104421779.png)

```python
f=0
for j in range(n):
    f+=w[i]*x[i]
f+=b
```

但还不够简便 我们其实可以通过numpy库 的一个方法直接完成

```python
f=np.dot(w,x)+b
```

![image-20220709110223659](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709110223659.png)

NumPy效率高的原理：传统的计算方法是每次计算一个算式 然后赋值 类似于并行，而numpy是并发的，系统为其分配了足够的资源使其能够同时对多组算式进行运算，然后直接赋值给向量 ，而不是赋值给向量中的一个元素。

## 4-3多元线性回归的梯度下降法

![image-20220709112439676](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709112439676.png)

将代价函数表示出来 并进行求导得出以下模型即为多元线性回归梯度下降算法：![image-20220709134114094](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709134114094.png)

**正规方程Normal Equation**

​	在前面我们学习了使用梯度下降法来计算参数最优解，其过程是对代价函数相对于每个参数求偏导数，通过迭代算法一步一步进行同步更新，直到收敛到全局最小值，从而得到最优参数值。而正规方程则是通过数学方法一次性求得最优解。

​	其主要思想是利用微积分的知识，我们知道对于一个简单的函数，我们可以对于其参数求导，并将其值置为0，这样就可以直接得到参数的值。就像就像下面这样：![img](https://img-blog.csdn.net/20161210141728050?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvc2Q5MTEwMTEw/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

​	但是现在的问题是现实的例子都是很多参数的，我们需要做的就是对于这些参数都求偏导数，从而就得到各个参数的最优解，也就是全局最优解，但是困难在于，如果按照上面这么做将会非常费时间，所以有更好的办法。

但其实随着NumPy的发展，现在也可以直接借用库函数进行运算，也不需要迭代了。并且在我们的日常开发过程中，更多人会选择NumPy。

## 5-1特征缩放

当我们有不同的特征，取值范围差异很大，这可能会导致梯度下降非常缓慢，但重新缩放不同的特征，使他们具有可比较的取值范围，这样速度即梯度的变化会更加有效。

![image-20220709143019665](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709143019665.png)

​	特征缩放法1：
![image-20220709152548866](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709152548866.png)

法2：均值归一化

![image-20220709152710334](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709152710334.png)

μi表示均值即 xi相加求平均，分母为区间端点相减

法三：Z-Score标准化
首先要计算出该特征的均值μi以及标准差σi

![image-20220709153625186](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709153625186.png)

我们在进行特征缩放时，最后将特征的取值范围定在-1~1附近

## 5-2检查梯度下降是否收敛

法1、先画个学习曲线图，看看你需要在迭代多少次后在开始训练模型
法2、通过自动收敛测试来决定什么时候来完成模型训练：设置ε表示一个很小的值例如10^-3 当某一次迭代后的代价函数 J小于10^-3 即停止迭代。但我们经常发现，选出正确的ε是困难的，因此我们更倾向于看学习曲线图。

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709155823572.png" alt="image-20220709155823572" style="zoom:33%;" />

## 5-3学习率的选择

![image-20220709160825491](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709160825491.png)

学习率α太大导致反复横跳，太小迭代的次数会很多。因此，我们通常会先设置α的初始值为一个较小的数字，比如0.001，并且绘画横轴为迭代次数，纵轴为代价的图像，寻找下降速率合适的作为学习率。小技巧：如果J一直在增大，那么我们可以将α设置为一个极小的值，若J仍然保持增大，表示你的代码存在BUG

## 5-4特征工程

即选择合适的特征x来构建模型，如预测房价，x1表示长，x2表示宽，这样就很麻烦，不如令x3=x1*x2表示面积，这样就会方便不少。

**多项式回归算法**：帮助你将数据拟合成曲线，非线性函数

## 6-1逻辑回归

![image-20220709174501480](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709174501480.png)

如果通过线性回归来建立模型，我们可以选择一个阈值，比如0.5，如果模型输出值>0.5那么就可以得出yes（1）反之得出no（0），通过水平线与模型函数相交于一点，然后过这点作垂线，在他左边的即为no，右边即为yes（1），但若在最右边添加一个数据，模型函数发生改变，改变之后的垂线左边不都是no了，说明用线性回归来解决分类问题并不适合。所以接下来会介绍，逻辑回归算法，以处理分类问题。

![image-20220709182850479](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220709182850479.png)此处的g（z)即为逻辑回归函数 f=g

## 6-2决策边界

<img src="C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711112602720.png" alt="image-20220711112602720" style="zoom:33%;" />

逻辑回归模型函数：表示在给定输入特征x和参数w、b的情况下y=1的概率为多少

![image-20220711113550156](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711113550156.png)

我们通常认为f>0.5=>y=1，f<0.5=>y=0 ，一层层推导下去即可得 w\*x+b>0=>y=1 	w\*x+b<0=>y=0

![image-20220711114612153](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711114612153.png)

如图所示：在线上方z>0即表示y=1 下方z<0 即表示 y=0，在这个例子中w1=1,w2=1,b=-3，当这些参数不同时，对应决策边界的线也会不同

## 6-3逻辑回归的代价函数	![image-20220711134621604](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711134621604.png)

如果我们直接套用线性回归的代价函数，如图所示，会产生具有多个极小值点的图像，这是不利于我们寻找最小值的。因此我们引入了误差函数L![image-20220711134843844](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711134843844.png)

当样本值为1时，因为f的值时0~1的，所以取上面一段图像，如图片左边所示。当我们的预测值越小，误差越大。预测值越接近1，误差越小。

![image-20220711134921893](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711134921893.png)

同理可得样本为0的相关图像 ：![image-20220711135204048](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711135204048.png)

简便写法：![image-20220711135507902](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711135507902.png) 
即代价函数：![image-20220711135658230](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711135658230.png)

将代价函数带入梯度下降函数得到：这个求导过程不需要掌握

![image-20220711142157360](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711142157360.png)

即得到逻辑回归的梯度下降：![image-20220711142307547](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711142307547.png)

## 7-1过拟合问题

![image-20220711144639111](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711144639111.png)

分别为欠拟合 正常 过拟合。欠拟合指模型连样本都无法很好满足，过拟合指样本数据可以很好满足，但是新的样本就很大几率不能满足。

解决方法：
1、收集更多的训练数据（首选）
2、观察是否可以选用更少的特征，即过滤掉用处不那么大的特征，选择至关重要的
3、利用正则化较小w1，w2.......wn参数的大小，b几乎是没有必要改变：![image-20220711154522597](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711154522597.png)

我们可知，当wi越小时，模型函数的图像就会更加贴近于样本数据点。因此，我们通过在代价函数后面加上λ/2m*Σwj j从1到n 来寻找最适合的wj值，此处还需要注意λ（正则化参数）的选取，λ如果太大，那么wj就会太小了相当于说是f=b了，这直接就欠拟合了。

![image-20220711155622125](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711155622125.png)

正则线性回归的梯度下降算法：因为仅仅对wj求导，那么与其无关的直接将当作常数了![image-20220711160954231](C:\Users\haipiw\AppData\Roaming\Typora\typora-user-images\image-20220711160954231.png)

同理正则逻辑回归梯度下降算法如上图所示

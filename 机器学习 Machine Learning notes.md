<h1 id="XHKxW">Chap1 引言</h1>
> 机器学习算法最常用的两种就是**监督学习**（supervised learning）和**非监督学习**（unsupervised learning），其次还有**强化学习**（reinforcement learning）和**推荐系统**（recommender system）。
>

<h2 id="DGrPC">监督学习</h2>
监督学习是指给定一个数据集，其中包含了正确的**答案**。比如房价预测，给定一些样本数据，监督学习需要根据这些样本算出更多的结果。用更专业的术语来定义，它也被称为**回归问题**（regression problem）。<font style="color:rgb(31, 35, 40);">回归这个词的意思是，我们在试着推测出这一系列</font>**<font style="color:rgb(31, 35, 40);">连续</font>**<font style="color:rgb(31, 35, 40);">值。</font>

![图1.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743859766976-6d8ab110-02a1-4d53-b871-c07444900fc0.png)



<font style="color:rgb(31, 35, 40);">我们再来看一组数据，在这个数据集中，横轴表示肿瘤的大小，纵轴上标出1和0表示是否为恶性肿瘤。我们之前见过的肿瘤，如果是恶性则记为1，不是恶性，或者说良性则记为0。</font>

![图1.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984780219-ecc504c9-3edc-4118-a69d-d8dc5c0ebcbd.png)

<font style="color:rgb(31, 35, 40);">机器学习可以尝试去估算出某个大小的肿瘤是恶性的或是良性的概率。用术语来讲，这是一个</font>**<font style="color:rgb(31, 35, 40);">分类问题</font>**<font style="color:rgb(31, 35, 40);">（classification problem）。</font>

<font style="color:rgb(31, 35, 40);">分类指的是，我们试着推测出</font>**<font style="color:rgb(31, 35, 40);">离散</font>**<font style="color:rgb(31, 35, 40);">的输出值：0或1良性或恶性，而事实上在分类问题中，输出可能不止两个值。比如说可能有三种乳腺癌，所以你希望预测离散输出0、1、2、3。0 代表良性，1 表示第1类乳腺癌，2表示第2类癌症，3表示第3类，但这也是分类问题。</font>

<font style="color:rgb(31, 35, 40);">如果有多个离散值的话，也可以尝试用不同的符号来表示这些数据。即图1.2下方的横轴所展示的，用不同的符号来表示良性和恶性肿瘤，良性的肿瘤改成用 </font>**<font style="color:rgb(31, 35, 40);">O</font>**<font style="color:rgb(31, 35, 40);"> 表示，恶性的继续用 </font>**<font style="color:rgb(31, 35, 40);">X</font>**<font style="color:rgb(31, 35, 40);"> 表示。来预测肿瘤的恶性与否。</font>

<font style="color:rgb(31, 35, 40);">在其它一些机器学习问题中，可能会遇到不止一种特征。举个例子，我们不仅知道肿瘤的尺寸，还知道对应患者的年龄。在其他机器学习问题中，我们通常有更多的特征。</font>

![图1.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984794078-59f743f0-859c-467c-8b90-d575daa50c22.png)

<font style="color:rgb(31, 35, 40);">上图中列举了总共5种不同的特征，坐标轴上的2种和右边的3种，但是在一些学习问题中，你希望不只用3种或5种特征。相反，你想用</font>**<font style="color:rgb(31, 35, 40);">无限多种特征</font>**<font style="color:rgb(31, 35, 40);">，好让你的算法可以利用大量的特征，或者说线索来做推测。那该怎么处理无限多个特征？以及怎么存储这些特征？电脑的内存肯定不够用。我们以后会讲一个算法，叫</font>**<font style="color:rgb(31, 35, 40);">支持向量机</font>**<font style="color:rgb(31, 35, 40);">（support vector machine），里面有一个巧妙的数学技巧，能让计算机处理无限多个特征。想象一下，我没有写下这两种和右边的三种特征，而是在一个无限长的列表里面，一直写一直写不停的写，写下无限多个特征，事实上，我们能用算法来处理它们。</font>

---

<h2 id="u2FgW">无监督学习</h2>
<font style="color:rgb(31, 35, 40);">在无监督学习中，我们已知的数据</font>**<font style="color:rgb(31, 35, 40);">没有任何的标签</font>**<font style="color:rgb(31, 35, 40);">用于区分不同的数据。所以我们已知数据集，却不知如何处理，也未告知每个数据点是什么。别的都不知道，就是一个数据集。你能从数据中找到某种结构吗？对于下图的情况，无监督学习就能判断出数据有两个不同的</font>**<font style="color:rgb(31, 35, 40);">聚集簇</font>**<font style="color:rgb(31, 35, 40);">（cluster）。无监督学习算法可能会把这些数据分成两个不同的簇。所以叫做</font>**<font style="color:rgb(31, 35, 40);">聚类算法</font>**<font style="color:rgb(31, 35, 40);">。事实证明，它能被用在很多地方。</font>



![图1.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984809883-d013b4a5-125b-44bd-923a-7528e4fd2026.png)

我们可以发现，无监督学习相比监督学习的一大特点就是，其没有针对数据集给出明确的答案。

---

<h1 id="CrIeq">Chap2 <font style="color:rgb(31, 35, 40);">单变量线性回归 (Linear Regression with One Variable)</font></h1>
<h2 id="PPNYW">模型表示</h2>
![图2.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984883882-21298878-1c7b-491c-bef5-bcdc924e0d0f.png)

我们在前面提到了，在监督学习中会给定<font style="color:rgb(31, 35, 40);">一个数据集，并且为每一个样本给定一个“正确答案”，这个数据集被称</font>**<font style="color:rgb(31, 35, 40);">训练集</font>**<font style="color:rgb(31, 35, 40);">。</font>

> <font style="color:rgb(31, 35, 40);">我们将在整个课程中用小写的</font>$ m $<font style="color:rgb(31, 35, 40);">来表示训练样本的数目；用</font>$ x $<font style="color:rgb(31, 35, 40);">来表示输入的变量/特征；用</font>$ y $<font style="color:rgb(31, 35, 40);">来表示输出的变量/目标结果。</font>
>

![图2.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743926700158-55da24ec-58f4-46e1-97e1-76d1dde25a3c.png)

就比如上图，我们还可以用$ (x,y) $来表示一个样本；$ (x^{(i)},y^{(i)}) $<font style="color:rgb(31, 35, 40);"> 代表第</font>$ i $<font style="color:rgb(31, 35, 40);">个样本。</font>

<font style="color:rgb(31, 35, 40);">关于一个监督学习算法的工作方式，我们可以在图2.3看到，我们把</font>**<font style="color:rgb(31, 35, 40);">训练集</font>**<font style="color:rgb(31, 35, 40);">里的房屋价格，喂给我们的</font>**<font style="color:rgb(31, 35, 40);">学习算法</font>**<font style="color:rgb(31, 35, 40);">，学习算法工作后输出一个函数，通常表示为小写</font>$ h $<font style="color:rgb(31, 35, 40);">表示。</font>$ h $<font style="color:rgb(31, 35, 40);">代表</font>**<font style="color:rgb(31, 35, 40);">hypothesis</font>**<font style="color:rgb(31, 35, 40);">(</font>**<font style="color:rgb(31, 35, 40);">假设</font>**<font style="color:rgb(31, 35, 40);">)，</font>$ h $<font style="color:rgb(31, 35, 40);">表示一个函数，输入是房屋尺寸大小，就像你朋友想出售的房屋，因此 h 根据输入的 x值来得出 y 值，</font>$ y $<font style="color:rgb(31, 35, 40);">值对应房子的价格 因此，</font>$ h $<font style="color:rgb(31, 35, 40);">是一个从</font>$ x $<font style="color:rgb(31, 35, 40);">到 </font>$ y $<font style="color:rgb(31, 35, 40);">的函数映射。</font>

![图2.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984916257-7e4b359d-e12a-45cb-bebe-ef9cc6bd5789.png)

<font style="color:rgb(31, 35, 40);">那么，对于我们的房价预测问题，我们该如何表达 h？</font>

<font style="color:rgb(31, 35, 40);">一种可能的表达方式为：</font>$ h_\theta \left( x \right)=\theta_{0} + \theta_{1}x $<font style="color:rgb(31, 35, 40);">，因为只含有一个特征/输入变量，因此这样的问题叫作</font>**<font style="color:rgb(31, 35, 40);">单变量线性回归问题</font>**<font style="color:rgb(31, 35, 40);">。</font>

---

<h2 id="Aeyt2">代价函数</h2>
![图2.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743927753885-8fd251b1-c541-45bb-a0b5-f546c91ea1b5.png)

前面提到的$ h $的表达可以如上，其实就是一个线性回归的最小化误差。<font style="color:rgb(31, 35, 40);">我们的目标便是选择出可以使得建模误差的平方和能够最小的模型参数。 即使得</font>**<font style="color:rgb(31, 35, 40);">代价函数</font>**<font style="color:rgb(31, 35, 40);"> </font>$ J(\theta_0,\theta_1)=\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})−y^{(i)})^2 $<font style="color:rgb(31, 35, 40);">最小。这里分子有个</font>$ 2 $<font style="color:rgb(31, 35, 40);">是为了后面求导形式可以把平方的</font>$ 2 $<font style="color:rgb(31, 35, 40);">约掉，让式子简单化。</font>

<font style="color:rgb(31, 35, 40);">代价函数也被称作</font>**<font style="color:rgb(31, 35, 40);">平方误差函数</font>**<font style="color:rgb(31, 35, 40);">，有时也被称为平方误差代价函数。我们之所以要求出误差的平方和，是因为误差平方代价函数，对于大多数问题，特别是回归问题，都是一个合理的选择。还有其他的代价函数也能很好地发挥作用，但是平方误差代价函数可能是解决回归问题</font><u><font style="color:rgb(31, 35, 40);">最常用的手段了</font></u><font style="color:rgb(31, 35, 40);">。</font>

如图2.5，如果我们假设$ \theta_0 = 0 $，那么根据$ \theta_1 $的不同取值，我们可以做出如下的图：

![图2.5](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984931275-0fd3fa2e-d134-4e4f-87d7-af410d118f87.png)

可以看到对于给定的三个样本，我们的$ J $函数在$ \theta_1 = 1 $时取到最小值$ 0 $。

刚刚我们假设$ \theta_0 = 0 $，然而实际情况中，根据两个参数的取值不同，我们的代价函数$ J $可能的取值如下：

![图2.6](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984938999-453a90a1-17e2-4637-9feb-b05f549ee01a.png)



其实也有其他的作图方式可以替代图2.6的三维图，就是**等高线图**（contour）：

![图2.7](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984703360-d7fd873f-de3f-4a9c-8788-535296318a8b.png)

图2.7右侧的图就是等高线图，两个坐标轴分别表示两个变量的取值，每一个椭圆形展示了一系列$ J $值相等的点。等高线图就像地形图一样。

---

<h2 id="EVrzq">梯度下降</h2>
**<font style="color:rgb(31, 35, 40);">梯度下降</font>**<font style="color:rgb(31, 35, 40);">（gradient descent）是一个用来求函数最小值的算法，梯度下降是很常用的算法，不只是用于线性回归领域，还应用于机器学习的很多其他领域。接下来介绍使用梯度下降算法来求出代价函数</font>$ J(\theta_{0}, \theta_{1}) $<font style="color:rgb(31, 35, 40);">的最小值。</font>

<font style="color:rgb(31, 35, 40);">梯度下降背后的思想是：开始时我们随机选择一个参数的组合</font>$ \left( {\theta_{0}},{\theta_{1}},......,{\theta_{n}} \right) $<font style="color:rgb(31, 35, 40);">，计算代价函数，然后我们寻找下一个能让代价函数值下降最多的参数组合。我们持续这么做直到找到一个局部最小值（</font>**<font style="color:rgb(31, 35, 40);">local minimum</font>**<font style="color:rgb(31, 35, 40);">），因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是全局最小值（</font>**<font style="color:rgb(31, 35, 40);">global minimum</font>**<font style="color:rgb(31, 35, 40);">），选择不同的初始参数组合，可能会找到</font>**<font style="color:rgb(31, 35, 40);">不同的局部最小值</font>**<font style="color:rgb(31, 35, 40);">。</font>

![图2.8](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984678901-00aaee44-79f6-480b-8db6-540ac7d2a4c6.png)

<font style="color:rgb(31, 35, 40);">批量梯度下降（</font>**<font style="color:rgb(31, 35, 40);">batch gradient descent</font>**<font style="color:rgb(31, 35, 40);">）算法的公式为：</font>

![图2.9](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743984953455-b1f8604c-6750-499a-b1e2-284757e7b0df.png)

<font style="color:rgb(31, 35, 40);">其中</font>$ \alpha $<font style="color:rgb(31, 35, 40);">是</font>**<font style="color:rgb(31, 35, 40);">学习率</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">learning rate</font>**<font style="color:rgb(31, 35, 40);">），它决定了我们沿着能让代价函数下降程度最大的方向向下迈出的</font>**<font style="color:rgb(31, 35, 40);">步子有多大</font>**<font style="color:rgb(31, 35, 40);">，在批量梯度下降中，我们每一次都同时让所有的参数减去学习速率乘以代价函数的导数。</font>

![图2.10](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743930093331-aaf2faa8-0f8f-41bc-ba28-b46c31f6a74e.png)

<font style="color:rgb(31, 35, 40);">接下来我们讲解一些微分项</font>$ \alpha \frac{\partial }{\partial {{\theta }{0}}}J({{\theta }{0}},{{\theta }{1}}) $<font style="color:rgb(31, 35, 40);">的具体作用。</font>

如下图，假设我们的代价函数是是一元的，那么根据偏导数（也就是梯度下降**最快**的方向）进行迭代，最终可以得到一个局部最小值。

![图2.11](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743930666466-d82451b2-dc9d-46d8-9651-37f7671f14e5.png)

<font style="color:rgb(31, 35, 40);">另外，如果</font>$ \alpha $<font style="color:rgb(31, 35, 40);">太小了，即学习速率太小，结果就是只能一点点地挪动，去努力接近最低点，这样就需要</font>**<font style="color:rgb(31, 35, 40);">很多步才能到达最低点</font>**<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">如果</font>$ \alpha $<font style="color:rgb(31, 35, 40);">太大，那么梯度下降法可能会越过最低点，甚至</font>**<font style="color:rgb(31, 35, 40);">可能无法收敛</font>**<font style="color:rgb(31, 35, 40);">，下一次迭代又移动了一大步，越过一次，又越过一次，一次次越过最低点，直到你发现实际上离最低点越来越远。</font>

![图2.12](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743930929176-cda6181f-725e-487b-bcac-ea543ece8e7d.png)

---

<h2 id="a2jZK">线性回归中的梯度下降</h2>
我们将之前线性回归的代价函数展开求偏导：

$ \frac{\partial }{\partial{{\theta }_{j}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{\partial }{\partial {{\theta }_{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}^{2}}=\frac{\partial }{\partial {{\theta }_{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( \theta_0 + \theta_1x^{(i)} - {{y}^{(i)}} \right)}}^{2}} $

<font style="color:rgb(31, 35, 40);">j=0 时：</font>$ \frac{\partial }{\partial {{\theta }_{0}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}} $

<font style="color:rgb(31, 35, 40);">j=1 时：</font>$ \frac{\partial }{\partial {{\theta }_{1}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}_j^{(i)}} \right)} $

接下来我们就可以利用图2.9的算法一直迭代，直到收敛了。我们刚刚提到了，梯度下降算法可能会给出一个局部最小值，但是在线性回归中，它总是会给出**全局最小值**。这是因为线性回归的代价函数是一个**凸函数**（convex function），如图2.6。

<font style="color:rgb(31, 35, 40);">我们刚刚使用的算法，有时也称为批量梯度下降。实际上，在机器学习中，通常不太会给算法起名字，但这个名字”</font>**<font style="color:rgb(31, 35, 40);">批量梯度下降</font>**<font style="color:rgb(31, 35, 40);">”，指的是在梯度下降的每一步中，我们都用到了</font>**<font style="color:rgb(31, 35, 40);">所有的训练样本</font>**<font style="color:rgb(31, 35, 40);">。而事实上，有时也有其他类型的梯度下降法，不是这种"批量"型的，不考虑整个的训练集，而是每次只关注训练集中的一些小的子集。在后面的课程中，我们也将介绍这些方法。</font>

---

<h1 id="ck4ap">Chap3 线性代数回顾</h1>
> 略过....
>

---

<h1 id="Oeffh">Chap4 <font style="color:rgb(31, 35, 40);">多变量线性回归 (Linear Regression with Multiple Variables)</font></h1>
<h2 id="vNOyD">多维特征</h2>
我们之前讨论了单变量的回归模型，接下来我们讨论一下多变量回归模型，<font style="color:rgb(31, 35, 40);">例如房间数楼层等，构成一个含有多个变量的模型，模型中的特征为</font>$ \left( {x_{1}},{x_{2}},...,{x_{n}} \right) $<font style="color:rgb(31, 35, 40);">。</font>

![图4.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743945512216-0afe4817-fb27-4103-a8d0-aa04cc763503.png)

<font style="color:rgb(31, 35, 40);">支持多变量的假设 h 表示为：</font>$ h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}} $<font style="color:rgb(31, 35, 40);">，</font>这个公式中有$ n+1 $个参数和$ n $个变量，为了使得公式能够简化一些，引入$ x_{0}=1 $，则公式转化为：$ h_{\theta} \left( x \right)={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}} $

<font style="color:rgb(31, 35, 40);">此时模型中的参数是一个</font>$ n+1 $<font style="color:rgb(31, 35, 40);">维的向量，任何一个训练实例也都是</font>$ n+1 $<font style="color:rgb(31, 35, 40);">维的向量，特征矩阵</font>$ X $<font style="color:rgb(31, 35, 40);">的维度是</font>$ m(n+1) $<font style="color:rgb(31, 35, 40);">，因此公式可以简化为：</font>$ h_{\theta} \left( x \right)={\theta^{T}}X $<font style="color:rgb(31, 35, 40);">，其中上标</font>$ T $<font style="color:rgb(31, 35, 40);">代表矩阵转置。</font>

---

<h2 id="etDC2">多变量梯度下降</h2>
<font style="color:rgb(31, 35, 40);">与单变量线性回归类似，在多变量线性回归中，我们也构建一个代价函数，这个代价函数是所有建模误差的平方和，即：</font>$ J\left( {\theta_{0}},{\theta_{1}}...{\theta_{n}} \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( h_{\theta} \left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}} $<font style="color:rgb(31, 35, 40);"> ，</font>

<font style="color:rgb(31, 35, 40);">其中：</font>$ h_{\theta}\left( x \right)=\theta^{T}X={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}} $<font style="color:rgb(31, 35, 40);">，</font>

<font style="color:rgb(31, 35, 40);">我们的目标和单变量线性回归问题中一样，是要找出使得代价函数最小的一系列参数。 多变量线性回归的批量梯度下降算法为：</font>

![图4.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743946183863-c43fc73f-272a-4c0a-9429-a870cb15d76f.png)

我们可以令$ x_0 = 1 $，那么可以将各$ \theta $同一用一个式子表示。

即：<font style="color:rgb(31, 35, 40);">当</font>$ n\geq1 $<font style="color:rgb(31, 35, 40);">时， </font>$ {{\theta }_{0}}:={{\theta }_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{0}^{(i)} $

<font style="color:rgb(31, 35, 40);">计算代价函数 </font>$ J\left( {\theta_{0}},{\theta_{1}}...{\theta_{n}} \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( h_{\theta} \left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}} $<font style="color:rgb(31, 35, 40);">， 其中：</font>$ {h_{\theta}}\left( x \right)={\theta^{T}}X={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}} $

<font style="color:rgb(31, 35, 40);">计算代价函数的 Python 代码：</font>

```python
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X)) # len(X)就是样本容量
```

---

<h2 id="nnneJ"><font style="color:rgb(31, 35, 40);">梯度下降法实践1-特征缩放</font></h2>
<font style="color:rgb(31, 35, 40);">在我们面对多维特征问题的时候，我们要保证这些特征都具有</font>**<font style="color:rgb(31, 35, 40);">相近的尺度</font>**<font style="color:rgb(31, 35, 40);">，这将帮助梯度下降算法更快地收敛。</font>

<font style="color:rgb(31, 35, 40);">以房价问题为例，假设我们使用两个特征，房屋的尺寸和房间的数量，尺寸的值为 0-2000平方英尺，而房间数量的值则是0-5，以两个参数分别为横纵坐标，绘制代价函数的等高线图能，看出图像会显得很扁（图3.3左），梯度下降算法需要非常多次的迭代才能收敛。</font>

<font style="color:rgb(31, 35, 40);">解决的方法是尝试将所有特征的尺度都尽量</font>**<font style="color:rgb(31, 35, 40);">缩放</font>**<font style="color:rgb(31, 35, 40);">到-1到1之间（图3.3右）。</font>

![图4.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743985210818-ef10b313-53f7-4d26-a304-1c65182a4fde.png)

<font style="color:rgb(31, 35, 40);">最简单的方法是令：</font>$ {{x}_{n}}=\frac{{{x}_{n}}-{{\mu}_{n}}}{{{s}_{n}}} $<font style="color:rgb(31, 35, 40);">，其中 </font>$ \mu_n $<font style="color:rgb(31, 35, 40);">是平均值，</font>$ {s_{n}} $<font style="color:rgb(31, 35, 40);">是标准差，这个过程叫做</font>**<font style="color:rgb(31, 35, 40);">均值归一化</font>**<font style="color:rgb(31, 35, 40);">（mean normalization），这里用样本</font>**<font style="color:rgb(31, 35, 40);">极差</font>**<font style="color:rgb(31, 35, 40);">取代标准差也没有关系。</font>

![图4.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743986235074-4a606c2c-932e-4f32-ad44-dbde82a1e967.png)

---

<h2 id="e05RI"><font style="color:rgb(31, 35, 40);">梯度下降法实践2-学习率</font></h2>
<font style="color:rgb(31, 35, 40);">梯度下降算法收敛所需要的迭代次数根据模型的不同而不同，我们无法提前预知，我们可以绘制迭代次数和代价函数的图表来观测算法在何时趋于收敛。</font>

![图4.5](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743991447773-89da9ca6-cc42-4f7e-ba1c-fa2378620c0e.png)

<font style="color:rgb(31, 35, 40);">梯度下降算法的每次迭代受到学习率的影响，如果学习率</font>$ \alpha $<font style="color:rgb(31, 35, 40);">过小，则达到收敛所需的迭代次数会非常高；如果学习率</font>$ \alpha $<font style="color:rgb(31, 35, 40);">过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛。</font>

<font style="color:rgb(31, 35, 40);">数学上已经证明过，当学习率</font>$ \alpha $**<font style="color:rgb(31, 35, 40);">足够小</font>**<font style="color:rgb(31, 35, 40);">时，代价函数</font>$ J $**<font style="color:rgb(31, 35, 40);">总是趋于收敛</font>**<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">通常可以考虑尝试这些学习率：</font>$ \alpha=0.01，0.03，0.1，0.3，1，3，10 $

---

<h2 id="CLuW6"><font style="color:rgb(31, 35, 40);">特征和多项式回归</font></h2>
还是以房价问题举例，以房子的长和宽作为特征，预测房价$ h_\theta(x)=\theta_0+\theta_1×frontage+\theta_2×depth $

![图4.6](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743992286518-e773a129-f569-4d0e-88fa-0b246250f5f5.png)

线性回归并不适用于所有数据，有时我们需要**曲线**来适应我们的数据，比如一个二次方模型：$ h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2} $ 或者三次方模型： $ h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2} + \theta_3x_3^{3} $

![图4.7](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743992450923-b2d11b3c-3792-43e4-b095-04a879ff1ee8.png)

<font style="color:rgb(31, 35, 40);">通常我们需要先观察数据然后再决定准备尝试怎样的模型。 另外，我们可以令：</font>$ {{x}_{2}}=x^{2},{{x}_{3}}=x^{3} $<font style="color:rgb(31, 35, 40);">，从而将模型转化为线性回归模型。</font>

<font style="color:rgb(31, 35, 40);">根据函数图形特性，我们还可以使：</font>$ {{{h}}_{\theta}}(x)={{\theta }_{0}}\text{+}{{\theta }_{1}}(size)+{{\theta}_{2}}{{(size)}^{2}} $

<font style="color:rgb(31, 35, 40);">或者:</font>$ {{{h}}_{\theta}}(x)={{\theta }_{0}}\text{+}{{\theta }_{1}}(size)+{{\theta }_{2}}\sqrt{size} $

<font style="color:rgb(31, 35, 40);">注：如果我们采用多项式回归模型，在运行梯度下降算法前，特征缩放非常有必要。</font>

---

<h2 id="DJBtc">正规方程</h2>
<font style="color:rgb(31, 35, 40);">到目前为止，我们都在使用梯度下降算法，但是对于某些线性回归问题，</font>**<font style="color:rgb(31, 35, 40);">正规方程</font>**<font style="color:rgb(31, 35, 40);">方法是更好的解决方案。</font>

<font style="color:rgb(31, 35, 40);">正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：</font>$ \frac{\partial}{\partial{\theta_{j}}}J\left( {\theta_{j}} \right)=0 $<font style="color:rgb(31, 35, 40);"> 。 假设我们的训练集特征矩阵为</font>$ X $<font style="color:rgb(31, 35, 40);">（包含了</font>$ x_0=1 $<font style="color:rgb(31, 35, 40);">）并且我们的训练集结果为向量 </font>$ y $<font style="color:rgb(31, 35, 40);">，则利用正规方程解出向量 </font>$ \theta=(X^TX)^{−1}X^Ty $<font style="color:rgb(31, 35, 40);"> （这里其实就是线代里面的</font>**<font style="color:rgb(31, 35, 40);">投影矩阵</font>**<font style="color:rgb(31, 35, 40);">的应用）。以下表示数据为例：</font>

![图4.8](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743993218344-aeb098e8-6eb6-4746-b640-f72759255b04.png)

这里需要注意的是，矩阵$ X^TX $可逆当且仅当$ X $可逆（详见线代笔记Chap14）。因此我们必须保证$ X $中**没有重复的样本**，如果我们可以保证各样本都是不同的话，由于$ x_0 = 1 $，所以$ X $各行一定线性无关，即$ X $一定可逆。

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1743995002714-f9a19493-8a8b-4396-ab4c-0eef3d59a86f.png)

<font style="color:rgb(31, 35, 40);">注：对于那些不可逆的矩阵（通常是因为特征之间不独立，如同时包含英尺为单位的尺寸和米为单位的尺寸两个特征，也有可能是特征数量大于训练集的数量），正规方程方法是不能用的。</font>

<font style="color:rgb(31, 35, 40);">总结一下，只要特征变量的数目并不大，正规方程是一个很好的计算参数</font>$ \theta $<font style="color:rgb(31, 35, 40);">的替代方法。</font>

<font style="color:rgb(31, 35, 40);">随着我们要讲的学习算法越来越复杂，例如，当我们讲到分类算法、逻辑回归算法，我们会看到，实际上对于那些算法，并不能使用标准方程法。对于那些更复杂的学习算法，我们将不得不仍然使用</font>**<font style="color:rgb(31, 35, 40);">梯度下降法</font>**<font style="color:rgb(31, 35, 40);">。因此，梯度下降法是一个非常有用的算法，可以用在有大量特征变量的线性回归问题。所以，根据具体的问题，以及你的特征变量的数量，这两种算法都是值得学习的。</font>

<font style="color:rgb(31, 35, 40);">正规方程的python实现：</font>

```python
import numpy as np
    
 def normalEqn(X, y):
   theta = np.linalg.inv(X.T@X)@X.T@y #X.T@X等价于X.T.dot(X)
   return theta
```

---

<h2 id="zP5nl">正规方程在矩阵不可逆情况下的解决方法</h2>
可以使用**伪逆**。



---

<h1 id="cf0Bq">Chap5 Ocatve教程</h1>
> 略过……
>



---

<h1 id="eTpH3">Chap6 逻辑回归（Logistic Regression）</h1>
> <font style="color:rgb(31, 35, 40);">在分类问题中，要预测的变量 y 是</font>**<font style="color:rgb(31, 35, 40);">离散</font>**<font style="color:rgb(31, 35, 40);">的值，我们将学习一种叫做逻辑回归 (</font>**<font style="color:rgb(31, 35, 40);">Logistic Regression</font>**<font style="color:rgb(31, 35, 40);">) 的算法，这是目前最流行使用最广泛的一种学习算法。</font>
>
> <font style="color:rgb(31, 35, 40);">在分类问题中，我们尝试预测的是结果是否属于某一个类（例如正确或错误）。分类问题的例子有：判断一封电子邮件是否是垃圾邮件；判断一次金融交易是否是欺诈；之前我们也谈到了肿瘤分类问题的例子，区别一个肿瘤是恶性的还是良性的。</font>
>

<h2 id="ZkquR">分类</h2>
<font style="color:rgb(31, 35, 40);">我们从二元的分类问题开始讨论。</font>

<font style="color:rgb(31, 35, 40);">我们将因变量(</font>**<font style="color:rgb(31, 35, 40);">dependent variable</font>**<font style="color:rgb(31, 35, 40);">)可能属于的两个类分别称为负向类（</font>**<font style="color:rgb(31, 35, 40);">negative class</font>**<font style="color:rgb(31, 35, 40);">）和正向类（</font>**<font style="color:rgb(31, 35, 40);">positive class</font>**<font style="color:rgb(31, 35, 40);">），则因变量</font>$ y\in \{ 0, 1 \} $<font style="color:rgb(31, 35, 40);"> ，其中 0 表示负向类，1 表示正向类。</font>

![图6.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745202061304-156e4943-8d9e-481b-bac5-07487d0e7ecb.png)

如图6.1，对于之前提到的肿瘤分类问题，如果使用线性回归来解决，对于原始的数据我们或许会得到一条拟合直线（粉红色），对于某个**阈值**（threshold）而言，按照样本与阈值的大小关系来进行分类。但是当我们增加一个新的样本（最右边的那个）后，拟合直线会发生变化，这时候阈值也随之变化，对于某些结果的预测会变得不准确。

<font style="color:rgb(31, 35, 40);">此外，如果我们要用线性回归算法来解决一个分类问题，对于分类 y 取值为 0 或者1，但如果你使用的是线性回归，那么假设函数的输出值可能远大于 1，或者远小于0，尽管我们知道标签应该取值0或者1，但是如果算法得到的值远大于1或者远小于0的话，就会感觉很奇怪。所以我们在接下来的要研究的算法就叫做</font>**<font style="color:rgb(31, 35, 40);">逻辑回归</font>**<font style="color:rgb(31, 35, 40);">算法，这个算法的性质是：它的</font>**<font style="color:rgb(31, 35, 40);">输出值永远在0到1之间</font>**<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">顺便说一下，逻辑回归算法是</font>**<font style="color:rgb(31, 35, 40);">分类算法</font>**<font style="color:rgb(31, 35, 40);">，我们将它作为分类算法使用。有时候可能因为这个算法的名字中出现了“回归”使你感到困惑，但逻辑回归算法实际上是一种分类算法，它适用于标签 y 取值离散的情况，如：1 0 0 1。</font>

<font style="color:rgb(31, 35, 40);">在接下来的学习中，我们将开始学习逻辑回归算法的细节。</font> 

---

<h2 id="PGU1X">假设函数表示</h2>
<font style="color:rgb(31, 35, 40);">我们希望想出一个满足某个性质的假设函数，这个性质是它的预测值要在0和1之间。</font>

![图6.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745202847870-6e3435a0-16f3-473b-bef3-bd4c2086f8ed.png)

这里的$ g $函数也被称为sigmoid函数。

python代码实现：

```python
import numpy as np
    
def sigmoid(z):
   return 1 / (1 + np.exp(-z))
```

对模型的理解：$ g(z) = \frac{1}{1 + e^{-z}} $的作用是，对于给定的输入变量，根据选择的参数计算输出变量=1的**可能性**（estimated probablity）即$ h_\theta \left( x \right)=P\left( y=1|x; \;\theta \right) $，如果对于给定的$ x $，通过已经确定的参数计算得出$ h_\theta \left( x \right)=0.7 $，则表示有70%的几率$ y $为正向类，相应地$ y $为负向类的几率为1-0.7=0.3。

---

<h2 id="mrdbQ">决策界限</h2>
![图6.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745204068528-438a6dd3-b9bc-4dcc-8127-24c01cf47ef9.png)

上图展示了如何根据输入数据进行二分类。

假设我们有一个图6.4所示的模型，<font style="color:rgb(31, 35, 40);">并且参数</font>$ \theta $<font style="color:rgb(31, 35, 40);"> 是向量[-3 1 1]。 则当</font>$ -3+{x_1}+{x_2} \geq 0 $<font style="color:rgb(31, 35, 40);">，即</font>$ {x_1}+{x_2} \geq 3 $<font style="color:rgb(31, 35, 40);">时，模型将预测 y=1。 我们可以绘制直线</font>$ {x_1}+{x_2} = 3 $<font style="color:rgb(31, 35, 40);">，这条线便是我们模型的分界线，将预测为1的区域和预测为 0的区域分隔开。这条分界线就称为</font>**<font style="color:rgb(31, 35, 40);">决策界限</font>**<font style="color:rgb(31, 35, 40);">。</font>

![图6.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745204522078-f9564391-84cd-45a7-bb66-698e4edcac8e.png)

<font style="color:rgb(31, 35, 40);">假使我们的数据呈现图6.5这样的分布情况，怎样的模型才能适合呢？</font>

![图6.5](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745204851975-575335fb-cb89-4dfd-aff8-8977f95c2e38.png)

<font style="color:rgb(31, 35, 40);">因为需要用曲线才能分隔 y=0 的区域和 y=1 的区域，我们需要二次方特征：</font>$ {h_\theta}\left( x \right)=g\left( {\theta_0}+{\theta_1}{x_1}+{\theta_{2}}{x_{2}}+{\theta_{3}}x_{1}^{2}+{\theta_{4}}x_{2}^{2} \right) $<font style="color:rgb(31, 35, 40);">，其中</font>$ \theta =  $<font style="color:rgb(31, 35, 40);">[-1 0 0 1 1]，我们得到的判定边界恰好是圆点在原点且半径为1的圆形。</font>

此外，<font style="color:rgb(31, 35, 40);">我们还可以用非常复杂的模型来适应非常复杂形状的判定边界。</font>

---

<h2 id="t6bGn">代价函数</h2>
现在我们学习如何根据样本数据拟合逻辑回归模型。

![图6.6](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745205268761-f979fd7c-a6b0-400c-ae8f-9c6f4f26d885.png)

<font style="color:rgb(31, 35, 40);">对于线性回归模型，我们定义的代价函数是所有模型误差的平方和。理论上来说，我们也可以对逻辑回归模型沿用这个定义，但是问题在于，当我们将</font>$ {h_\theta}\left( x \right)=\frac{1}{1+{e^{-\theta^{T}x}}} $<font style="color:rgb(31, 35, 40);">带入到这样定义了的代价函数中时，我们得到的代价函数将是一个非凸函数：</font>

![图6.7](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745209111719-95cababf-ab84-42e7-b583-701ef6167ae3.png)

<font style="color:rgb(31, 35, 40);">这意味着我们的代价函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。</font>

<font style="color:rgb(31, 35, 40);">线性回归的代价函数为：</font>$ J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{1}{2}{{\left( {h_\theta}\left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}} $<font style="color:rgb(31, 35, 40);">。 我们重新定义逻辑回归的代价函数为：</font>$ J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)} $<font style="color:rgb(31, 35, 40);">，其中</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745209226299-f561d1ad-f484-46cb-ba18-326d3b20b49f.png)

我们可以看到样本不同y值的代价函数图像：

![图6.8](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745209540313-ac213241-fdf7-466e-93cd-ee2297dd5dc1.png)

<font style="color:rgb(31, 35, 40);">这样构建的</font>$ Cost\left( {h_\theta}\left( x \right),y \right) $<font style="color:rgb(31, 35, 40);">函数的特点是：当实际样本的 y=1 且</font>$ {h_\theta}\left( x \right) $<font style="color:rgb(31, 35, 40);">也为 1 时误差为 0，当 y=1 但</font>$ {h_\theta}\left( x \right) $<font style="color:rgb(31, 35, 40);">不为1时误差随着</font>$ {h_\theta}\left( x \right) $<font style="color:rgb(31, 35, 40);">变小而激增。</font>

![图6.9](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745209713858-8f568efb-a624-4a3a-b757-cd480f8ba3c6.png)

<font style="color:rgb(31, 35, 40);">同理，当实际的 y=0 且</font>$ {h_\theta}\left( x \right) $<font style="color:rgb(31, 35, 40);">也为 0 时代价为 0，当</font>$ y=0 $<font style="color:rgb(31, 35, 40);"> 但</font>$ {h_\theta}\left( x \right) $<font style="color:rgb(31, 35, 40);">不为 0时误差随着 hθ(x)的变大而变大。 </font>

---

<h2 id="uNx90">简化的代价函数和梯度下降</h2>
<font style="color:rgb(31, 35, 40);">将构建的</font>$ Cost \Big(h_{\theta}(x), y  \Big) $<font style="color:rgb(31, 35, 40);">简化如下：</font>$ Cost \big(h_{\theta}(x), y  \big) = -y\times\log\big( h_{\theta}(x) \big) - (1-y)\times\log \big(1 - h_{\theta}(x) \big) $

<font style="color:rgb(31, 35, 40);">带入代价函数得到：</font>$ J\left( \theta \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{\Big[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)\Big]} $

我们可以看到，根据样本$ y^{(i)} $取值的不同，代价函数取的term也不同，正好对上前面的公式。

```python
import numpy as np
    
def cost(theta, X, y):
  theta = np.matrix(theta)
  X = np.matrix(X)
  y = np.matrix(y)
  first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
  return np.sum(first - second) / (len(X))
```

求导后我们可以得到迭代公式：

$ \theta_j := \theta_j - \alpha\frac{1}{m}\sum_{i = 1}^{m}\big( h_{\theta}(x^{(i)}) - y^{(i)} \big)x_j $

不难发现，这个递推公式与我们前面的线性回归的递归公式形式上是一样的，唯一的不同就在于假设函数多套了一层sigmoid函数。由于假设函数发生了变化，所以逻辑回归和线性回归实际上是两个完全不同的东西。

<font style="color:rgb(31, 35, 40);">最后还有一点，我们之前在谈线性回归时讲到的</font>**<font style="color:rgb(31, 35, 40);">特征缩放</font>**<font style="color:rgb(31, 35, 40);">，我们看到了特征缩放是如何提高梯度下降的收敛速度的，这个特征缩放的方法，也适用于逻辑回归。如果你的特征范围差距很大的话，那么应用特征缩放的方法，同样也可以让逻辑回归中，梯度下降收敛更快。</font>

---

<h2 id="ahFz2">高级优化</h2>
<font style="color:rgb(31, 35, 40);">在上一个视频中，我们讨论了用梯度下降的方法最小化逻辑回归中代价函数</font>$ J\left( \theta \right) $<font style="color:rgb(31, 35, 40);">。在本节内容，我们会看到一些高级优化算法和一些高级的优化概念，利用这些方法，我们就能够使通过梯度下降，进行逻辑回归的速度大大提高，而这也将使算法更加适合解决大型的机器学习问题，比如，我们有数目庞大的特征量。 </font>

<font style="color:rgb(31, 35, 40);">梯度下降并不是我们可以使用的唯一算法，还有其他一些算法，更高级、更复杂。我们能用这些方法来计算代价函数</font>$ J\left( \theta \right) $<font style="color:rgb(31, 35, 40);">和偏导数项</font>$ \frac{\partial }{\partial {\theta_j}}J\left( \theta \right) $<font style="color:rgb(31, 35, 40);">两个项，</font>**<font style="color:rgb(31, 35, 40);">共轭梯度法</font>**<font style="color:rgb(31, 35, 40);">、</font>**<font style="color:rgb(31, 35, 40);">BFGS</font>**<font style="color:rgb(31, 35, 40);"> (</font>**<font style="color:rgb(31, 35, 40);">变尺度法</font>**<font style="color:rgb(31, 35, 40);">) 和</font>**<font style="color:rgb(31, 35, 40);">L-BFGS</font>**<font style="color:rgb(31, 35, 40);"> (</font>**<font style="color:rgb(31, 35, 40);">限制变尺度法</font>**<font style="color:rgb(31, 35, 40);">) 就是其中一些更高级的优化算法，它们需要有一种方法来计算</font>$ J\left( \theta \right) $<font style="color:rgb(31, 35, 40);">，以及需要一种方法计算导数项，然后使用比梯度下降更复杂的算法来最小化代价函数。这三种算法的具体细节超出了本门课程的范畴，此处我们只关注它们的一些特性。</font>

<font style="color:rgb(31, 35, 40);">这三种算法有许多优点：一个是使用这其中任何一个算法，你通常不需要手动选择学习率 </font>$ \alpha $<font style="color:rgb(31, 35, 40);">，所以对于这些算法的一种思路是，给出计算导数项和代价函数的方法，你可以认为算法有一个智能的内部循环，而且，事实上，他们确实有一个智能的内部循环，称为</font>**<font style="color:rgb(31, 35, 40);">线性搜索</font>**<font style="color:rgb(31, 35, 40);">(</font>**<font style="color:rgb(31, 35, 40);">line search</font>**<font style="color:rgb(31, 35, 40);">)算法，它可以自动尝试不同的学习速率</font>$ \alpha $<font style="color:rgb(31, 35, 40);">，并自动选择一个好的学习速率，因此它甚至可以为每次迭代选择不同的学习速率，那么你就不需要自己选择。这些算法实际上在做更复杂的事情，不仅仅是选择一个好的学习速率，所以它们往往最终比梯度下降收敛得快多了，不过关于它们到底做什么的详细讨论，已经超过了本门课程的范围。</font>

---

<h2 id="T2Ifl">多类别分类：一对多</h2>
<font style="color:rgb(31, 35, 40);">先看这样一些例子：</font>

<font style="color:rgb(31, 35, 40);">第一个例子：假如说你现在需要一个学习算法能自动地将邮件归类到不同的文件夹里，或者说可以自动地加上标签，那么，你也许需要一些不同的文件夹，或者不同的标签来完成这件事，来区分开来自工作的邮件、来自朋友的邮件、来自家人的邮件或者是有关兴趣爱好的邮件，那么，我们就有了这样一个分类问题：其类别有四个，分别用</font>$ y=1 $<font style="color:rgb(31, 35, 40);">、</font>$ y=2 $<font style="color:rgb(31, 35, 40);">、</font>$ y=3 $<font style="color:rgb(31, 35, 40);">、</font>$ y=4 $<font style="color:rgb(31, 35, 40);"> 来代表。</font>

<font style="color:rgb(31, 35, 40);">第二个例子是有关药物诊断的，如果一个病人因为鼻塞来到你的诊所，他可能并没有生病，用 y=1 这个类别来代表；或者患了感冒，用 y=2 来代表；或者得了流感用</font>$ y=3 $<font style="color:rgb(31, 35, 40);">来代表。</font>

<font style="color:rgb(31, 35, 40);">第三个例子：如果你正在做有关天气的机器学习分类问题，那么你可能想要区分哪些天是晴天、多云、雨天、或者下雪天，对上述所有的例子，</font>$ y $<font style="color:rgb(31, 35, 40);"> 可以取一个很小的数值，一个相对"谨慎"的数值，比如1 到3、1到4或者其它数值，以上说的都是多类分类问题，顺便一提的是，对于下标是0 1 2 3，还是 1 2 3 4 都不重要，我更喜欢将分类从 1 开始标而不是0，其实怎样标注都不会影响最后的结果。</font>

<font style="color:rgb(31, 35, 40);">然而对于之前的一个，二元分类问题，我们的数据看起来可能是像下图左；对于多分类问题，数据看起来则像是下图右：</font>

![图6.10](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745243681042-c3dc1cc7-c1ce-4ebf-a245-83093784099a.png)

<font style="color:rgb(31, 35, 40);">我用3种不同的符号来代表3个类别，问题就是给出3个类型的数据集，我们如何得到一个学习算法来进行分类呢？</font>

<font style="color:rgb(31, 35, 40);">我们现在已经知道如何进行二元分类，可以使用逻辑回归，对于直线或许你也知道，可以将数据集一分为二为正类和负类。用</font>**<font style="color:rgb(31, 35, 40);">一对多</font>**<font style="color:rgb(31, 35, 40);">的分类思想，我们可以将其用在多类分类问题上。下面将介绍如何进行一对多的分类工作，有时这个方法也被称为"一对余"方法。</font>

![图6.11](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745243843541-80a10940-157d-4cb2-ba93-46fc6ec2a5cb.png)

如上图，我们可以伪造一些假的负样本，将多分类转换为二分类问题。<font style="color:rgb(31, 35, 40);">最后我们得到一系列的模型简记为：</font>$ h_{\theta}(x)^{(i)} = p(y = i\mid x; \; \theta) $<font style="color:rgb(31, 35, 40);">，其中：</font>$ i=\left( 1,2,3....k \right) $<font style="color:rgb(31, 35, 40);">，第</font>$ i $<font style="color:rgb(31, 35, 40);">个假设函数给出了某个样本是</font>$ i $<font style="color:rgb(31, 35, 40);">分类的</font>**<font style="color:rgb(31, 35, 40);">概率</font>**<font style="color:rgb(31, 35, 40);">。选择出一个分类是可信度最高效果最好的，那么就可认为得到一个正确的分类。</font>

---

<h1 id="wVH5i">Chap7 正则化（regularization）</h1>
<h2 id="x4BOs">过拟合</h2>
<font style="color:rgb(31, 35, 40);">到现在为止，我们已经学习了几种不同的学习算法，包括线性回归和逻辑回归，它们能够有效地解决许多问题，但是当将它们应用到某些特定的机器学习应用时，会遇到</font>**<font style="color:rgb(31, 35, 40);">过拟合</font>**<font style="color:rgb(31, 35, 40);">(</font>_<font style="color:rgb(31, 35, 40);">over-fitting</font>_<font style="color:rgb(31, 35, 40);">)的问题，可能会导致它们效果很差。</font>

<font style="color:rgb(31, 35, 40);">在这一节，将解释什么是过度拟合问题，并且在此之后，我们将谈论一种称为</font>**<font style="color:rgb(31, 35, 40);">正则化</font>**<font style="color:rgb(31, 35, 40);">(</font>_<font style="color:rgb(31, 35, 40);">regularization</font>_<font style="color:rgb(31, 35, 40);">)的技术，它可以改善或者减少过度拟合问题。</font>

<font style="color:rgb(31, 35, 40);">如果我们有非常多的特征，我们通过学习得到的假设可能能够非常好地适应训练集（代价函数可能几乎为0），但是可能会不能推广到新的数据。</font>

<font style="color:rgb(31, 35, 40);">下图是一个回归问题的例子：</font>

![图7.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745388608048-ce822c30-3086-48aa-9aeb-48cba8a409f0.png)

<font style="color:rgb(31, 35, 40);">第一个模型是一个线性模型，欠拟合（</font>_<font style="color:rgb(31, 35, 40);">under-fitting</font>_<font style="color:rgb(31, 35, 40);">），不能很好地适应我们的训练集；第三个模型是一个四次方的模型，过于强调拟合原始数据，而丢失了算法的本质：预测新数据。我们可以看出，若给出一个新的值使之预测，它将表现的很差，这就是过拟合，虽然能非常好地适应我们的训练集，但在新输入变量进行预测时可能会效果不好；而中间的模型似乎最合适。</font>

<font style="color:rgb(31, 35, 40);">逻辑回归中也存在这样的问题：</font>

![图7.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745388894230-486ce06c-648c-493a-97d1-232482a15610.png)

<font style="color:rgb(31, 35, 40);">就以多项式理解，变量的幂次越高，拟合的越好，但相应的预测的能力就可能变差。</font>

<font style="color:rgb(31, 35, 40);">问题是，如果我们发现了过拟合问题，应该如何处理？</font>

1. <font style="color:rgb(31, 35, 40);">丢弃一些不能帮助我们正确预测的特征。可以是手工选择保留哪些特征，或者使用一些模型选择的算法来帮忙（例如</font>**<font style="color:rgb(31, 35, 40);">PCA</font>**<font style="color:rgb(31, 35, 40);">）</font>
2. <font style="color:rgb(31, 35, 40);">正则化。 保留所有的特征，但是减少参数</font>$ \theta_j $<font style="color:rgb(31, 35, 40);">的大小（</font>**<font style="color:rgb(31, 35, 40);">magnitude</font>**<font style="color:rgb(31, 35, 40);">）。</font>

---

<h2 id="VcVEb">代价函数</h2>
<font style="color:rgb(31, 35, 40);">上面的回归问题中如果我们的模型是：</font>$ h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x^2 + \theta_3x^3 + \theta_4x^4 $<font style="color:rgb(31, 35, 40);">我们可以从之前的事例中看出，正是那些高次项导致了过拟合的产生，所以如果我们能让这些高次项的系数接近于0的话，我们就能很好的拟合了。 所以我们要做的就是在一定程度上减小这些参数</font>$ \theta $<font style="color:rgb(31, 35, 40);"> 的值，这就是正则化的基本方法。</font>

<font style="color:rgb(31, 35, 40);">比如，我们决定要减少</font>$ {\theta_{3}} $<font style="color:rgb(31, 35, 40);">和</font>$ {\theta_{4}} $<font style="color:rgb(31, 35, 40);">的大小，我们要做的便是修改代价函数，在其中</font>$ {\theta_{3}} $<font style="color:rgb(31, 35, 40);">和</font>$ {\theta_{4}} $<font style="color:rgb(31, 35, 40);"> 设置一点惩罚。这样做的话，我们在尝试最小化代价时也需要将这个惩罚纳入考虑中，并最终导致选择较小一些的</font>$ {\theta_{3}} $<font style="color:rgb(31, 35, 40);">和</font>$ {\theta_{4}} $<font style="color:rgb(31, 35, 40);">。 修改后的代价函数如下：</font>

$ \underset{\theta }{\mathop{\min }}\Big\{\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{\left( {{h}_{\theta }}\left( {{x}^{(i)}} \right)-{{y}^{(i)}} \right)}^{2}}+1000\theta _{3}^{2}+1000\theta _{4}^{2}]} \Big\} $

<font style="color:rgb(31, 35, 40);">通过这样的代价函数选择出的</font>$ {\theta_{3}} $<font style="color:rgb(31, 35, 40);">和</font>$ {\theta_{4}} $<font style="color:rgb(31, 35, 40);">对预测结果的影响就比之前要小许多。假如我们有非常多的特征，我们并不知道其中哪些特征我们要惩罚，我们就选择对所有的特征进行惩罚，并且让代价函数最优化的软件来选择这些惩罚的程度。这样的结果是得到了一个较为简单的能防止过拟合问题的假设：</font>$ J\left( \theta \right)=\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta_{j}^{2}}]} $

<font style="color:rgb(31, 35, 40);">其中</font>$ \lambda $<font style="color:rgb(31, 35, 40);">又称为</font>**<font style="color:rgb(31, 35, 40);">正则化参数</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">Regularization Parameter</font>**<font style="color:rgb(31, 35, 40);">）。 注：根据惯例，我们不对</font>$ {\theta_{0}} $<font style="color:rgb(31, 35, 40);"> 进行惩罚。经过正则化处理的模型与原模型的可能对比如下图所示：</font>

![图7.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745390104488-f1621d5e-b236-4e40-82fe-65976e5866da.png)

<font style="color:rgb(31, 35, 40);">如果选择的正则化参数</font>$ \lambda $<font style="color:rgb(31, 35, 40);"> 过大，则会把所有的参数都最小化了，导致模型变成 </font>$ h_{\theta}(x) = \theta_0 $<font style="color:rgb(31, 35, 40);">，也就是变成一条平行于x轴的直线，造成欠拟合。 所以对于正则化，我们要取一个合理的 </font>$ \lambda $<font style="color:rgb(31, 35, 40);"> 的值，这样才能更好的应用正则化。</font>

---

<h2 id="f5FkF">线性回归的正则化</h2>
根据前面介绍的正则化方法， 我们可以将线性回归的梯度下降迭代更新为：

$ {{\theta }_{0}}:={{\theta }_{0}}-\alpha\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{0}^{(i)} $

$ {{\theta }_{j}}:={{\theta }_{j}}-\alpha \Big[ \frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{j}^{(i)} + \frac{\lambda}{m}\theta_j \Big] \; , \; for \; j = 1,2,3,...,n $

对于加上了惩罚的参数，我们还可以写为：

$ {{\theta }_{j}}:={(1 - \alpha\frac{\lambda}{m}){\theta }_{j}}-\alpha \frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{j}^{(i)} \; , \; for \; j = 1,2,3,...,n $

其中$ 1 - \alpha\frac{\lambda}{m} $通常是一个略小于1的数字，我们可以看作在原先梯度下降的基础上收缩了参数$ \theta_j $。

<font style="color:rgb(31, 35, 40);">我们同样也可以利用正规方程来求解正则化线性回归模型：</font>

![图7.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745391491773-9379aab5-b0d7-49d6-8515-c46228d6173d.png)

数学上可以证明，当$ \lambda > 0 $时，括号里面的矩阵是非奇异矩阵。

---

<h2 id="lQGL1">逻辑回归的正则化</h2>
对于逻辑回归，我们同样为其代价函数增加一项以进行正则化：

$ J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{\Big[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right) \Big]} + \frac{\lambda}{2m}\sum_{j = 1}^{n}\theta_j^2  $

同样地，我们将梯度下降迭代更新为：

$ {{\theta }_{0}}:={{\theta }_{0}}-\alpha\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{0}^{(i)} $

$ {{\theta }_{j}}:={{\theta }_{j}}-\alpha \Big[ \frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{j}^{(i)} + \frac{\lambda}{m}\theta_j \Big] \; , \; for \; j = 1,2,3,...,n $

计算正则化后代价函数的python代码：

```python
import numpy as np

def costReg(theta, X, y, learningRate):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X*theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X*theta.T)))
    reg = (learningRate / (2 * len(X))* np.sum(np.power(theta[:,1:theta.shape[1]],2))
    return np.sum(first - second) / (len(X)) + reg
```

---

<h1 id="O3ocN">Chap8 神经网络：表述（Neural Network：Representation）</h1>
<h2 id="n79Ql">非线性假设</h2>
<font style="color:rgb(31, 35, 40);">我们之前学的，无论是线性回归还是逻辑回归都有这样一个缺点，即：当特征太多时，计算的负荷会非常大。</font>

![图8.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745463360156-3cab50be-929e-49a3-9012-124e35985e96.png)

<font style="color:rgb(31, 35, 40);">加入拥有大于100个变量，我们希望用这100个特征来构建一个非线性的多项式模型，结果将是数量非常惊人的特征组合，即便我们只采用两两特征的组合</font>$ (x_1x_2+x_1x_3+x_1x_4+...+x_2x_3+x_2x_4+...+x_{99}x_{100}) $<font style="color:rgb(31, 35, 40);">，我们也会有接近5000个组合而成的特征。这对于一般的逻辑回归来说需要计算的特征太多了。</font>

<font style="color:rgb(31, 35, 40);">假设我们希望训练一个模型来识别视觉对象（例如识别一张图片上是否是一辆汽车），我们怎样才能这么做呢？一种方法是我们利用很多汽车的图片和很多非汽车的图片，然后利用这些图片上一个个像素的值（饱和度或亮度）来作为特征。</font>

<font style="color:rgb(31, 35, 40);">假如我们只选用灰度图片，每个像素则只有一个值（而非RGB值），我们可以选取图片上的两个不同位置上的两个像素，然后训练一个逻辑回归算法利用这两个像素的值来判断图片上是否是汽车：</font>

| <br/>![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745463565721-76275006-9496-41fd-b768-7a9a846be6fe.png) | ![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745463576954-10291b72-6809-4bc1-aac2-f6e6d39c1a29.png) | ![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745463589138-f7d3ae82-a332-46ac-b4a7-06d86663ba11.png) |
| --- | --- | --- |


<font style="color:rgb(31, 35, 40);">假使我们采用的都是50x50像素的小图片，并且我们将所有的像素视为特征，则会有 2500个特征，如果我们要进一步将两两特征组合构成一个多项式模型，则会有约</font>$ {{2500}^{2}}/2 $<font style="color:rgb(31, 35, 40);">个（接近3百万个）特征。普通的逻辑回归模型，不能有效地处理这么多的特征，这时候我们需要神经网络。</font>

---

<h2 id="fTfkX">模型表示</h2>
<font style="color:rgb(31, 35, 40);">为了构建神经网络模型，我们需要首先思考大脑中的神经网络是怎样的？每一个神经元都可以被认为是一个处理单元/神经核（</font>_<font style="color:rgb(31, 35, 40);">processing unit/Nucleus</font>_<font style="color:rgb(31, 35, 40);">），它含有许多输入/树突（</font>_<font style="color:rgb(31, 35, 40);">input/Dendrite</font>_<font style="color:rgb(31, 35, 40);">），并且有一个输出/轴突（</font>_<font style="color:rgb(31, 35, 40);">output/Axon</font>_<font style="color:rgb(31, 35, 40);">）。神经网络是大量神经元相互链接并通过电脉冲来交流的一个网络。</font>

![图8.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745478923942-7e225a6d-4e3f-4a14-a2e7-406c4609b5d6.png)

一个神经元的输出可能会传递给下一个神经元的输入。<font style="color:rgb(31, 35, 40);">神经网络模型建立在很多神经元之上，每一个神经元又是一个个学习模型。这些神经元（也叫</font>**<font style="color:rgb(31, 35, 40);">激活单元</font>**<font style="color:rgb(31, 35, 40);">，</font>**<font style="color:rgb(31, 35, 40);">activation unit</font>**<font style="color:rgb(31, 35, 40);">）接受一些特征作为输入，并且根据本身的模型提供一个输出。下图是一个以逻辑回归模型作为自身学习模型的神经元示例，在神经网络中，参数又可被称为</font>**<font style="color:rgb(31, 35, 40);">权重</font>**<font style="color:rgb(31, 35, 40);">（weight），这个逻辑回归模型的sigmoid函数也被称为</font>**<font style="color:rgb(31, 35, 40);">激活函数</font>**<font style="color:rgb(31, 35, 40);">（activation function）。</font>

![图8.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745479276789-c7a320d5-5557-4d95-b3dd-8c4bab02eb3c.png)

<font style="color:rgb(31, 35, 40);">假如我们有一个类似于神经元的神经网络，效果如下：</font>

![图8.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745479359946-f4c96aeb-29ea-42f3-afe4-c94b6333a155.png)

<font style="color:rgb(31, 35, 40);">其中</font>$ x_1 $<font style="color:rgb(31, 35, 40);">, </font>$ x_2 $<font style="color:rgb(31, 35, 40);">, </font>$ x_3 $<font style="color:rgb(31, 35, 40);">是</font>**<font style="color:rgb(31, 35, 40);">输入单元</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">input units</font>**<font style="color:rgb(31, 35, 40);">），我们将原始数据输入给它们。 </font>$ a_1 $<font style="color:rgb(31, 35, 40);">, </font>$ a_2 $<font style="color:rgb(31, 35, 40);">, </font>$ a_3 $<font style="color:rgb(31, 35, 40);">是中间单元，它们负责将数据进行处理，然后呈递到下一层。 最后是输出单元，它负责计算</font>$ {h_\theta}\left( x \right) $<font style="color:rgb(31, 35, 40);">。神经网络模型是许多逻辑单元按照不同层级组织起来的网络，每一层的输出变量都是下一层的输入变量。这是一个3层的神经网络，第一层成为</font>**<font style="color:rgb(31, 35, 40);">输入层</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">Input Layer</font>**<font style="color:rgb(31, 35, 40);">），最后一层称为</font>**<font style="color:rgb(31, 35, 40);">输出层</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">Output Layer</font>**<font style="color:rgb(31, 35, 40);">），中间一层成为</font>**<font style="color:rgb(31, 35, 40);">隐藏层</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">Hidden Layers</font>**<font style="color:rgb(31, 35, 40);">）。我们为每一层都增加一个偏差单位（</font>**<font style="color:rgb(31, 35, 40);">bias unit</font>**<font style="color:rgb(31, 35, 40);">）</font>$ x_0 $<font style="color:rgb(31, 35, 40);">、</font>$ a_0 $<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">下面引入一些标记法来帮助描述模型： </font>$ a_i^{(j)} $<font style="color:rgb(31, 35, 40);"> 代表第</font>$ j $<font style="color:rgb(31, 35, 40);">层的第</font>$ i $<font style="color:rgb(31, 35, 40);">个激活单元。矩阵</font>$ {{\Theta }^{\left( j \right)}} $<font style="color:rgb(31, 35, 40);">代表从第</font>$ j $<font style="color:rgb(31, 35, 40);">层映射到第</font>$ j+1 $<font style="color:rgb(31, 35, 40);">层时的权重的矩阵，例如</font>$ {{\Theta }^{\left( 1 \right)}} $<font style="color:rgb(31, 35, 40);">代表从第一层映射到第二层的权重的矩阵。其尺寸为：以第</font>$ j+1 $<font style="color:rgb(31, 35, 40);">层的激活单元数量为行数，以第</font>$ j $<font style="color:rgb(31, 35, 40);">层的激活单元数加一（偏差单位）为列数的矩阵。例如：上图所示的神经网络中</font>$ {{\Theta }^{\left( 1 \right)}} $<font style="color:rgb(31, 35, 40);">的尺寸为</font>$ 3\times4 $<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">对于上图所示的模型，各激活单元和输出分别表达为：</font>

$ a_{1}^{(2)}=g(\Theta_{10}^{(1)}{{x}_{0}}+\Theta_{11}^{(1)}{{x}_{1}}+\Theta_{12}^{(1)}{{x}_{2}}+\Theta_{13}^{(1)}{{x}_{3}}) $<font style="color:rgb(31, 35, 40);"> </font>$ a_{2}^{(2)}=g(\Theta_{20}^{(1)}{{x}_{0}}+\Theta_{21}^{(1)}{{x}_{1}}+\Theta_{22}^{(1)}{{x}_{2}}+\Theta_{23}^{(1)}{{x}_{3}}) $<font style="color:rgb(31, 35, 40);"> </font>$ a_{3}^{(2)}=g(\Theta_{30}^{(1)}{{x}_{0}}+\Theta_{31}^{(1)}{{x}_{1}}+\Theta_{32}^{(1)}{{x}_{2}}+\Theta_{33}^{(1)}{{x}_{3}}) $$ {{h}_{\Theta }}(x)=g(\Theta_{10}^{(2)}a_{0}^{(2)}+\Theta_{11}^{(2)}a_{1}^{(2)}+\Theta_{12}^{(2)}a_{2}^{(2)}+\Theta_{13}^{(2)}a_{3}^{(2)}) $



我们可以用$ z_{1}^{(2)} $表示第二层的第一个激活单元接受第一层输入的**线性组合**，即$ z_1^{(2)} = \Theta_{10}^{(1)}{{x}_{0}}+\Theta_{11}^{(1)}{{x}_{1}}+\Theta_{12}^{(1)}{{x}_{2}}+\Theta_{13}^{(1)}{{x}_{3}} $。第二层有4个激活单元（其中$ a_0 $是偏差项），所以$ z^{(2)} $是一个四维向量。

![图8.5](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745480570989-4e0b4444-c61a-4c05-a9ed-34d90308ad78.png)

<font style="color:rgb(31, 35, 40);">这只是针对训练集中一个训练实例所进行的计算。如果我们要对整个训练集进行计算，我们需要将训练集特征矩阵进行转置，使得同一个实例的特征都在同一列里。即： </font>$ z^{(2)}=\Theta^{(1)}X^T $<font style="color:rgb(31, 35, 40);">。</font>

这个计算的过程也被称为**前向传播**（_forward propagation_），因为我们从输入单元开始，计算第一层的各激活单元，并以此作为输入传递到第二层的各激活单元，循环往复，最终得到输出层的结果。



<font style="color:rgb(31, 35, 40);">为了更好了了解</font>**<font style="color:rgb(31, 35, 40);">Neuron Networks</font>**<font style="color:rgb(31, 35, 40);">的工作原理，我们先把左半部分遮住：</font>

![图8.6](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745481556163-cf7e4562-eb3f-4bce-a98f-d4c7000ad209.png)

<font style="color:rgb(31, 35, 40);">右半部分其实就是以</font>$ a_0, a_1, a_2, a_3 $<font style="color:rgb(31, 35, 40);">, 按照</font>**<font style="color:rgb(31, 35, 40);">Logistic Regression</font>**<font style="color:rgb(31, 35, 40);">的方式输出</font>$ h_\theta(x) $<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">其实神经网络就像是</font>**<font style="color:rgb(31, 35, 40);">logistic regression</font>**<font style="color:rgb(31, 35, 40);">，只不过我们把</font>**<font style="color:rgb(31, 35, 40);">logistic regression</font>**<font style="color:rgb(31, 35, 40);">中的输入向量</font>$ \left[ x_1\sim {x_3} \right] $<font style="color:rgb(31, 35, 40);"> 变成了中间层的</font>$ \left[ a_1^{(2)}\sim a_3^{(2)} \right] $<font style="color:rgb(31, 35, 40);">, 即: </font>$ h_{\theta}(x)=g(\Theta_0^{(2)}a_0^{(2)}+\Theta_1^{(2)}a_1^{(2)}+\Theta_2^{(2)}a_2^{(2)}+\Theta_3(2)a_3^{(2)}) $

<font style="color:rgb(31, 35, 40);"> 我们可以把</font>$ a_0, a_1, a_2, a_3 $<font style="color:rgb(31, 35, 40);">看成更为高级的特征值，也就是</font>$ x_0, x_1, x_2, x_3 $<font style="color:rgb(31, 35, 40);">的进化体，并且它们是由</font>$ x $<font style="color:rgb(31, 35, 40);">与</font>$ \Theta $<font style="color:rgb(31, 35, 40);">决定的，因为是梯度下降的，所以</font>$ a $<font style="color:rgb(31, 35, 40);">是变化的，并且变得越来越厉害，所以这些更高级的特征值远比仅仅使用x的某次方厉害，也能更好的预测新数据。 这就是神经网络相比于逻辑回归和线性回归的优势。</font>

![图8.7](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745482152323-363e3a58-752b-4882-b23b-394b07bd8d36.png)

现在我们清楚了前向传播在神经网络中的步骤，并且清楚了如何将这些 计算向量化。接下来我们讨论更加细节的例子，来展示如何使用神经网络来计算输入的非线性函数。

---

<h2 id="Ol6gt">例子和直观理解（1）</h2>
<font style="color:rgb(31, 35, 40);">从本质上讲，神经网络能够通过</font>**<font style="color:rgb(31, 35, 40);">学习</font>**<font style="color:rgb(31, 35, 40);">得出其自身的一系列特征（映射矩阵</font>$ \Theta $<font style="color:rgb(31, 35, 40);">的参数）。在普通的逻辑回归中，我们被限制为使用数据中的原始特征</font>$ x_1,x_2,...,{{x}_{n}} $<font style="color:rgb(31, 35, 40);">，我们虽然可以使用一些二项式项来组合这些特征，但是我们仍然受到这些原始特征的限制。在神经网络中，原始特征只是输入层，在我们上面三层的神经网络例子中，输出层做出的预测利用的是第二层的特征，而非输入层中的原始特征，我们可以认为第二层中的特征是神经网络通过</font>**<font style="color:rgb(31, 35, 40);">学习</font>**<font style="color:rgb(31, 35, 40);">后自己得出的一系列用于预测输出变量的新特征。</font>

<font style="color:rgb(31, 35, 40);">举例说明：假设我们需要一个实现逻辑与(AND)激活单元；如下图中左半部分是神经网络的设计与output层表达式，右边上部分是sigmod函数，下半部分是真值表。</font>

![图8.8](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745482918860-2e4cf67a-2b7c-4a6f-bd58-f543a2df3ad2.png)

<font style="color:rgb(31, 35, 40);">其中</font>$ \theta_0 = -30, \theta_1 = 20, \theta_2 = 20 $<font style="color:rgb(31, 35, 40);"> 我们的输出函数</font>$ h_\theta(x) $<font style="color:rgb(31, 35, 40);">即为：</font>$ h_\Theta(x)=g\left( -30+20x_1+20x_2 \right) $

所以我们可以得出与实际逻辑与运算一致的结果。而神经网络的任务，就是通过**学习**，找到这样的参数$ \Theta $。

逻辑或（OR）运算也是一样的：

![图8.9](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745483090472-fd1dd20f-2502-4f58-9889-1ce7240e9953.png)

---

<h2 id="tXXMR">例子和直观理解（2）</h2>
下面这个例子展示了逻辑补的激活单元：

![图8.10](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745504807447-42a90ab6-3e3d-4bb9-ad26-92a40f133ee7.png)

我们还可以利用神经元组合来实现更加复杂的运算，比如XNOR（两个输入相同时输出才为1）：

![图8.11](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745504923833-e5404790-b8d5-4639-a2d8-39a987445190.png)

$ a_1^{(2)} $就是一个逻辑与，$ a_2^{(2)} $就是中间那个青色的神经元。将这两个隐藏层取或后，就是XNOR了。

神经网络还可以用来识别手写字体，这也是一个使用神经网络的学习来完成复杂任务的例子。

![图8.12](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745505049854-26e37c71-b129-4a00-9a53-6f91e107bd50.png)

---

<h2 id="Im7Ig">多元分类</h2>
假设我们有一个计算机视觉任务，判断某张图片包含四个物体中的哪个。这个神经网络的输出是多元的（有四个），我们可以看成是一个四维向量。

![图8.13](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745505754781-506b3611-17c0-4580-9822-a0dc651613d9.png)

我们要做的就是对于每个样本$ (x^{(i)}, y^{(i)}) $，使得$ h_{\Theta}(x^{(i)}) \approx y^{(i)} $。

![图8.14](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745505846317-37cc21c6-1a1a-41a6-9aee-5aff31c48962.png)

至此，我们了解了神经网络的表示，下一部分我们将学习如何构建我们的训练集以及让神经网络自动地学习相应的参数。

---

<h1 id="exyzo">Chap9 神经网络：学习（Neural Network：Learning）</h1>
<h2 id="cWTPT">代价函数</h2>
<font style="color:rgb(31, 35, 40);">首先引入一些便于稍后讨论的新标记方法：</font>

<font style="color:rgb(31, 35, 40);">假设神经网络的训练样本有</font>$ m $<font style="color:rgb(31, 35, 40);">个，每个包含一组输入</font>$ x $<font style="color:rgb(31, 35, 40);">和一组输出信号</font>$ y $<font style="color:rgb(31, 35, 40);">，</font>$ L $<font style="color:rgb(31, 35, 40);">表示神经网络层数，</font>$ S_l $<font style="color:rgb(31, 35, 40);">表示每层的神经元个数，</font>$ S_L $<font style="color:rgb(31, 35, 40);">代表最后一层中神经元的个数（输出维度）。</font>

<font style="color:rgb(31, 35, 40);">将神经网络的分类定义为两种情况：二类分类和多类分类，</font>

<font style="color:rgb(31, 35, 40);">二类分类：</font>$ S_L=1,\; y=0, or, 1 $<font style="color:rgb(31, 35, 40);">表示哪一类；</font>

<font style="color:rgb(31, 35, 40);">多类分类：</font>$ S_L=k, \; y_i = 1 $<font style="color:rgb(31, 35, 40);">表示分到第</font>$ i $<font style="color:rgb(31, 35, 40);">类；</font>$ (k \geq 3) $

![图9.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745542460539-3ca9296d-ba16-42e0-8e52-5c1522f0c035.png)

<font style="color:rgb(31, 35, 40);">我们回顾逻辑回归问题中我们的代价函数为:</font>

$ J\left(\theta \right)=-\frac{1}{m}\left[\sum_\limits{i=1}^{m}{y}^{(i)}\log{h_\theta({x}^{(i)})}+\left(1-{y}^{(i)}\right)log\left(1-h_\theta\left({x}^{(i)}\right)\right)\right]+\frac{\lambda}{2m}\sum_\limits{j=1}^{n}{\theta_j}^{2} $

<font style="color:rgb(31, 35, 40);">在逻辑回归中，我们只有一个输出变量，又称标量（</font>_<font style="color:rgb(31, 35, 40);">scalar</font>_<font style="color:rgb(31, 35, 40);">），也只有一个因变量</font>$ y $<font style="color:rgb(31, 35, 40);">，但是在神经网络中，我们可以有很多输出变量，我们的</font>$ h_\theta(x) $<font style="color:rgb(31, 35, 40);">是一个维度为</font>$ K $<font style="color:rgb(31, 35, 40);">的向量，并且我们训练集中的因变量</font>$ y $<font style="color:rgb(31, 35, 40);">也是同样维度的一个向量，因此我们的代价函数会比逻辑回归更加复杂一些，为：</font>$ h_{\Theta}(x) \in R^K, \; (h_{\Theta}(x))_i = i^{th}output $

$ J(\Theta) = -\frac{1}{m}\sum\limits_{i=1}^{m} \sum_{k = 1}^{K}{\Big[{{y}_{k}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)_{k}+\left( 1-{{y}_{k}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)_{k} \Big]} + \frac{\lambda}{2m} \sum_{l = 1}^{L - 1} \sum_{i = 1}^{s_l} \sum_{j = 1}^{s_{l + 1}} (\Theta_{ji}^{(l)})^2  $

<font style="color:rgb(31, 35, 40);">这个看起来复杂很多的代价函数背后的思想还是一样的，我们希望通过代价函数来观察算法预测的结果与真实情况的</font>**<font style="color:rgb(31, 35, 40);">误差</font>**<font style="color:rgb(31, 35, 40);">有多大，唯一不同的是，对于每一行特征，我们都会给出</font>$ K $<font style="color:rgb(31, 35, 40);">个预测，基本上我们可以利用循环，对每一行特征都预测</font>$ K $<font style="color:rgb(31, 35, 40);">个不同结果，然后在利用循环在</font>$ K $<font style="color:rgb(31, 35, 40);">个预测中选择可能性最高的一个，将其与</font>$ y $<font style="color:rgb(31, 35, 40);">中的实际数据进行比较。</font>

<font style="color:rgb(31, 35, 40);">正则化的那一项只是排除了每一层的</font>$ \theta_0 $<font style="color:rgb(31, 35, 40);">后，每一层的</font>$ \Theta $<font style="color:rgb(31, 35, 40);">矩阵的和。最里层的循环</font>$ j $<font style="color:rgb(31, 35, 40);">循环所有的行（由</font>$ s_{l+1} $<font style="color:rgb(31, 35, 40);">层的激活单元数决定），循环</font>$ i $<font style="color:rgb(31, 35, 40);">则循环所有的列，由该层（</font>$ s_l $<font style="color:rgb(31, 35, 40);">层）的激活单元数所决定。</font>

---

<h2 id="uEP3R">反向传播算法（Back-propagation Algorithm）</h2>
<font style="color:rgb(31, 35, 40);">之前我们在计算神经网络预测结果的时候我们采用了一种正向传播方法，我们从第一层开始正向一层一层进行计算，直到最后一层的</font>$ h_{\theta}\left(x\right) $<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">现在，为了计算代价函数的偏导数</font>$ \frac{\partial}{\partial\Theta^{(l)}_{ij}}J\left(\Theta\right) $<font style="color:rgb(31, 35, 40);">，我们需要采用一种</font>**<font style="color:rgb(31, 35, 40);">反向传播</font>**<font style="color:rgb(31, 35, 40);">算法，也就是首先计算最后一层的误差，然后再一层一层反向求出各层的误差，直到倒数第二层。 以一个例子来说明反向传播算法。</font>

<font style="color:rgb(31, 35, 40);">假设我们的训练集只有一个样本</font>$ \left({x}^{(1)},{y}^{(1)}\right) $<font style="color:rgb(31, 35, 40);">，我们的神经网络是一个四层的神经网络，其中</font>$ K=4，S_{L}=4，L=4 $<font style="color:rgb(31, 35, 40);">：</font>

<font style="color:rgb(31, 35, 40);">前向传播算法：</font>

![图9.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745546413929-5dbc6758-fc49-4927-b3f7-7ce4716abe27.png)

上一层的神经元经过$ \Theta $映射之后，输入到下一层神经元中，再经过Sigmoid变换，成为其输出。

<font style="color:rgb(31, 35, 40);">我们从</font>**<font style="color:rgb(31, 35, 40);">最后一层</font>**<font style="color:rgb(31, 35, 40);">的误差开始计算，误差是激活单元的</font>**<font style="color:rgb(31, 35, 40);">预测</font>**<font style="color:rgb(31, 35, 40);">（</font>$ {a^{(4)}_k} $<font style="color:rgb(31, 35, 40);">）与</font>**<font style="color:rgb(31, 35, 40);">实际值</font>**<font style="color:rgb(31, 35, 40);">（</font>$ y_k $<font style="color:rgb(31, 35, 40);">）之间的误差，（</font>$ k \in [1, K] $<font style="color:rgb(31, 35, 40);">）。</font>

<font style="color:rgb(31, 35, 40);">我们用</font>$ \delta $<font style="color:rgb(31, 35, 40);">来表示误差，则向量化的表示为：</font>$ \delta^{(4)}=a^{(4)}-y $<font style="color:rgb(31, 35, 40);">，我们利用这个误差值来计算</font>**<font style="color:rgb(31, 35, 40);">前一层</font>**<font style="color:rgb(31, 35, 40);">的误差：</font>$ \delta^{(3)}=\left({\Theta^{(3)}}\right)^{T}\delta^{(4)}\ast g^{\prime} \left(z^{(3)}\right) $<font style="color:rgb(31, 35, 40);">其中</font>$ g^{\prime} \left(z^{(3)}\right) $<font style="color:rgb(31, 35, 40);">是 S 形函数Sigmoid的导数，由于</font>$ g(z) = \frac{1}{1+e^{-z}} $<font style="color:rgb(31, 35, 40);">，有：</font>$ g'(z^{(3)})= g(z^{(3)}) \big(1 - g(z^{(3)}) \big) = a^{(3)}\ast(1-a^{(3)}) $<font style="color:rgb(31, 35, 40);">。而</font>$ (\Theta^{(3)})^{T}\delta^{(4)} $<font style="color:rgb(31, 35, 40);">则是权重导致的误差的和。</font>

<font style="color:rgb(31, 35, 40);">下一步是继续计算第二层的误差： </font>$ \delta^{(2)}=(\Theta^{(2)})^{T}\delta^{(3)}\ast g'(z^{(2)}) $<font style="color:rgb(31, 35, 40);">。而第一层由于是输入变量，故不存在误差。</font>

<font style="color:rgb(31, 35, 40);">我们有了所有的误差的表达式后，便可以计算代价函数的偏导数了，假设</font>$ λ=0 $<font style="color:rgb(31, 35, 40);">，即我们</font>**<font style="color:rgb(31, 35, 40);">不做任何正则化处理</font>**<font style="color:rgb(31, 35, 40);">时有： </font>$ \frac{\partial}{\partial\Theta^{(l)}_{ij}}J\left(\Theta\right) = a_j^{(l)} \delta_i^{(l+1)} $

<font style="color:rgb(31, 35, 40);">重要的是清楚地知道上面式子中上下标的含义：</font>

$ l $<font style="color:rgb(31, 35, 40);">代表目前所计算的是第几层；</font>$ j $<font style="color:rgb(31, 35, 40);">代表目前计算层中的激活单元的下标，也将是下一层的第</font>$ j $<font style="color:rgb(31, 35, 40);">个输入变量的下标；</font>$ i $<font style="color:rgb(31, 35, 40);">代表下一层中误差单元的下标，是受到权重矩阵中第</font>$ i $<font style="color:rgb(31, 35, 40);">行影响的下一层中的误差单元的下标。</font>

<font style="color:rgb(31, 35, 40);">如果我们考虑正则化处理，并且我们的训练集是一个特征矩阵而非向量。在上面的特殊情况中，我们需要计算每一层的误差单元来计算代价函数的偏导数。在更为一般的情况中，我们同样需要计算每一层的误差单元，此时的误差单元是一个矩阵，我们用</font>$ \Delta^{(l)}_{ij} $<font style="color:rgb(31, 35, 40);">来表示这个</font>**<font style="color:rgb(31, 35, 40);">误差矩阵</font>**<font style="color:rgb(31, 35, 40);">。第</font>$ l $<font style="color:rgb(31, 35, 40);">层的第</font>$ i $<font style="color:rgb(31, 35, 40);">个激活单元受到第</font>$ j $<font style="color:rgb(31, 35, 40);">个参数影响而导致的</font>**<font style="color:rgb(31, 35, 40);">误差</font>**<font style="color:rgb(31, 35, 40);">。</font>

给出反向传播的步骤：

![图9.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745548634266-f85257b8-016a-467d-ae33-148465c45cf5.png)

进一步地，我们可以可以将最后一步向量化，即$ \Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)} (a^{(l)})^T $

我们可以将$ \Delta_{ij}^{(l)} $看为是$ \Theta_{ij}^{(l)} $的变化情况。最后我们计算：

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745548666709-e26af4ac-859a-4921-b555-65d8b8734c55.png)

数学上可以证明：![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745548688612-5c0c9b34-d9fc-4c27-a031-0b6e3e4d68d8.png)，即神经网络中代价函数$ J(\Theta) $关于参数$ \Theta_{ij}^{(l)} $的导数就是$ D_{ij}^{(l)} $。

（有关详细推导可见：[https://blog.csdn.net/qq_29762941/article/details/80343185](https://blog.csdn.net/qq_29762941/article/details/80343185)）

---

<h2 id="E8H1q">理解反向传播</h2>
前向传播示意图：

![图9.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745635624924-73b6cfe8-e0d0-4376-915d-4642d17d209d.png)

而反向传播做的是：

![图9.5](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745635684472-9b08a737-d985-489e-abc0-2cdb0045db08.png)

---

<h2 id="qC6rq">梯度检验（Gradient Checking）</h2>
<font style="color:rgb(31, 35, 40);">当我们对一个较为复杂的模型（例如神经网络）使用梯度下降算法时，可能会存在一些不容易察觉的错误，意味着，虽然代价看上去在不断减小，但最终的结果可能并不是最优解。</font>

<font style="color:rgb(31, 35, 40);">为了避免这样的问题，我们采取一种叫做梯度的数值检验（</font>**<font style="color:rgb(31, 35, 40);">Numerical Gradient Checking</font>**<font style="color:rgb(31, 35, 40);">）方法。这种方法的思想是通过</font>**<font style="color:rgb(31, 35, 40);">估计梯度值</font>**<font style="color:rgb(31, 35, 40);">来检验我们计算的导数值是否真的是我们期望的。</font>

<font style="color:rgb(31, 35, 40);">对梯度的估计采用的方法是在代价函数上沿着切线的方向选择离两个非常近的点然后计算两个点的斜率用以估计梯度。即对于某个特定的</font>$ \Theta_{ij}^{(l)} $<font style="color:rgb(31, 35, 40);">，我们计算出在</font>$ \Theta_{ij}^{(l)} - \varepsilon $<font style="color:rgb(31, 35, 40);">处和</font>$ \Theta_{ij}^{(l)} + \varepsilon $<font style="color:rgb(31, 35, 40);">的代价值（</font>$ \varepsilon $<font style="color:rgb(31, 35, 40);"> 是一个非常小的值，通常选取</font>$ 10^{-4} $<font style="color:rgb(31, 35, 40);">），然后求两个代价的平均，用以估计在</font>$ \Theta_{ij}^{(l)} $<font style="color:rgb(31, 35, 40);">处的偏导数值。</font>

![图9.6](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745637228702-ee36d5ad-41aa-4132-99b0-1035b8f652c5.png)

<font style="color:rgb(31, 35, 40);">下面是一个只针对</font>$ \theta_1 $<font style="color:rgb(31, 35, 40);">进行检验的示例：</font>

$ \frac{\partial}{\partial\theta_1}=\frac{J\left(\theta_1+\varepsilon_1,\theta_2,\theta_3...\theta_n \right)-J \left( \theta_1-\varepsilon_1,\theta_2,\theta_3...\theta_n \right)}{2\varepsilon} $

<font style="color:rgb(31, 35, 40);">最后我们还需要对通过反向传播方法计算出的偏导数进行检验。</font>

<font style="color:rgb(31, 35, 40);">根据上面的算法，计算出的偏导数存储在矩阵</font>$ D_{ij}^{(l)} $<font style="color:rgb(31, 35, 40);">中。检验时，我们要将该矩阵展开成为向量，同时我们也将</font>$ \Theta $<font style="color:rgb(31, 35, 40);">矩阵展开为向量，我们针对每一个</font>$ \Theta_{ij}^{(l)} $<font style="color:rgb(31, 35, 40);">都计算一个近似的梯度值，将这些值存储于一个近似梯度矩阵中，最终将得出的这个矩阵同</font>$ D_{ij}^{(l)} $<font style="color:rgb(31, 35, 40);">进行比较，如果两者的差距足够小的话，那么我们可以认为当前编写的代码是正确的。</font>

非常重要的一点是，当我们在第一次学习迭代时，通过梯度检验确定了当前梯度下降**无误**后，记得**关闭**梯度检验，否则再往后每一次学习迭代进行梯度下降时，都会进行梯度检验，而梯度检验往往拥有很大的计算量，且速度很慢，这样子会造成整个算法耗费巨大的时间。

---

<h2 id="Idl0C">随机初始化</h2>
如果我们像之前一样，将$ \Theta $的所有参数都初始化为零，将引发非常严重的后果，每一次学习迭代后，所有的参数都拥有相同的值：

![图9.7](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745643206131-971067d7-1094-49f4-9709-d28446ea571e.png)

这个问题有时也被称为：对称权重问题（symmetric weights）。为了避免这样的情况，我们需要将$ \Theta $的各参数初始化为接近零的**随机数**：$ \Theta_{ij}^{(l)} \in [-\varepsilon, \varepsilon] $

---

<h1 id="sFawF">Chap10 应用机器学习的建议</h1>
<h2 id="u89vY">评估一个假设</h2>
<font style="color:rgb(31, 35, 40);">在本节中我们介绍一下怎样评估假设函数。在之后的讨论中，我们将以此为基础来讨论如何避免过拟合和欠拟合的问题。</font>

<font style="color:rgb(31, 35, 40);">那么，该如何判断一个假设函数是过拟合的呢？对于一些简单的例子，我们可以对假设函数</font>$ h_{\theta}(x) $<font style="color:rgb(31, 35, 40);">进行画图，然后观察图形趋势，但对于有很多特征变量的情况，想要通过画出假设函数来进行观察，就会变得很难甚至是不可能实现。</font>

 ![图10.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745822996844-bf56e2f8-378d-45b7-b0f4-ae5f72bbc7ab.png)

<font style="color:rgb(31, 35, 40);">因此，我们需要另一种方法来评估我们的假设函数是否过拟合。</font>

<font style="color:rgb(31, 35, 40);">为了检验算法是否过拟合，我们将数据分成</font>**<font style="color:rgb(31, 35, 40);">训练集</font>**<font style="color:rgb(31, 35, 40);">和</font>**<font style="color:rgb(31, 35, 40);">测试集</font>**<font style="color:rgb(31, 35, 40);">，通常用70%的数据作为训练集，用剩下30%的数据作为测试集。很重要的一点是训练集和测试集均要含有各种类型的数据，通常我们要对数据进行</font>**<font style="color:rgb(31, 35, 40);">随机打乱</font>**<font style="color:rgb(31, 35, 40);">，然后再分成训练集和测试集。</font>

![图10.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745823178078-54711b1a-9a4c-4a9c-a1f4-49756db16bab.png)

<font style="color:rgb(31, 35, 40);">测试集评估在通过训练集让我们的模型学习得出其参数后，对测试集运用该模型，我们有两种方式计算误差：</font>

1. <font style="color:rgb(31, 35, 40);">对于线性回归模型，我们利用测试集数据计算代价函数</font>$ J $
2. <font style="color:rgb(31, 35, 40);">对于逻辑回归模型，我们可以利用测试数据集来计算代价函数：</font>

$ J_{test}{(\theta)} = -\frac{1}{{m}_{test}}\sum\limits_{i=1}^{m_{test}}\log\big({h_{\theta}(x^{(i)}{test})} \big)+(1-{y^{(i)}{test}})\log \big( {h_{\theta}(x^{(i)}_{test})} \big) $

<font style="color:rgb(31, 35, 40);">也可以对于每一个测试集样本，计算：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745823243702-8ad390ce-cf6f-4924-a122-71fd62d73178.png)

最后，我们就可以根据拟合完毕的模型在测试集的表现，评估这个模型是否过拟合。

---

<h2 id="wX2uH">模型选择和交叉验证集</h2>
当我们面临一个机器学习的问题时，我们可能需要决定假设函数的次幂。

<font style="color:rgb(31, 35, 40);">假设我们要在10个不同次数的二项式模型之间进行选择：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745824373371-351efe37-382b-4eb8-a8f8-6717a2e45fb5.png)

<font style="color:rgb(31, 35, 40);">显然越高次数的多项式模型越能够适应我们的训练数据集，但是适应训练数据集并不代表着能推广至一般情况，我们应该选择一个更能适应一般情况的模型。我们需要使用</font>**<font style="color:rgb(31, 35, 40);">交叉验证集</font>**<font style="color:rgb(31, 35, 40);">（</font>_<font style="color:rgb(31, 35, 40);">Cross Validation</font>_<font style="color:rgb(31, 35, 40);">）来帮助选择模型。</font>

<font style="color:rgb(31, 35, 40);">即：使用60%的数据作为训练集，使用 20%的数据作为</font>**<font style="color:rgb(31, 35, 40);">交叉验证集</font>**<font style="color:rgb(31, 35, 40);">，使用20%的数据作为测试集。</font>

<font style="color:rgb(31, 35, 40);">模型选择的方法为：</font>

1. <font style="color:rgb(31, 35, 40);">使用训练集训练出10个模型</font>
2. <font style="color:rgb(31, 35, 40);">用10个模型分别对交叉验证集计算得出</font>**<font style="color:rgb(31, 35, 40);">交叉验证误差</font>**<font style="color:rgb(31, 35, 40);">（代价函数的值）</font>
3. <font style="color:rgb(31, 35, 40);">选取代价函数值最小的模型</font>
4. <font style="color:rgb(31, 35, 40);">用步骤3中选出的模型对测试集计算得出推广误差（代价函数的值）</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745825148154-4184415d-cbdd-4cce-91ed-86aeb2f20726.png)

---

<h2 id="EyHMb">诊断偏差（bias）和方差（variance）</h2>
<font style="color:rgb(31, 35, 40);">当运行一个学习算法时，如果这个算法的表现不理想，那么多半是出现两种情况：要么是偏差比较大，要么是方差比较大。换句话说，出现的情况要么是欠拟合，要么是过拟合问题。</font>

<font style="color:rgb(31, 35, 40);">我们通常会通过将训练集和交叉验证集的代价函数误差与多项式的次数绘制在同一张图表上来帮助分析：</font>

![图10.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745825821093-f463ce70-22df-4a40-a715-cfabcec2a9d6.png)

<font style="color:rgb(31, 35, 40);">对于训练集，当 d 较小时，模型拟合程度更低，误差较大；随着 d 的增长，拟合程度提高，误差减小。</font>

<font style="color:rgb(31, 35, 40);">对于交叉验证集，当 d 较小时，模型拟合程度低，误差较大；但是随着 d 的增长，误差呈现先减小后增大的趋势，转折点是我们的模型开始</font>**<font style="color:rgb(31, 35, 40);">过拟合</font>**<font style="color:rgb(31, 35, 40);">训练数据集的时候。</font> 

---

<h2 id="fkOBr">正则化和偏差/方差</h2>
<font style="color:rgb(31, 35, 40);">在我们在训练模型的过程中，一般会使用一些正则化方法来防止过拟合。但是我们可能会面临正则化的程度太大或太小了，即我们在选择</font>$ \lambda $<font style="color:rgb(31, 35, 40);">的值时也需要思考与刚才选择多项式模型次数类似的问题。</font>

<font style="color:rgb(31, 35, 40);">我们选择一系列的想要测试的 λ 值，通常是 0-10之间的呈现2倍关系的值（如：</font>$ 0,0.01,0.02,0.04,0.08,0.15,0.32,0.64,1.28,2.56,5.12,10 $<font style="color:rgb(31, 35, 40);">共12个）。 我们同样把数据分为训练集、交叉验证集和测试集。</font>

![图10.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745828853453-674bf772-d122-4bb6-959e-841c10596c69.png)

<font style="color:rgb(31, 35, 40);">选择</font>$ \lambda $<font style="color:rgb(31, 35, 40);">的方法为：</font>

1. <font style="color:rgb(31, 35, 40);">使用训练集训练出12个不同程度正则化的模型</font>
2. <font style="color:rgb(31, 35, 40);">用12个模型分别对</font>**<font style="color:rgb(31, 35, 40);">交叉验证集</font>**<font style="color:rgb(31, 35, 40);">计算的出交叉验证误差</font>
3. <font style="color:rgb(31, 35, 40);">选择得出交叉验证误差</font>**<font style="color:rgb(31, 35, 40);">最小</font>**<font style="color:rgb(31, 35, 40);">的模型</font>
4. <font style="color:rgb(31, 35, 40);">运用步骤3中选出模型对测试集计算得出推广误差，我们也可以同时将训练集和交叉验证集模型的代价函数误差与</font>$ \lambda $<font style="color:rgb(31, 35, 40);">的值绘制在一张图表上：</font>

![图10.5](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745828996518-9b70fee1-5a1c-407f-a13a-19084f1e23e5.png)

注意训练集和交叉验证集的代价函数并没有包含正则化部分。

+ 当$ \lambda $较大时，受正则化部分的影响，训练集的代价函数将会非常大，此时由于$ \theta $微小的变动都会引起代价函数大幅度的变化，所以我们训练出来的$ \theta $往往与没有正则化的参数会有很大偏差，故此时交叉验证集的代价函数也很大。此时属于bias的情况
+ 当$ \lambda $较小时，相当于没有进行正则化，此时训练集的代价函数虽然比较小，但是由于假设函数高幂次部分没有受到约束，所以导致了“过拟合”的情况，交叉验证集的误差非常大。

---

<h2 id="UUxva">学习曲线</h2>
<font style="color:rgb(31, 35, 40);">学习曲线就是一种很好的工具，我经常使用学习曲线来判断某一个学习算法是否处于偏差、方差问题。学习曲线是将</font>**<font style="color:rgb(31, 35, 40);">训练集误差</font>**<font style="color:rgb(31, 35, 40);">和</font>**<font style="color:rgb(31, 35, 40);">交叉验证集误差</font>**<font style="color:rgb(31, 35, 40);">作为训练集</font>**<font style="color:rgb(31, 35, 40);">样本数量</font>**<font style="color:rgb(31, 35, 40);">（</font>$ m $<font style="color:rgb(31, 35, 40);">）的函数绘制的图表。</font>

![图10.6](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745851703865-531f8cd0-329d-4ff1-8f56-5d8b95756c6b.png)

训练集误差往往随着样本数量增大而增大，这是因为当样本数量较小时，模拟往往能很好地拟合数据，当$ m $逐渐增大时，假设误差产生的代价将逐渐累积。

而对于交叉验证集误差来说，情况恰好相反。这是因为使用的样本数量越多，模拟的**泛化**能力就越强。

![图10.7](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745851737038-b19332b6-07a9-4373-9adb-f886a96e5422.png)



对于High bias，也就是假设函数幂次不够高的情况（也被称为**欠拟合**），我们会有下图的情况：

![图10.8](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745852166333-6fd01335-36fd-4fee-b80c-4d144baaff73.png)

当样本数量足够大时，两条曲线将趋于水平，这是因为两个误差函数是**平均值**。由于模型维度不够，所以拟合的效果并不好，最终会拥有很大的误差值。

我们再来看看High variance的情况，也被称为**过拟合**：

![图10.9](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745852738073-17276387-0eb1-4122-9159-3f796b806253.png)

对于训练集误差，依然是随着$ m $的增大而增大，并且这个误差的值将**非常小**（过拟合的特性）。对于交叉验证集误差，由于模型过拟合，所以交叉验证集误差将一直保持**非常大**的水平。我们会发现两种误差之间拥有非常大的Gap，并且图10.9也可以大致看出：当我们增加$ m $时，两条曲线倾向于逐渐靠拢。所以在High variance情况中，增加样本数量是有效的。

---

<h2 id="Za8Ds">决定接下来做什么</h2>
<font style="color:rgb(31, 35, 40);">我们已经介绍了怎样评价一个学习算法，我们讨论了模型选择问题，偏差和方差的问题。那么这些诊断法则怎样帮助我们判断，哪些方法可能有助于改进学习算法的效果，而哪些可能是徒劳的呢？</font>

<font style="color:rgb(31, 35, 40);">让我们再次回到最开始的例子，在那里寻找答案，这就是我们之前的例子。回顾前面提出的六种可选的下一步，让我们来看一看我们在什么情况下应该怎样选择：</font>

![图10.10](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745853498534-9e509f27-3e6d-4a55-ba98-3dc08b33d1c5.png)

正则化参数$ \lambda $对应的是对于高维度参数的惩罚力度，所以增大$ \lambda $有助于处理过拟合的问题。

对于神经网络而言，如果使用过于简单的网络结构的话，很容易造成**欠拟合**的问题，所以我们通常选择使用更多的激活单元和更多的隐藏层，并且利用**正则化**来防止过拟合，尽管这会带来比较大的计算量。<font style="color:rgb(31, 35, 40);">通常选择较大的神经网络并采用正则化处理会比采用较小的神经网络效果要好。</font>

![图10.11](https://cdn.nlark.com/yuque/0/2025/png/40390053/1745853799126-b828982b-f0bf-4f13-94ec-ab84477b370d.png)

<font style="color:rgb(31, 35, 40);">关于神经网络中的隐藏层的层数以及各层激活单元数量的选择，通常从一层开始逐渐增加层数，为了更好地作选择，可以把数据分为训练集、交叉验证集和测试集，针对不同隐藏层层数的神经网络训练神经网络， 然后选择</font>**<font style="color:rgb(31, 35, 40);">交叉验证集代价最小</font>**<font style="color:rgb(31, 35, 40);">的神经网络。</font>

---

<h1 id="eqgQt">Chap11 机器学习系统的设计</h1>
<h2 id="VQ67B">误差分析</h2>
我们之前介绍了如何使用交叉验证集来对模型进行评估。为了提高模型学习效率及正确率，我们还可以使用**误差分析**，即：<font style="color:rgb(31, 35, 40);">人工检查交叉验证集中我们算法中产生预测误差的样本，看看这些样本是否有某种系统化的</font>**<font style="color:rgb(31, 35, 40);">趋势</font>**<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">以我们的垃圾邮件过滤器为例，误差分析要做的既是检验交叉验证集中我们的算法产生错误预测的所有邮件，看是否能将这些邮件按照类分组。例如医药品垃圾邮件，仿冒品垃圾邮件或者密码窃取邮件等。然后看分类器对哪一组邮件的预测误差最大，并着手优化。</font>

<font style="color:rgb(31, 35, 40);">思考怎样能改进分类器。例如，发现是否缺少某些特征，记下这些特征出现的次数。</font>

<font style="color:rgb(31, 35, 40);">总结一下，在研究一个新的机器学习问题时，可以实现一个较为简单快速、即便不是那么完美的算法。当有了初始的实现之后，它会变成一个非常有力的工具，来帮助你决定下一步的做法。因为我们可以先看看算法造成的错误，通过误差分析，来看看他犯了什么错，然后来决定优化的方式，从而你会更快地做出决定，在算法中放弃什么，吸收什么误差分析可以帮助我们系统化地选择该做什么。</font>

---

<h2 id="kLimR">偏斜类的误差度量</h2>
<font style="color:rgb(31, 35, 40);">假如我们希望用算法来预测癌症是否是恶性的，在我们的训练集中，只有0.5%的实例是恶性肿瘤。假设我们编写一个非学习而来的算法，在所有情况下都预测肿瘤是良性的，那么误差只有0.5%。然而我们通过训练而得到的神经网络算法却有1%的误差。</font>

<font style="color:rgb(31, 35, 40);">像这种某个分类的数量要远大于另一个分类的数量的情况，我们称为</font>**<font style="color:rgb(31, 35, 40);">偏斜类</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">skewed classes</font>**<font style="color:rgb(31, 35, 40);">）。这时，误差的大小是不能视为评判算法效果的依据的。</font>

我们引入新的评估标准：**查准率**（**precision**）和**召回率**（**recall**）。

假设$ y=1 $表示我们要预测的比较稀少的分类，那么我们可以将情况分为四种：

|  | | 实际值 |
| --- | --- | :---: |
| | | 1 | 0 |
| 预测值 | 1 | True Positive | False Positive |
| | 0 | False Negative | True Negative |


那么查准率为：$ \frac{TP}{TP + FP} $，即在所有预测为正样本的数据中实际是正样本的比例；

召回率为：$ \frac{TP}{TP + FN} $，即在所有实际为正样本的数据中，被预测（探测）为正样本的比例。

<font style="color:rgb(31, 35, 40);">这样，对于我们刚才那个总是预测病人肿瘤为良性的算法，其查全率是0。通过这两个指标我们就可以较好地衡量一个偏斜类的模型效果了。</font>

---

<h2 id="i5yPA">查准率和召回率的权衡</h2>
<font style="color:rgb(31, 35, 40);">查准率和召回率作为遇到偏斜类问题的评估度量值。在很多应用中，我们希望能够保证查准率和召回率的相对平衡。</font>

<font style="color:rgb(31, 35, 40);">如果我们希望只在非常确信的情况下预测为真（肿瘤为恶性），即我们希望更高的查准率，我们可以使用比0.5更大的阀值，如0.7，0.9。这样做我们会减少错误预测病人为恶性肿瘤的情况，同时却会增加未能成功预测肿瘤为恶性的情况。</font>

<font style="color:rgb(31, 35, 40);">如果我们希望提高召回率，尽可能地让所有有可能是恶性肿瘤的病人都得到进一步地检查、诊断，我们可以使用比0.5更小的阀值，如0.3。</font>

<font style="color:rgb(31, 35, 40);">我们可以将不同阀值情况下，查全率与查准率的关系绘制成图表，曲线的形状根据数据的不同而不同。</font>

![图11.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746500956962-de7ea97e-3fd0-474e-9286-2be7be822642.png)

那么是否存在一种度量可供我们评估不同的查准率和召回率呢？

我们可以使用$ F_1 $Score。$ F_1 = 2\frac{PR}{P + R} $，我们可以看到无论是查准率还是召回率比较低时，$ F_1 $Score都会很低，只有两者都比较大时，$ F_1 $才会较大。这也是经常被作为权衡查准率和召回率的一个度量（还有其他方法）。

---

<h1 id="jh0Na">Chap12 支持向量机（Support Vector Machines）</h1>
<h2 id="mfHoi">优化目标</h2>
<font style="color:rgb(31, 35, 40);">正如我们之前学习的机器学习算法，我们从优化目标开始。为了描述支持向量机，接下来将会从逻辑回归开始展示我们如何一点一点修改来得到本质上的支持向量机。</font>

<font style="color:rgb(31, 35, 40);">如下图，我们可以将传统逻辑回归的代价函数修改为图中粉红色的</font>**<font style="color:rgb(31, 35, 40);">折线</font>**<font style="color:rgb(31, 35, 40);">。</font>

![图12.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746605809626-95ce6f05-67ae-40b9-8186-78da73f78c31.png)

我们为上述两条折现分别命名$ cost_1 $和$ cost_0 $，表示对应于$ y=1 $或$ y=0 $的代价函数。

那么我们可以将目标/代价函数修改如下：

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746606039829-b14b81bf-0af2-4293-9a25-c193edb57019.png)

注意，在SVM中，我们相当于从逻辑回归的整体中乘上了一个$ \frac{1}{\lambda} $，使得由$ A + \lambda B $变为了：$ CA + B $的形式。

<font style="color:rgb(31, 35, 40);">如果给</font>$ \lambda $<font style="color:rgb(31, 35, 40);">一个非常大的值，意味着给予</font>$ B $<font style="color:rgb(31, 35, 40);">更大的权重。而这里，就对应于将</font>$ C $<font style="color:rgb(31, 35, 40);">设定为非常小的值，那么，相应的将会给</font>$ B $<font style="color:rgb(31, 35, 40);">比给</font>$ A $<font style="color:rgb(31, 35, 40);">更大的权重。因此，这只是一种不同的方式来控制这种权衡或者一种不同的方法，即用参数来决定是更关心第一项的优化，还是更关心第二项的优化。</font>



<font style="color:rgb(31, 35, 40);">最后有别于逻辑回归的是：逻辑回归输出的是概率；在这里，我们的代价函数，当最小化代价函数，获得参数</font>$ {{\theta }} $<font style="color:rgb(31, 35, 40);">时，支持向量机所做的是它来直接预测</font>$ y $<font style="color:rgb(31, 35, 40);">的值等于1，还是等于0。因此，当</font>$ \theta^Tx $<font style="color:rgb(31, 35, 40);">大于或者等于0时，这个假设函数会预测1。否则会预测0。</font>

---

<h2 id="XRJPC">大边界的直观理解</h2>
<font style="color:rgb(31, 35, 40);">有时将支持向量机看作是大间距分类器（Large Margin Classifier）。</font>

如下图，是支持向量机的代价函数，我们希望当$ y=1 $时，有$ \theta^T x \geq 1 $，反之同理。<font style="color:rgb(31, 35, 40);">这就相当于在支持向量机中嵌入了一个额外的安全因子，或者说安全的间距因子。</font>

![图12.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746607150128-b58316e8-5a9f-4b10-9f78-1540148a4ed7.png)

假设我们的C取得非常大，那么在最小化代价函数的时候，我们期望找到第一项尽可能小的最优解。

具体地，我们可以观察这样一个数据集，<font style="color:rgb(31, 35, 40);">可以看到这个数据集是线性可分的。存在一条直线把正负样本分开。</font>

![图12.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746607398715-1d112b2b-2b8b-4689-8483-93c87862875a.png)

<font style="color:rgb(31, 35, 40);">当然，有多条不同的直线，可以把正样本和负样本完全分开：</font>

![图12.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746607468673-69f347d5-ee5c-4d2a-abfd-aa0ec203a06f.png)

<font style="color:rgb(31, 35, 40);">这些决策边界看起来都不是特别好的选择，支持向量机将会选择这个</font>**<font style="color:rgb(31, 35, 40);">黑色</font>**<font style="color:rgb(31, 35, 40);">的决策边界，相较于另外用粉色或者绿色画的决策界。这条黑色的看起来好得多，黑线看起来是更稳健的决策界。在分离正样本和负样本上它显得的更好。数学上来讲，这是什么意思呢？这条黑线有更大的距离，这个距离叫做</font>**<font style="color:rgb(31, 35, 40);">间距</font>**<font style="color:rgb(31, 35, 40);">(</font>**<font style="color:rgb(31, 35, 40);">margin</font>**<font style="color:rgb(31, 35, 40);">)。我们看到黑色的决策界和训练样本之间有更大的最短距离。然而其他两条分界线离训练样本就非常近。</font>

<font style="color:rgb(31, 35, 40);">因此支持向量机有时被称为</font>**<font style="color:rgb(31, 35, 40);">大间距分类器</font>**<font style="color:rgb(31, 35, 40);">，而这其实是求解上一页幻灯片上优化问题的结果。</font>

ok，这只是$ C $很大（可以看成$ \lambda $很小）的情形，<font style="color:rgb(31, 35, 40);">实际上当</font>$ C $<font style="color:rgb(31, 35, 40);">不是非常非常大的时候，它可以忽略掉一些异常点的影响，得到更好的决策界（下图黑线）。甚至当数据不是线性可分的时候，支持向量机也可以给出好的结果。</font>

 ![图12.5](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746607822852-e9e7f863-6d7e-4e4c-a035-eb31e8288139.png)

<font style="color:rgb(31, 35, 40);">也可以把</font>$ C $<font style="color:rgb(31, 35, 40);">看成</font>$ \frac{1}{\lambda} $<font style="color:rgb(31, 35, 40);">，因此：</font>

<font style="color:rgb(31, 35, 40);">C 较大时，相当于</font>$ \lambda $<font style="color:rgb(31, 35, 40);">较小，可能会导致过拟合，高方差。</font>

<font style="color:rgb(31, 35, 40);">C 较小时，相当于</font>$ \lambda $<font style="color:rgb(31, 35, 40);">较大，可能会导致低拟合，高偏差。</font>

---

<h2 id="XO53V">大边界分类背后的数学原理</h2>
根据之前的分析，我们可以得出以下结论：

![图12.6](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746625943578-61e96da1-03ff-42d8-8aba-1cc11aefff67.png)

由于$ \sum\theta_j^2 \Lrarr \Big( \sqrt{\sum\theta_j^2} \Big)^2 \Lrarr \Big( {|| \vec{\theta }||} \Big)^2 $，故正则化的操作就等价于最小化参数向量的**模长**。

并且$ \theta^T x^{(i)} \Lrarr || \vec{\theta} || p^{(i)} $，$ p^{(i)} $是$ x^{(i)} $在参数向量上的投影长度，那么为了最小化代价函数，就必须在参数向量的模长尽可能小的前提下，让正负样本在参数向量上的投影尽可能大。这就是**最大化样本到决策边界的最小距离**，决策边界就是垂直于参数向量的直线（投影为零）。

如下图，像这样的绿色的决策边界显然就不够优秀：

![图12.7](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746626653586-97e3fd00-2f73-4ab5-ab76-2a563791acad.png)

如果使用SVM的话，我们会得到如图12.8的大边界分类的一个分界线，这样一来各样本在参数向量上的投影都尽可能大。

![图12.8](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746626738846-ece98aa7-f2d4-47a8-9d0e-988c05e3f619.png)

这便是SVM最大化各样本到决策边界的最短距离背后的数学原理。

---

<h2 id="bgzvJ">核函数</h2>
假设我们有一个非线性的分类问题：

![图12.9](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746628135843-f5fb9db0-4ead-44ac-8400-e0acdb26acda.png)

<font style="color:rgb(31, 35, 40);">为了获得上图所示的判定边界，我们的模型可能是</font>$ {{\theta }_{0}}+{{\theta }_{1}}{{x}_{1}}+{{\theta }_{2}}{{x}_{2}}+{{\theta }_{3}}{{x}_{1}}{{x}_{2}}+{{\theta }_{4}}x_{1}^{2}+{{\theta }_{5}}x_{2}^{2}+\cdots $<font style="color:rgb(31, 35, 40);">的形式。</font>

<font style="color:rgb(31, 35, 40);">我们可以用一系列的新的特征</font>$ f $<font style="color:rgb(31, 35, 40);">来替换模型中的每一项。例如令： </font>$ {{f}_{1}}={{x}_{1}},{{f}_{2}}={{x}_{2}},{{f}_{3}}={{x}_{1}}{{x}_{2}},{{f}_{4}}=x_{1}^{2},{{f}_{5}}=x_{2}^{2} $

<font style="color:rgb(31, 35, 40);">除了对原有的特征进行组合以外，有没有更好的方法来构造</font>$ f_1,f_2,f_3 $<font style="color:rgb(31, 35, 40);">？我们可以利用</font>**<font style="color:rgb(31, 35, 40);">核函数</font>**<font style="color:rgb(31, 35, 40);">来计算出新的特征。</font>

<font style="color:rgb(31, 35, 40);">给定一个训练样本</font>$ x $<font style="color:rgb(31, 35, 40);">，我们利用</font>$ x $<font style="color:rgb(31, 35, 40);">的各个特征与我们预先选定的</font>**<font style="color:rgb(31, 35, 40);">地标</font>**<font style="color:rgb(31, 35, 40);">(</font>**<font style="color:rgb(31, 35, 40);">landmarks</font>**<font style="color:rgb(31, 35, 40);">)</font>$ l^{(1)},l^{(2)},l^{(3)} $<font style="color:rgb(31, 35, 40);">的近似程度来选取新的特征</font>$ f_1,f_2,f_3 $<font style="color:rgb(31, 35, 40);">。</font>

![图12.10](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746628461128-0b2ca7ac-6e35-4506-a869-e7ae76f14f93.png)

<font style="color:rgb(31, 35, 40);">例如</font><font style="color:rgb(31, 35, 40);">：</font>$ {{f}_{1}}=similarity(x,{{l}^{(1)}})=e^{(-\frac{{{\left| x-{{l}^{(1)}} \right|}^{2}}}{2{{\sigma }^{2}}})} $<font style="color:rgb(31, 35, 40);">,</font><font style="color:rgb(31, 35, 40);">其中：</font>$ {{\left| x-{{l}^{(1)}} \right|}^{2}}=\sum_{j=1}^{n}{{({{x}_{j}}-l_{j}^{(1)})}^{2}} $<font style="color:rgb(31, 35, 40);">，为实例</font>$ x $<font style="color:rgb(31, 35, 40);">中所有特征与地标</font>$ l^{(1)} $<font style="color:rgb(31, 35, 40);">之间的距离的和。上例中的</font>$ similarity(x,{{l}^{(1)}}) $<font style="color:rgb(31, 35, 40);">就是</font>**<font style="color:rgb(31, 35, 40);">核函数</font>**<font style="color:rgb(31, 35, 40);">，具体而言，这里是一个</font>**<font style="color:rgb(31, 35, 40);">高斯核函数</font>**<font style="color:rgb(31, 35, 40);">(Gaussian Kernel)。 注：这个函数与正态分布没什么实际上的关系，只是看上去像而已。</font>

<font style="color:rgb(31, 35, 40);">这些地标的作用是什么？如果一个训练样本</font>$ x $<font style="color:rgb(31, 35, 40);">与地标</font>$ l $<font style="color:rgb(31, 35, 40);">之间的距离近似于0，则新特征 f近似于</font>$ e^{-0}=1 $<font style="color:rgb(31, 35, 40);">，如果训练样本</font>$ x $<font style="color:rgb(31, 35, 40);">与地标</font>$ l $<font style="color:rgb(31, 35, 40);">之间距离较远，则</font>$ f $<font style="color:rgb(31, 35, 40);">近似于</font>$ e^{-\infin}=0 $<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">假设我们的训练样本含有两个特征</font>$ [x_{1},x_2] $<font style="color:rgb(31, 35, 40);">，给定地标</font>$ l^{(1)} $<font style="color:rgb(31, 35, 40);">与不同的</font>$ \sigma $<font style="color:rgb(31, 35, 40);">值（</font>$ \sigma $<font style="color:rgb(31, 35, 40);">就是核函数的参数），见下图</font>

![图12.11](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746628826934-070c8da5-1e12-4756-8667-2513701bcb66.png)

<font style="color:rgb(31, 35, 40);">图中水平面的坐标为</font>$ x_1 $<font style="color:rgb(31, 35, 40);">，</font>$ x_{2} $<font style="color:rgb(31, 35, 40);">而垂直坐标轴代表</font>$ f $<font style="color:rgb(31, 35, 40);">。可以看出，只有当</font>$ x $<font style="color:rgb(31, 35, 40);">与</font>$ l^{(1)} $<font style="color:rgb(31, 35, 40);">重合时</font>$ f $<font style="color:rgb(31, 35, 40);">才具有最大值。随着</font>$ x $<font style="color:rgb(31, 35, 40);">的改变</font>$ f $<font style="color:rgb(31, 35, 40);">值改变的速率受到</font>$ \sigma^2 $<font style="color:rgb(31, 35, 40);">的控制。</font>

根据这个性质，假如我们的模型训练得到了如下的$ \theta $值，那么我们根据三个地标点可以大致得出下图红色的决策边界：

![图12.12](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746629091696-1023e44f-21b9-4a31-8e45-30542e2339e7.png)

<font style="color:rgb(31, 35, 40);">在预测时，我们采用的特征不是训练样本本身的特征，而是通过核函数计算出的新特征</font>$ f_1,f_2,f_3 $<font style="color:rgb(31, 35, 40);">。</font>



那么对于给定的数据集X，假设有$ m $个样本，那么我们可以将每一个样本看成一个地标，那么就有$ m $个地标，<font style="color:rgb(31, 35, 40);">令:</font>$ l^{(1)}=x^{(1)},l^{(2)}=x^{(2)},.....,l^{(m)}=x^{(m)} $<font style="color:rgb(31, 35, 40);">。这样做的好处在于：现在我们得到的新特征是建立在原有特征与训练集中所有其他特征之间距离的基础之上的，即：</font>

$ f_i = \begin{bmatrix} 
f_0^{(i)} = 1 \\
f_1^{(i)} = sim(x^{(i)}, l^{(1)}) \\
f_2^{(i)} = sim(x^{(i)}, l^{(2)}) \\
\vdots \\
f_i^{(i)} = sim(x^{(i)}, l^{(i)}) = e^{0} = 1 \\
\vdots \\
f_m^{(i)} = sim(x^{(i)}, l^{(m)}) \\
\end{bmatrix} $

<font style="color:rgb(31, 35, 40);">下面我们将核函数运用到支持向量机中，修改我们的支持向量机假设为：</font>

+ <font style="color:rgb(31, 35, 40);">给定</font>$ x $<font style="color:rgb(31, 35, 40);">，计算新特征</font>$ f $<font style="color:rgb(31, 35, 40);">，当</font>$ \theta^T f \geq 0 $<font style="color:rgb(31, 35, 40);"> 时，预测</font>$ y=1 $<font style="color:rgb(31, 35, 40);">，否则反之。</font>

<font style="color:rgb(31, 35, 40);">相应地修改代价函数为：</font>$ \min_{\theta} C\sum\limits_{i=1}^{m}{[{{y}^{(i)}}cost_{1}}( {{\theta }^{T}}{{f}^{(i)}})+(1-{{y}^{(i)}})cost_{0}( {{\theta }^{T}}{{f}^{(i)}})]+\frac{1}{2}\sum\limits_{j=1}^{m}{\theta {j}^{2}} $<font style="color:rgb(31, 35, 40);">，其中</font>$ \sum_{j=1}^{m}\theta _{j}^{2}={{\theta}^{T}}\theta = \big( ||\vec{\theta}|| \big)^2 $

<font style="color:rgb(31, 35, 40);">在具体实施过程中，我们还需要对最后的正则化项进行些微调整，在计算</font>$ \sum_{j=1}^{m}\theta _{j}^{2}={{\theta}^{T}}\theta $<font style="color:rgb(31, 35, 40);">时，我们用</font>$ \theta^T M \theta $<font style="color:rgb(31, 35, 40);">代替</font>$ \theta^T \theta $<font style="color:rgb(31, 35, 40);">，其中</font>$ M $<font style="color:rgb(31, 35, 40);">是根据我们选择的核函数而不同的一个矩阵。这样做的原因是为了简化计算。</font>

<font style="color:rgb(31, 35, 40);">理论上讲，我们也可以在逻辑回归中使用核函数，但是计算将非常耗费时间。</font>

<font style="color:rgb(31, 35, 40);">在此，我们不介绍最小化支持向量机的代价函数的方法，可以使用现有的软件包（如</font>**<font style="color:rgb(31, 35, 40);">liblinear</font>**<font style="color:rgb(31, 35, 40);">,</font>**<font style="color:rgb(31, 35, 40);">libsvm</font>**<font style="color:rgb(31, 35, 40);">等）。在使用这些软件包最小化我们的代价函数之前，我们通常需要编写核函数，并且如果我们使用高斯核函数，那么在使用之前进行</font>**<font style="color:rgb(31, 35, 40);">特征缩放</font>**<font style="color:rgb(31, 35, 40);">是非常必要的。</font>



<font style="color:rgb(31, 35, 40);">最后，是SVM的两个参数</font>$ C(=\frac{1}{\lambda}) $<font style="color:rgb(31, 35, 40);">和</font>$ \sigma $<font style="color:rgb(31, 35, 40);">的影响：</font>

+ $ C $<font style="color:rgb(31, 35, 40);">较大时，我们的主要目标是减小代价函数的第一项，导致模型在训练数据上表现得过好，相当于</font>$ \lambda $<font style="color:rgb(31, 35, 40);">较小，可能会导致过拟合，高方差；</font>
+ $ C $<font style="color:rgb(31, 35, 40);">较小时，主要目标就变成了减小代价函数的第二项，导致模型没有训练充分，相当于</font>$ \lambda $<font style="color:rgb(31, 35, 40);">较大，可能会导致低拟合，高偏差；</font>
+ $ \sigma $<font style="color:rgb(31, 35, 40);">较大时，相似函数的变化将非常平和（训练不充分），可能会导致低方差，高偏差，欠拟合；</font>
+ $ \sigma $<font style="color:rgb(31, 35, 40);">较小时，相似函数的变化将非常剧烈（训练过度），可能会导致低偏差，高方差，过拟合。</font>

![图12.13](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746694063169-e248287d-cac4-4ac6-a5dd-e17dd0ee8d5e.png)

---

<h1 id="NQyUS">Chap13 聚类（Clustering）</h1>
<h2 id="pylNQ">K-Means 算法</h2>
K-均值算法是无监督学习中最普及的聚类算法，<font style="color:rgb(31, 35, 40);">算法接受一个未标记的数据集，然后将数据聚类成不同的组。</font>

![图13.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746846935893-d68216f0-9b56-437f-9b9a-abb53bb69334.png)

<font style="color:rgb(31, 35, 40);">K-均值是一个</font>**<font style="color:rgb(31, 35, 40);">迭代</font>**<font style="color:rgb(31, 35, 40);">算法，假设我们想要将数据聚类成n个组，其方法为:</font>

+ <font style="color:rgb(31, 35, 40);">首先选择</font>$ K $<font style="color:rgb(31, 35, 40);">个随机的点，称为</font>**<font style="color:rgb(31, 35, 40);">聚类中心</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">cluster centroids</font>**<font style="color:rgb(31, 35, 40);">）；</font>
+ <font style="color:rgb(31, 35, 40);">对于数据集中的每一个数据，按照距离</font>$ K $<font style="color:rgb(31, 35, 40);">个中心点的距离，将其与距离</font>**<font style="color:rgb(31, 35, 40);">最近</font>**<font style="color:rgb(31, 35, 40);">的中心点关联起来，与同一个中心点关联的所有点聚成一类。</font>
+ <font style="color:rgb(31, 35, 40);">计算每一个类所有点的中心点，将该组所关联的聚类中心移动到平均值的位置。</font>

![图13.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746847052961-461260ad-1996-4b59-bfac-0237e36406f6.png)

<font style="color:rgb(31, 35, 40);">用</font>$ μ_1,μ_2,...,μ_K $<font style="color:rgb(31, 35, 40);">来表示聚类中心，用</font>$ c^{(1)},c^{(2)},...,c^{(m)} $<font style="color:rgb(31, 35, 40);">来存储与第</font>$ i $<font style="color:rgb(31, 35, 40);">个实例数据最近的聚类中心的索引，</font>**<font style="color:rgb(31, 35, 40);">K-均值</font>**<font style="color:rgb(31, 35, 40);">算法的伪代码如下：</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746847163295-e2bd19c9-c785-4019-886d-072767009912.png)

需要注意的是，有时候可能存在某个聚簇中心不包含任何的样本点，那么我们可以**移除**这个聚簇中心，或者重新进行**随机初始化**$ K $个聚簇中心**。**

---

<h2 id="rvJ6c">优化目标</h2>
<font style="color:rgb(31, 35, 40);">K-均值最小化问题，是要最小化所有的数据点与其所关联的聚类中心点之间的距离之和，因此 K-均值的代价函数（又称</font>**<font style="color:rgb(31, 35, 40);">畸变函数</font>**<font style="color:rgb(31, 35, 40);"> </font>**<font style="color:rgb(31, 35, 40);">Distortion function</font>**<font style="color:rgb(31, 35, 40);">）为：</font>

<font style="color:rgb(31, 35, 40);"></font>$ J(c^{(1)},...,c^{(m)},μ_1,...,μ_K)=\dfrac {1}{m}\sum^{m}_{i=1}\left| X^{\left( i\right) }-\mu_{c^{(i)}}\right| ^{2} $

<font style="color:rgb(31, 35, 40);">其中</font>$ {{\mu }_{{{c}^{(i)}}}} $<font style="color:rgb(31, 35, 40);">代表与</font>$ {{x}^{(i)}} $<font style="color:rgb(31, 35, 40);">最近的聚类中心点。 我们的的优化目标便是找出使得代价函数最小的</font>$ c^{(1)},c^{(2)}, \cdots, c^{(m)} $<font style="color:rgb(31, 35, 40);">和</font>$ μ^1,μ^2,...,μ^k $<font style="color:rgb(31, 35, 40);">： </font>![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8605f0826623078a156d30a7782dfc3c.png)

<font style="color:rgb(31, 35, 40);">回顾刚才给出的: </font>**<font style="color:rgb(31, 35, 40);">K-均值</font>**<font style="color:rgb(31, 35, 40);">迭代算法，我们知道，第一个循环是用于减小</font>$ c^{(i)} $<font style="color:rgb(31, 35, 40);">引起的代价，而第二个循环则是用于减小</font>$ {{\mu }_{i}} $<font style="color:rgb(31, 35, 40);">引起的代价。迭代的过程一定会是每一次迭代都在减小代价函数，不然便是出现了错误。</font>

---

<h2 id="KBgWj">随机初始化</h2>
<font style="color:rgb(31, 35, 40);">在运行K-均值算法的之前，我们首先要随机初始化所有的聚类中心点，下面介绍怎样做：</font>

1. <font style="color:rgb(31, 35, 40);">我们应该选择</font>$ K<m $<font style="color:rgb(31, 35, 40);">，即聚类中心点的个数要小于所有训练集实例的数量</font>
2. <font style="color:rgb(31, 35, 40);">随机选择</font>$ K $<font style="color:rgb(31, 35, 40);">个训练样本实例，然后令</font>$ K $<font style="color:rgb(31, 35, 40);">个聚类中心分别与这</font>$ K $<font style="color:rgb(31, 35, 40);">个训练实例相等</font>

<font style="color:rgb(31, 35, 40);">K-均值的一个问题在于，它有可能会停留在一个</font>**<font style="color:rgb(31, 35, 40);">局部最小值处</font>**<font style="color:rgb(31, 35, 40);">（</font>**<font style="color:rgb(31, 35, 40);">local optima</font>**<font style="color:rgb(31, 35, 40);">），而这取决于初始化的情况。</font>

![图13.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746848661559-c1cea369-2231-4346-9370-9cd7b1f1afe8.png)

<font style="color:rgb(31, 35, 40);">为了解决这个问题，我们通常需要</font>**<font style="color:rgb(31, 35, 40);">多次</font>**<font style="color:rgb(31, 35, 40);">运行K-均值算法，每一次都重新进行随机初始化，最后再比较多次运行K-均值的结果，选择代价函数最小的结果。这种方法在</font>$ K $<font style="color:rgb(31, 35, 40);">较小的时候还是可行的，但是如果</font>$ K $<font style="color:rgb(31, 35, 40);">较大，这么做也可能不会有明显地改善。</font>

---

<h2 id="LjSNd">选取聚类数量</h2>
<font style="color:rgb(31, 35, 40);">事实上，没有所谓最好的选择聚类数的方法，通常是需要根据不同的问题，</font>**<font style="color:rgb(31, 35, 40);">人工</font>**<font style="color:rgb(31, 35, 40);">进行选择的。选择的时候思考我们运用K-均值算法聚类的动机是什么，然后选择能最好服务于该目标的聚类数。</font>

<font style="color:rgb(31, 35, 40);">当人们在讨论，选择聚类数目的方法时，有一个可能会谈及的方法叫作“肘部法则”。关于“肘部法则”，我们所需要做的是改变</font>$ K $<font style="color:rgb(31, 35, 40);">值，也就是聚类类别数目的总数，然后计算成本函数</font>$ J $

![图13.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746865624214-b2509d43-8d3e-4ed4-97fe-ecb39b8805de.png)

<font style="color:rgb(31, 35, 40);">我们可能会得到一条类似于左图这样的曲线。像一个人的肘部。这就是“肘部法则”所做的，你会发现这种模式，它的畸变值会迅速下降，从1到2，从2到3之后，你会在3的时候达到一个肘点。在此之后，畸变值就下降的非常慢，看起来就像使用3个聚类来进行聚类是正确的，这是因为那个点是曲线的肘点，畸变值下降得很快，</font>$ K=3 $<font style="color:rgb(31, 35, 40);">之后就下降得很慢，那么我们就选</font>$ K=3 $<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">此外，当你应用“肘部法则”的时候，如果你得到了一个像上面右边的图，那么可能需要考虑别的方法来选择</font>$ K $<font style="color:rgb(31, 35, 40);">值。</font>

---

<h1 id="i6FsR">Chap14 降维（Dimensionality Reduction）</h1>
在这一节我们将学习第二个无监督学习算法，<font style="color:rgb(31, 35, 40);">，</font><font style="color:rgb(31, 35, 40);">称为</font>**<font style="color:rgb(31, 35, 40);">降维</font>**<font style="color:rgb(31, 35, 40);">。有几个不同的的原因使我们可能想要做降维。其一就是数据压缩，数据压缩不仅允许我们压缩数据，因而使用较少的计算机内存或磁盘空间，但它也让我们加快我们的学习算法。</font>

<h2 id="RCs9e">动机1：数据压缩</h2>
<font style="color:rgb(31, 35, 40);">假设我们未知两个的特征：</font>$ x_1 $<font style="color:rgb(31, 35, 40);">:长度（厘米）；</font>$ x_2 $<font style="color:rgb(31, 35, 40);">（英寸表示同一物体的长度）。</font>

![图14.1](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746866821455-ac6e31b6-e1d2-4afc-9714-1fd8993aebc7.png)

<font style="color:rgb(31, 35, 40);">这是两个冗余的特征，我们可以将数据从二维降至一维，从而只保留它们在直线上的一个特征</font>$ z $<font style="color:rgb(31, 35, 40);">。</font>

<font style="color:rgb(31, 35, 40);">类似地，也存在从三维压到二维的例子：</font>

![图14.2](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746867045311-99edd0a0-8bd5-41d0-b9b0-5a0f839dd0fa.png)

我们可以发现，三维空间中的点大致落在某个平面上，所以我们可以利用它们在平面上的投影$ (z_1, z_2) $来表示原先的三个特征。

---

<h2 id="egm0W">动机2：数据可视化</h2>
降维还有一个好处就是，方便我们将高维的特征转为低维的特征进行可视化。

<font style="color:rgb(31, 35, 40);">假使我们有有关于许多不同国家的数据，每一个特征向量都有50个特征（如GDP，人均GDP，平均寿命等）。如果要将这个50维的数据可视化是不可能的。使用降维的方法将其降至2维，我们便可以将其可视化了。</font>

![图14.3](https://cdn.nlark.com/yuque/0/2025/png/40390053/1746867645028-1afddb39-1e92-4c6c-9ee1-f929482785c9.png)

<font style="color:rgb(31, 35, 40);">这样做的问题在于，降维的算法只负责减少维数，新产生的特征的意义就必须由我们自己去发现了。</font>

---

<h2 id="MejtZ">PCA问题规划</h2>
**<font style="color:rgb(31, 35, 40);">主成分分析</font>**<font style="color:rgb(31, 35, 40);">(Principal Component Analysis)是最常见的降维算法。</font>

<font style="color:rgb(31, 35, 40);">在2D压到1D的PCA中，我们要做的是找到一条直线，当我们把所有的数据都投射到该直线上时，我们希望投射平均均方误差能尽可能地小。</font>

![图14.4](https://cdn.nlark.com/yuque/0/2025/png/40390053/1747014213176-4718cbc3-2c74-4dd7-ac0b-b8da8700e773.png)

<font style="color:rgb(31, 35, 40);">下面给出主成分分析问题的描述：</font>

<font style="color:rgb(31, 35, 40);">如果是要将</font>$ n $<font style="color:rgb(31, 35, 40);">维数据降至</font>$ k $<font style="color:rgb(31, 35, 40);">维，目标是找到向量</font>$ u^{(1)},u^{(2)},...,u^{(k)} $<font style="color:rgb(31, 35, 40);">使得总的投射误差最小。</font>

![图14.5](https://cdn.nlark.com/yuque/0/2025/png/40390053/1747014335833-16f31a40-6060-4b4c-ad45-f78591b8a747.png)

<font style="color:rgb(31, 35, 40);">虽然PCA看起来与线性回归很类似，但它们不是一个东西。</font>

<font style="color:rgb(31, 35, 40);">主成分分析与线性回归是两种不同的算法。主成分分析最小化的是</font>**<font style="color:rgb(31, 35, 40);">投射误差</font>**<font style="color:rgb(31, 35, 40);">（Projected Error），而线性回归尝试的是最小化</font>**<font style="color:rgb(31, 35, 40);">预测误差</font>**<font style="color:rgb(31, 35, 40);">。线性回归的目的是预测结果，而主成分分析不作任何预测。</font>

![图14.6](https://cdn.nlark.com/yuque/0/2025/png/40390053/1747014414852-ad6cff51-4f41-42a8-85cb-d5a5d284647f.png)

<font style="color:rgb(31, 35, 40);">上图中，左边的是线性回归的误差（垂直于横轴投影），右边则是主要成分分析的误差（垂直于红线投影）。</font>

---

<h2 id="ZhOEH">PCA算法</h2>
接下来我们介绍PCA的具体步骤。

1. 首先，在进行PCA之前，我们需要对数据进行**特征缩放**/**均值标准化**（feature scaling/mean normalization）：

假设我们有训练集：$ x^{(1)}, x^{(2)},\cdots, x^{(m)} $，那么我们计算均值$ \mu_j = \frac{1}{m} \sum_{i=1}^{m} x_j^{(i)} $

并将每个样本$ x^{(i)} $替换为$ x^{(i)} - \mu $。

<font style="color:rgb(31, 35, 40);">如果特征是在不同的数量级上，我们还需要将其除以标准差</font>$ \sigma ^2 $<font style="color:rgb(31, 35, 40);">。</font>



2. 第二步是计算**协方差**矩阵（covariance matrix）$ \Sigma $：

$ \Sigma = \frac{1}{m} \sum_{i=1}^{m} (x^{(i)})(x^{(i)})^T $，我们知道$ x^{(i)} \in \mathbb{R}^n $（$ n $是原始特征数），最后协方差矩阵$ \Sigma \in \mathbb{R}^{n \times n} $

实际上如果我们的数据集是以下面的形式存储的（每一行是一个样本，$ X \in \mathbb{R}^{m \times n} $	）：

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1747016698403-b4bfba97-0ec5-4d61-98d5-686a30eb474d.png)

那么我们的协方差矩阵还可以写作：$ \Sigma = \frac{1}{m} (X^T X) $



3. 第三步是对协方差矩阵进行**奇异值分解**（SVD）：`<font style="color:rgb(31, 35, 40);">[U, S, V]= svd(sigma)</font>`

<font style="color:rgb(31, 35, 40);">对于一个</font>$ n \times n $<font style="color:rgb(31, 35, 40);">维度的矩阵，上式中的</font>$ U $<font style="color:rgb(31, 35, 40);">是一个具有与数据之间最小投射误差的方向向量构成的矩阵。如果我们希望将数据从</font>$ n $<font style="color:rgb(31, 35, 40);">维降至</font>$ k $<font style="color:rgb(31, 35, 40);">维，我们只需要从</font>$ U $<font style="color:rgb(31, 35, 40);">中选取</font>**<font style="color:rgb(31, 35, 40);">前</font>**$ k $**<font style="color:rgb(31, 35, 40);">个向量</font>**<font style="color:rgb(31, 35, 40);">，获得一个</font>$ n×k $<font style="color:rgb(31, 35, 40);">维度的矩阵，我们用</font>$ U_{reduce} $<font style="color:rgb(31, 35, 40);">表示，然后通过如下计算获得要求的新特征向量</font>$ z^{(i)}=U_{reduce}^T x^{(i)} $

这里得到的矩阵$ U $应该是按奇异值（类似于特征向量重要程度）递减，所以选取前$ k $个就可以得到最重要的$ k $个向量。

---

<h2 id="UyTLy">主成分数量的选择</h2>
首先我们引入两个概念：

+ 平均投影方差：$ \frac{1}{m} \sum_{i=1}^{m} || x^{(i)} - x_{approx}^{(i)} || ^ 2 $
+ <font style="color:rgb(31, 35, 40);">训练集方差：</font>$ \dfrac {1}{m}\sum^{m}_{i=1}|| x^{\left( i\right) }|| ^{2} $

<font style="color:rgb(31, 35, 40);">我们希望在平均均方误差与训练集方差的比例尽可能小的情况下选择</font>**<font style="color:rgb(31, 35, 40);">尽可能小</font>**<font style="color:rgb(31, 35, 40);">的</font>$ k $<font style="color:rgb(31, 35, 40);">值。</font>

<font style="color:rgb(31, 35, 40);">如果我们希望这个比例小于1%，就意味着原本数据的偏差有99%都保留下来了，便能非常显著地降低模型中特征的维度了。</font>

![](https://cdn.nlark.com/yuque/0/2025/png/40390053/1747019097912-88ae214c-aea5-4404-9476-819754bd7e11.png)

<font style="color:rgb(31, 35, 40);">我们可以先令</font>$ k=1 $<font style="color:rgb(31, 35, 40);">，然后进行主成分分析，获得</font>$ U_{reduce} $<font style="color:rgb(31, 35, 40);">和</font>$ z $<font style="color:rgb(31, 35, 40);">，然后计算比例是否小于1%。如果不是的话再令</font>$ k=2 $<font style="color:rgb(31, 35, 40);">，如此类推，直到找到可以使得比例小于1%的最小</font>$ k $<font style="color:rgb(31, 35, 40);">值。</font>

事实上，<font style="color:rgb(31, 35, 40);">还有一些更好的方式来选择</font>$ k $<font style="color:rgb(31, 35, 40);">，在进行SVD分解的时候：</font>`<font style="color:rgb(31, 35, 40);">[U, S, V] = svd(sigma)</font>`<font style="color:rgb(31, 35, 40);">。矩阵</font>$ S \in \mathbb{R}^{n \times n} $<font style="color:rgb(31, 35, 40);">是一个</font>**<font style="color:rgb(31, 35, 40);">对角矩阵</font>**<font style="color:rgb(31, 35, 40);">，并且其奇异值从大到小排列，所以有：</font>$ \dfrac {\dfrac {1}{m} \sum_{i=1}^{m}\left| x^{\left( i\right) }-x^{\left( i\right) }_{approx}\right| ^{2}}{\dfrac {1}{m}\sum^{m}_{i=1}\left| x^{(i)}\right| ^{2}}=1-\dfrac {\sum^{k}_{i=1}S_{ii}}{\sum^{n}_{i=1}S_{ii}}\leq 0.01 $

  
 <font style="color:rgb(31, 35, 40);">也就是：</font>$ \frac {\sum^{k}_{i=1}S_{ii}}{\sum^{n}_{i=1}S_{ii}}\geq0.99 $

<font style="color:rgb(31, 35, 40);">这样一来，我们只需要进行</font>**<font style="color:rgb(31, 35, 40);">一次SVD</font>**<font style="color:rgb(31, 35, 40);">，然后在</font>$ S $<font style="color:rgb(31, 35, 40);">上枚举</font>$ k $<font style="color:rgb(31, 35, 40);">即可。</font>

---

<h2 id="FrNtr">压缩重现</h2>
根据SVD的性质，我们可以知道$ U $是**正交**矩阵，所以$ U^{-1} = U^T $，这是因为$ U^TU = I $

所以<font style="color:rgb(31, 35, 40);">在压缩过数据后，我们可以采用如下方法来</font>**<font style="color:rgb(31, 35, 40);">近似</font>**<font style="color:rgb(31, 35, 40);">地获得原有的特征：</font>$ x^{\left( i\right) }_{approx}=U_{reduce}z^{(i)} $

---

<h2 id="pbWQf">应用PCA的建议</h2>
通常我们在面对一个监督学习的问题时，可以在**训练集**上进行PCA降维，得出各个最优参数之后，然后对交叉验证集和测试集应用最优的PCA参数和机器学习参数。

<font style="color:rgb(31, 35, 40);">一个常见错误使用PCA的情况是，将其用于减少过拟合（减少了特征的数量）。这样做非常不好，不如尝试正则化处理。原因在于主要成分分析只是近似地丢弃掉一些特征，它并不考虑任何与结果变量有关的信息，因此可能会丢失非常重要的特征。然而当我们进行正则化处理时，会考虑到结果变量，不会丢掉重要的数据。</font>

---










































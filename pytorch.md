
## 张量 Tensor
~~~
import torch  
  
x = torch.randn(3,2)  
  
print(f"内容：\n{x}")  
print(f"形状:{x.shape}")  
print(f"设备：{x.device}")  
print(f"类型：{x.dtype}")
~~~
~~~
内容：
tensor([[ 0.9321,  0.0209],
        [-1.0357, -0.1104],
        [ 0.6874, -0.3536]])
形状:torch.Size([3, 2])
设备：cpu
类型：torch.float32
~~~
#### tensor创建
~~~
1.全 0 tensor ：torch.zeros(3,3)
2.全 1 tensor ：torch.ones(3,3)
3.单位矩阵 tensor ：torch.eye(2,3) 这里可以输入两个不一样的维度
4.随机数 tensor ：torch.randn(3,3)
5.给定一个整数 n，生成一个从 0 到 n-1 的随机打乱的整数序列 : torch.randperm(n)
6.指定序列：torch.arange(start, end, step)其中start默认为0，step默认为1
~~~
##### 用法辨析
##### **1.随机函数**
~~~
torch.rand(n)/ 生成0~1之间的n个随机浮点数
torch.randn(n)/ 符合标准正态分布的n个随机数
torch.randint(low, high, size)/随机整数，torch.randint(1, 12, (3,2))
torch.randperm(n)/0 ~ n-1 的不重复随机排列
~~~
##### 2.torch.tensor()  VS  torch.tensor([])
**torch.Tensor(2, 3)**：创建的是一个 形状为 2 行 3 列，但内容未初始化（随机脏数据）的 32 位浮点型（float32）张量。大写Tensor默认为浮点数。
**torch.Tensor([2,3])** ：传入的是一个tensor，为tensor([2., 3.])
**torch.tensor([2,3])**：传入的是一个tensor，为tensor([2, 3])

#### torch维度变换
##### **1.shape 或者 size() 查看维度**
两个函数输出**结果一致**
~~~
x = torch.arange(11)
print(x.size())

输出：
torch.Size([11])
~~~
##### 2.view() 或 reshape() 改变形状
view要求原始张量在内存里是连续存储的，所以容易报错，但是速度快。reshape更加智能，不会报错。
~~~
x = torch.arange(12)  
y = x.reshape(3,4)  
print(x.shape)  
print(y.shape)

输出：
torch.Size([12])
torch.Size([3, 4])
~~~
其中，如果变形后的数据总数和变形前**对应不上**，会报错。所以我们如果不想手动计算维度的值，我们可以**用 -1 替代**
~~~
x = torch.arange(12)  
y = x.reshape(3,-1)  
z = x.reshape(2,-1)  
print(x.shape)  
print(y.shape)  
print(z.shape)

输出：
torch.Size([12])
torch.Size([3, 4])
torch.Size([2, 6])
~~~
##### 3.增加或减少维度 unsqueeze() , squeeze()
###### unsqueeze() 增加维度
`unsqueeze(dim)` 会在指定的维度位置插入一个长度为 **1** 的新维度。
~~~
x = torch.arange(12)  
y = x.unsqueeze(0)  
print(x.shape)  
print(y.shape)  
print(f"y的内容：{y}")

输出：
torch.Size([12])
torch.Size([1, 12])
y的内容：tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
~~~
###### squeeze() 降维
`squeeze(dim)` 会移除该长度为 **1** 的维度。如果该维度长度不为 1，则不会发生任何变化。当（）里没有维度时，**默认删除所有维度为1的维度**
~~~
x = torch.randn(3,4)  
y = x.unsqueeze(1)  
z = y.squeeze(1)  
u = y.squeeze(0)  
print(x.shape)  
print(y.shape)  
print(z.shape)  
print(u.shape)

输出：
torch.Size([3, 4])
torch.Size([3, 1, 4])
torch.Size([3, 4])
torch.Size([3, 1, 4])
~~~

###### 维度想象
~~~
tensor([[[ 0.6084, -1.1171, -0.2943,  0.3075]],

        [[-0.5362,  1.2430,  0.6177,  0.5902]],

        [[-1.0201, -1.2249,  0.9602, -0.5440]]])
~~~
最外层（dim=0）：最外层的一个[ ]包裹了三个块，所以dim0=3
次外层（dim=1）：次外层里只有一个块，所以dim1=1
最内层（dim=2）：最内层有四个数字，所以dim2=4

#### tensor计算
##### 加减乘除 add，sub，mul，div
其中mul是**逐元素乘法**，不是矩阵乘法，且加减乘除可以用**对应的符号**来同等使用。
~~~
import torch  
  
a = torch.tensor([1,2,3])  
b = torch.tensor([3,4,5])  
  
c = a + b  
d = torch.add(a,b)  
  
print(c)  
print(d)

输出：
tensor([4, 6, 8])
tensor([4, 6, 8])
~~~
**注意辨析：** add 和 add_
- **`add`**: 计算后返回一个**新**的 Tensor，原数据不变。
- **`add_`**: 直接修改**原** Tensor 的值，不占用额外内存。
~~~
import torch  
  
x = torch.tensor([1])  
  
# 非原地操作  
y = x.add(5)  
print(x) # x 还是 1
print(y) # y 是 6  

# 原地操作 (带下划线)  
x.add_(5)  
print(x) # x 变成了 6，原数据被修改了
~~~

##### 线性代数运算
###### 矩阵乘法 matmul 符号可以用 @
~~~
import torch  
  
a = torch.tensor([[1,2],[3,4]])  
b = torch.tensor([[5,6],[7,8]])  
  
c = torch.matmul(a,b)  
d = a @ b  
  
print(c)  
print(d)

输出：
tensor([[19, 22],
        [43, 50]])
tensor([[19, 22],
        [43, 50]])
~~~
###### 求逆 inverse
矩阵必须是方阵，且pytorch的矩阵求逆运算不支持整数类型

~~~
torch.inverse(a)
~~~
##### 比较运算
###### 大于，小于，等于，返回的是一个 **布尔型**
gt（>）：大于
lt （<）：小于
eq（ == ）：等于
~~~
import torch  
  
x = torch.tensor([1, 2, 3])  
y = torch.tensor([2, 2, 2])  
  
print(torch.gt(x, y))  # 或者 x > y
print(x==y)

输出：
tensor([False, False,  True])
tensor([False,  True, False])
~~~
###### 在一个 Tensor 里找到前 $k$ 个最大的值，topk
它会返回**两个**结果：
**Values**: 这 $k$ 个最大的数具体是多少。 
**Indices**: 这 $k$ 个数原本在什么位置（索引）。
~~~
probs = torch.tensor([0.1, 0.7, 0.2, 0.9, 0.3])  
values, indices = torch.topk(probs, k=2)  
  
print(values)  # 最大的两个值  
print(indices)  # 它们所在的索引位置

输出：
tensor([0.9000, 0.7000])
tensor([3, 1])
~~~

##### 索引，切片，拼接

###### 切片
切片的核心逻辑是：**`tensor[第0维, 第1维, 第2维, ...]`**。 
每一维的写法和 Python 的 `list` 完全一致：`start:end:step`（左闭右开）。

**常用符号：**
- **`:`** ：代表选取该维度的**所有**元素。
- **`int`**：代表选取该维度的**某一个**索引，会减少一个维度。
- **`start:end`**：代表选取从 `start` 到 `end-1` 的元素。
- **`...`** (省略号)：代表跨越中间所有的维度，全都选中。

~~~
 x = torch.tensor([[1, 2, 3, 4],  
                  [5, 6, 7, 8],  
                  [9, 10, 11, 12]])  
  
print(x[0,:]) #tensor([1, 2, 3, 4])  
print(x[0,0:2]) #tensor([1, 2])  
print(x[1:3,2]) #tensor([ 7, 11])

假设张量 x 的形状是 (2, 3, 4, 5)
x[0, ...]：等价于 x[0, :, :, :]
~~~
###### 拼接 cat 和 stack
**cat：** 在现有维度连接，除了要拼接的那一个维度外，其他维度的形状（Shape）必须完全一致。
~~~
a = torch.ones(2, 3) # 2行3列  
b = torch.zeros(2, 3) # 2行3列  
  
# 在第0维（行）拼接：变成 4行3列  
c = torch.cat([a, b], dim=0)  
  
# 在第1维（列）拼接：变成 2行6列  
d = torch.cat([a, b], dim=1)  
  
print(c)  
print(d)

输出：
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [0., 0., 0.],
        [0., 0., 0.]])
tensor([[1., 1., 1., 0., 0., 0.],
        [1., 1., 1., 0., 0., 0.]])
~~~
**stack：** 在新维度堆叠，参与拼接的张量形状必须**一模一样**

###### 拆分 split 和 chunk
**split：** 按**长度**拆分，两个用法，**固定长度**，或者按**列表**每个元素的长度精确控制。
~~~
a = torch.tensor([1,2,3,4,5,6,7,8,9])  
  
b,c,d = torch.split(a,3)  
  
e,f,g = torch.split(a,[3,2,4])  
  
print(b)  
print(c)  
print(d)  
print(e)  
print(f)  
print(g)

输出：
tensor([1, 2, 3])
tensor([4, 5, 6])
tensor([7, 8, 9])
tensor([1, 2, 3])
tensor([4, 5])
tensor([6, 7, 8, 9])
~~~
**chunk：** 按数量拆分，把tensor拆成想要的几分
~~~
a = torch.tensor([1,2,3,4,5,6,7,8,9])  
  
chunks = torch.chunk(a,2)  
  
print(chunks)

输出：
(tensor([1, 2, 3, 4, 5]), tensor([6, 7, 8, 9]))

~~~
###### 高级索引 gather 和 index_select
**index_select 按维度取值**
普通的切片只能取连续范围（如 `0:5`），而 `index_select` 可以根据一个**索引列表**，把不连续的行或列“挑出来”。
~~~
x = torch.randn(5, 3)  # 5行3列  
indices = torch.tensor([0, 2, 4])  # 我们想要第0, 2, 4行  
  
# dim=0 代表按行挑  
res = torch.index_select(x, dim=0, index=indices)  
# 结果形状: (3, 3)  
  
print(x)  
print(res)

输出：
tensor([[ 0.1230, -1.3948, -1.3542],
        [-0.6384,  0.2434,  1.0503],
        [ 0.6380,  1.1085,  1.9507],
        [-0.8891, -0.1073, -0.1175],
        [-1.5072,  1.0312, -0.2697]])
tensor([[ 0.1230, -1.3948, -1.3542],
        [ 0.6380,  1.1085,  1.9507],
        [-1.5072,  1.0312, -0.2697]])
~~~
**gather 查表式取值**
根据提供的索引矩阵，在对应位置点对点地取数据
- **`dim=1` (横向找)**：行号固定，去每一行里按索引**左右**挑。
    
- **`dim=0` (纵向找)**：列号固定，去每一列里按索引**上下**挑。
~~~
import torch

# 1. 原始数据 (Input): 3行2列
# 代表 3 个人，每人有两门课(课0, 课1)的成绩
scores = torch.tensor([[10, 20],  # 学生0
                       [30, 40],  # 学生1
                       [50, 60]]) # 学生2

# 2. 索引 (Index): 我们想要获取的成绩位置
# 注意：index 的维度必须和 scores 一致（都是 2D）
# 假设我们想取：学生0的课1，学生1的课0，学生2的课1
index = torch.tensor([[1], 
                      [0], 
                      [1]])

# 3. 执行 gather
# dim=1 表示我们在“列”这个维度上进行挑选
result = torch.gather(scores, dim=1, index=index)

print("提取后的成绩：")
print(result)
# 输出结果：
# tensor([[20],   <-- 学生0的第1门课
#         [30],   <-- 学生1的第0门课
#         [60]])  <-- 学生2的第1门课
~~~
##### 统计与规约
###### 基础统计 mean sum median std var
- **`mean`**: 算平均值。
- **`sum`**: 求和。
- **`median`**: 找中位数。
- **`std` / `var`**: 算标准差和方差。
**指定 `dim` 的效果：**
- `x.sum(dim=0)`：**纵向压缩**。把所有的“行”压在一起，最后剩下一行。
- `x.sum(dim=1)`：**横向压缩**。把所有的“列”压在一起，最后剩下一列。
~~~
import torch
x = torch.tensor([[1., 2., 3.], 
                  [4., 5., 6.]])

print(x.mean())        # 全部平均: 3.5
print(x.mean(dim=0))   # 纵向平均: tensor([2.5, 3.5, 4.5])
print(x.sum(dim=1))    # 横向求和: tensor([6., 15.])

输出：
tensor(3.5000)
tensor([2.5000, 3.5000, 4.5000])
tensor([ 6., 15.])
~~~
###### 极值查询 max min argmax argmin
- **`max` / `min`**: 找最大值/最小值。
- **`argmax` / `argmin`**: 找最大值/最小值**所在的索引（位置）**。
~~~
probs = torch.tensor([0.1, 0.7, 0.2])  
  
print(probs.max())    # 结果: 0.7 (最大值)  
print(probs.argmax()) # 结果: 1 (最大值所在的下标)

输出：
tensor(0.7000)
tensor(1)
~~~
#### tensor 自动求导
##### 属性设置 requires_grad = True
在 PyTorch 中，并不是所有的张量（Tensor）都需要计算梯度。这是一个布尔值标志。当你创建一个 Tensor 并设置 `requires_grad=True` 时，PyTorch 就会开始记录这个张量的计算过程。PyTorch 会构建一个**计算图（Computational Graph）**。所有基于这个张量进行的数学运算（加减乘除、卷积等）都会被记录下来，以便后续计算导数。
##### 反向传播 backward()
自动计算梯度。会根据之前记录下的计算图，从最后的结果（通常是损失值 Loss）开始，利用**链式法则**一路往回推，算出每一个 `requires_grad=True` 的变量对结果的贡献（即**导数**）。计算出来的梯度不会作为一个返回值给到你，而是直接存进了每个变量自己的 **`.grad`** 属性里。
在普通的训练循环里，如果你不手动把上一次的**梯度清零**，模型更新就会乱套。所以每次计算新梯度前，必须调用 **`optimizer.zero_grad()` 或者 `x.grad.zero_()`.**
~~~
# 这里的 x 就是我们要优化的参数（比如权重）  
x = torch.tensor([2.0, 3.0], requires_grad=True)  
  
# y 是基于 x 计算出来的  
y = x**2 + 5  
  
# 接上面的例子  
z = y.mean()  # z 是一个标量（求平均值）  
z.backward()  # 开始反向传播  
  
# 查看结果  
print(x.grad)  # 此时 x.grad 里就存好了 dz/dx 的值

输出：
tensor([2., 3.])
~~~
###### 计算过程
**第一阶段：前向传播 (Forward Pass)**
这是数据从 $x$ 流向 $z$ 的过程：
1. **输入 $x$**:$$x = [x_1, x_2] = [2.0, 3.0]$$
2. **计算 $y$**:$$y_1 = x_1^2 + 5 = 2.0^2 + 5 = 9.0$$$$y_2 = x_2^2 + 5 = 3.0^2 + 5 = 14.0$$所以 $y = [9.0, 14.0]$。
3. **计算 $z$ (标量化)**:
    $z$ 是 $y$ 的平均值：$$z = \frac{y_1 + y_2}{2} = \frac{9.0 + 14.0}{2} = 11.5$$
**第二阶段：反向传播 (Backward Pass)**
当调用 `z.backward()` 时，PyTorch 从 $z$ 开始倒着求导，目标是求 $\frac{\partial z}{\partial x}$。
根据链式法则：

$$\frac{\partial z}{\partial x} = \frac{\partial z}{\partial y} \cdot \frac{\partial y}{\partial x}$$
**先看 $\frac{\partial z}{\partial y}$ (z 对 y 的偏导):**
因为 $z = \frac{1}{2}(y_1 + y_2)$，所以：

- $\frac{\partial z}{\partial y_1} = \frac{1}{2}$
    
- $\frac{\partial z}{\partial y_2} = \frac{1}{2}$
**再看 $\frac{\partial y}{\partial x}$ (y 对 x 的偏导):**
因为 $y = x^2 + 5$，根据幂函数求导法则 ($x^n$ 的导数是 $nx^{n-1}$):

- $\frac{\partial y_1}{\partial x_1} = 2x_1 = 2 \times 2.0 = 4.0$
    
- $\frac{\partial y_2}{\partial x_2} = 2x_2 = 2 \times 3.0 = 6.0$
**最后相乘得到 $\frac{\partial z}{\partial x}$:**
- **对于 $x_1$**:$$\frac{\partial z}{\partial x_1} = \frac{\partial z}{\partial y_1} \cdot \frac{\partial y_1}{\partial x_1} = \frac{1}{2} \times 4.0 = 2.0$$
- **对于 $x_2$**:$$\frac{\partial z}{\partial x_2} = \frac{\partial z}{\partial y_2} \cdot \frac{\partial y_2}{\partial x_2} = \frac{1}{2} \times 6.0 = 3.0$$
##### 上下文控制 with torch.no_gard()
这是一个上下文管理器。在这个代码块包裹的范围内，PyTorch 会停止所有梯度的跟踪和计算图的构建。
可以**节省内存，加速计算，防止验证或测试的时候意外修改模型梯度**
~~~
# 训练完了，想看看模型效果
model.eval() # 切换到评估模式

with torch.no_grad():
    # 这里的计算不会产生任何梯度信息
    prediction = model(test_data)
    loss = criterion(prediction, target)
    # 你在这里调用 loss.backward() 会直接报错，因为没有计算图
~~~

#### 模型建构
##### 全连接层 nn.Linear()
它把输入的每一个数字都乘以一个“权重”（Weight），加在一起后再加一个“偏置”（Bias），最后输出。$y = xA^T + b$
**参数含义**：
- `in_features`: 输入的维度。
- `out_features`: 输出的维度。
~~~
import torch.nn as nn

# 创建一个输入 3 维，输出 1 维的全连接层
layer = nn.Linear(3, 1)

# 假设输入数据 x
x = torch.randn(1, 3) 
output = layer(x)
~~~
###### 计算过程
**内部结构**

当执行 `layer = nn.Linear(3, 1)` 时，PyTorch 会自动初始化了两个参数：
- **权重 (Weight)**: 一个形状为 $(1, 3)$ 的矩阵。我们记作 $W$。 
- **偏置 (Bias)**: 一个形状为 $(1,)$ 的向量（标量）。我们记作 $b$。

这两个参数**默认**都是 `requires_grad=True` 的，它们就是模型要学习的参数。

**数学公式**

`nn.Linear` 执行的操作本质上就是：
$$y = x \cdot W^T + b$$
_(注：PyTorch 为了计算方便，内部对权重矩阵进行了转置操作)_

**详细计算过程演示**

我们假设一下具体的数值（实际代码中 `randn` 是随机生成的）：
- **输入 $x$**: `[1.0, 2.0, 3.0]` (形状为 $1 \times 3$)
- **权重 $W$**: `[[0.1, 0.2, 0.3]]` (形状为 $1 \times 3$)
- **偏置 $b$**: `[0.5]`
**第一步：点积运算** 
将输入的每个特征与对应的权重相乘并求和：

$$\text{Sum} = (1.0 \times 0.1) + (2.0 \times 0.2) + (3.0 \times 0.3)$$

$$\text{Sum} = 0.1 + 0.4 + 0.9 = 1.4$$

**第二步：加上偏置 (Add Bias)**
将上一步的结果加上偏置项：
$$\text{Output} = 1.4 + 0.5 = 1.9$$
所以，最终 `print(output)` 得到的就是 `tensor([[1.9]])`。

##### 二维卷积层 nn.Convd
使用**卷积核**在图片上滑行。提取其中的特征。不管特征出现在图片的左上角还是右下角，同一个卷积核都能识别出来，这**大大减少了需要训练的参数量**。此外**它能保持图片像素之间的位置关系**

**参数含义：**
- `in_channels`: 输入的通道数（比如彩色图是 3，黑白图是 1）
- `out_channels`: 输出的通道数（想提取出多少种特征） 
- `kernel_size`: 卷积核的大小
~~~
import torch.nn as nn

# 假设输入是一个 3 通道的彩色图
# 我们用 16 个 3x3 的滤镜去扫描它
conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
~~~
###### 计算过程

**核心参数拆解**
当定义 `nn.Conv2d(3, 16, kernel_size=3)` 时，实际上创造了一个拥有以下属性的卷积层：
- **`in_channels=3`**：输入有 3 层（通常是 RGB 红绿蓝）。
- **`out_channels=16`**：我们会得到 16 张新的特征图（Feature Maps）。
- **`kernel_size=3`**：每个卷积核的大小是 $3 \times 3$。

**内部隐藏的“权重量”**
如何计算这个层里到底有多少个权重参数？
**每个输出通道**其实都对应一个**卷积核组**。
- 一个卷积核组包含 **3 个**（对应输入通道数RGB）大小为 **$3 \times 3$** 的矩阵。
- 再加上 1 个偏置项（Bias）。
- 总参数量 = $16 \times (3 \times 3 \times 3) + 16 = 448$ 个权重。

**详细计算步骤（以得到 16 个输出中的“第一层”为例）**

**第一步：空间卷积（相乘再相加）**
1. **R 通道**：卷积核的第一个矩阵与图像的 R 通道左上角 $3 \times 3$ 区域对应元素相乘并求和。
2. **G 通道**：卷积核的第二个矩阵与图像的 G 通道同一位置做同样操作。
3. **B 通道**：卷积核的第三个矩阵与图像的 B 通道同一位置做同样操作。

**第二步：跨通道融合**
将这三个通道算出来的数值**加在一起**，再加上该层特有的**偏置项 (Bias)**。
最终，这 27 个位置的乘法结果汇聚成了**一个数字**。这就是输出特征图中左上角的第一个像素。

**第三步：移动（Stride）**
卷积核向右移动一个像素（默认步长 Stride=1），重复上述过程。直到扫完整个图像，就生成了**第一张**完整的输出特征图（$out\_channel = 1$）。

**第四步：并行处理**
剩下的 15 个输出通道，会由另外 15 组不同的卷积核（它们拥有不同的权重，检测不同的特征）**并行**完成。

**关键点总结**
- **输入形状**：`[Batch, 3, Height, Width]`
- **输出形状**：`[Batch, 16, New_Height, New_Width]`
- **特征提取**：每一个 `out_channel` 都在尝试从原图中提取某种特定的特征。第 1 个通道可能在找“横边”，第 2 个可能在找“圆圈”。

##### 激活函数 nn.ReLu
没有激活函数，深层网络就和单层网络没区别，所以要引入非线性变化的激活函数。其他激活函数也同理。
$$ReLU(x) = \max(0, x)$$

~~~
import torch.nn as nn  
import torch  
  
relu = nn.ReLU()  
input_data = torch.tensor([-1.0, 2.0, -0.5, 3.0])  
output = relu(input_data)  
 
print(output)
tensor([0., 2., 0., 3.])
输出：
tensor([0., 2., 0., 3.])
~~~

##### 归一化 nn.BatchNorm
在每一层运算之后，将输出的数据拉回到一个**均值为 0，方差为 1** 的标准分布。可以防止梯度爆炸或消失，让训练更快，对模型权重的初始化要求降低。
~~~
import torch
import torch.nn as nn

# 1. 声明 BatchNorm 层（参数 1 对应输入张量的通道数）  
bn = nn.BatchNorm2d(1)  
  
# 2. 模拟一个卷积层的输出  
# 维度含义：[Batch_Size, Channels, Height, Width]  
input_tensor = torch.randn(1, 1, 8, 8)  
  
# 3. 真正执行归一化  
output = bn(input_tensor)  
print(output.mean())

输出：
tensor(0., grad_fn=<MeanBackward0>)
~~~

##### 按顺序组合层 nn.Sequential
将一系列的层（Layer）按顺序包装起来，构成一个model，如果不使用Sequential，我们需要手动在forward函数里写清楚数据如何从一层流向下一层。
~~~
import torch
import torch.nn as nn

# 定义一个简单的三层结构
model = nn.Sequential(
    nn.Linear(784, 256), # 第一层
    nn.ReLU(),           # 激活
    nn.Linear(256, 10)   # 输出层
)

# 只需要调用一次 model(x)，数据就会自动跑完这三层
~~~

##### 模型基类 nn.Module
它是一个 Python 类。定义的任何神经网络模型，都必须继承与他。
- Module可以**自动追踪参数**，把 `nn.Linear` 或 `nn.Conv2d` 这样的层定义在 `__init__` 中时，`nn.Module` 会自动识别这些层里的权重（Weights）和偏置（Bias）。这样只需要调用 `model.parameters()`，就能把模型中成千上万个参数一次性传给优化器。
- **状态管理**，可以通过 `model.train()` 和 `model.eval()` 切换训练/评估模式。
当定义自己的网络时，必须遵循这个模板：
~~~
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__() # 必须执行！父类里有很多初始化注册逻辑
        self.conv = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32 * 26 * 26, 10)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = x.view(x.size(0), -1) # 展平
        x = self.fc(x)
        return x
~~~
###### 损失函数 
- **回归任务：** 常用 `nn.MSELoss` (均方误差) 或 `nn.L1Loss`(MAE 平均绝对误差)。
    
- **分类任务：** 常用 `nn.CrossEntropyLoss`（交叉熵）。
    
    - **注意：** PyTorch 的 `CrossEntropyLoss` 内部集成了 **Softmax** 运算。因此，模型的最后一层**不应**添加 Softmax 激活函数，否则会导致重复计算。
~~~
import torch
import torch.nn as nn

# 1. 实例化官方损失函数
mse_criterion = nn.MSELoss()
ce_criterion = nn.CrossEntropyLoss()

# 2. 准备模拟数据
predict = torch.randn(3, 5, requires_grad=True)
target_reg = torch.randn(3, 5)        # 回归目标
target_cls = torch.empty(3, dtype=torch.long).random_(5) # 分类目标（类别索引）

# 3. 计算损失
loss_mse = mse_criterion(predict, target_reg)
loss_ce = ce_criterion(predict, target_cls)

print(f"MSE Loss: {loss_mse.item()}")
print(f"CrossEntropy Loss: {loss_ce.item()}")
~~~

**自定义的损失函数**需要继承 `torch.nn.Module`。其输入通常为 `(input, target)`，输出为一个标量（默认对 Batch 取平均）。
~~~
import torch
import torch.nn as nn

class SimpleMSELoss(nn.Module):
    def __init__(self):
        super(SimpleMSELoss, self).__init__()

    def forward(self, input, target):
        # 计算平方差
        squared_diff = (input - target) ** 2
        # 对整个 Batch 取平均值并返回标量
        return torch.mean(squared_diff)

# 验证代码
criterion = SimpleMSELoss()
prediction = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)

loss = criterion(prediction, target)
print(f"Loss: {loss.item()}")
~~~

#### 数据处理
##### 数据集 Dataset
`Dataset` 是 PyTorch 处理数据的核心抽象类。它的本质是一个**索引映射器**，负责将原始数据（如磁盘上的文件）封装成模型可读取的格式。它并不一次性将所有数据载入内存，而是在模型需要时，按需读取。
需要重写以下三个方法：
**`__init__` 初始化** 

**`__len__` 数据总量**

- **功能**：返回数据集的总样本数 $N$。
- **作用**：让后续的采样器（Sampler）知道索引的具体范围（$0$ 到 $N-1$）。

**`__getitem__` 获取样本**

- **输入**：一个整数索引 `index`。
- **操作流程**：
    
    1. **读取**：根据索引从磁盘加载对应的数据。
        
    2. **变换 (Transform)**：将原始、杂乱的输入数据，通过一系列预处理步骤，转化为模型能够接收的标准化张量（Tensor）格式。
    3. **输出**：返回处理后的 Tensor 及其对应的 Label。
~~~
from torchvision import transforms

my_transform = transforms.Compose([
    transforms.Resize((256, 256)),      # 1. 缩放
    transforms.RandomCrop(224),         # 2. 随机裁剪到 224
    transforms.ToTensor(),              # 3. 转为张量
    transforms.Normalize(mean, std)     # 4. 标准化
])
~~~
        
~~~
from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # 1. 实时从磁盘读取
        item = load_from_disk(self.file_paths[index])
        label = self.labels[index]
        
        # 2. 应用预处理
        if self.transform:
            item = self.transform(item)
            
        # 3. 返回转换后的 Tensor
        return item, torch.tensor(label)
~~~

##### DataLoader 数据流控制器
###### 批处理
`DataLoader` 会调用 `Dataset` 的 `__getitem__` $n$ 次（$n$ 为 `batch_size`），然后将这 $n$ 个独立的 Tensor 拼接（Stack）成一个更高维度的 Tensor。例如，单张图片维度为 $(3, 224, 224)$，`batch_size=32` 后，输出维度变为 $(32, 3, 224, 224)$

###### 多线程并行加载
- **参数**：`num_workers`。
- **原理**：由于 `__getitem__` 中包含磁盘 I/O 和 CPU 预处理，单线程会导致 GPU 在等待数据时处于闲置状态。
- **操作**：通过开启多个子进程，`DataLoader` 可以在 GPU 计算当前 Batch 的同时，**提前在后台准备好后续 Batch**。

###### 数据洗牌
- **参数**：`shuffle=True`。
- **重要性**：在每个 Epoch 开始时**打乱数据索引**。这能防止模型学习到数据存放的顺序规律，提高模型的泛化能力。

###### 丢弃最后的不完整批次
- **参数**：`drop_last=True`。
- **场景**：如果数据集总量 $100$ 条，`batch_size=32`，最后会剩 $4$ 条。在分布式训练或使用 BatchNorm 层时，不完整的批次可能导致统计量失真，通常选择将其丢弃。

~~~
from torch.utils.data import DataLoader

# 实例化 DataLoader
train_loader = DataLoader(
    dataset=my_dataset,      # 刚才定义的 Dataset 实例
    batch_size=64,           # 每一批处理 64 个样本
    shuffle=True,            # 每一轮训练开始前打乱顺序
    num_workers=4,           # 开启 4 个子进程并行加载数据
    pin_memory=True,         # 将数据锁在内存中，加速向 GPU 拷贝（工业级常用）
    drop_last=True           # 丢弃最后不够一批的数据
)

# 在训练循环中使用
for images, labels in train_loader:
    # 此时 images 是形状为 [64, C, H, W] 的 Tensor
    outputs = model(images)
    ...
~~~

#### GPU管理
##### 设备指定与检测
PyTorch 不会自动将模型搬运至 GPU。需通过 `torch.cuda.is_available()` 确认环境，并定义 `device` 对象。
~~~
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
~~~
##### 数据与模型迁移至GPU
使用 **`.to(device)`** 或 **`.cuda()`** 方法。
注意：**Tensor 的迁移是非原地操作**（需重新赋值），而 **Module（模型）的迁移是原地操作**。
~~~
Tensor: x = x.to(device)

Module: model.to(device)
~~~
##### 显存管理机制
PyTorch 使用 **缓存分配器 (Caching Allocator)** 以提高内存分配效率。其中，显存占用类型分为**已分配显存**和**缓存显存**。缓存显存被缓存池用来备用。
`torch.cuda.empty_cache()` 可以用来释放缓存显存，但是没办法解决显存碎片化导致的`Out of Memory OOM`
`Out of Memory OOM` 一般是因为batch size过大或模型计算图未释放导致。


#### 训练循环
训练循环的核心由 **数据加载、前向传播、计算损失、反向传播** 以及 **参数更新** 五个关键步骤组成。
~~~
# 初始化优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    for input, target in dataloader:
        optimizer.zero_grad()      # 1. 清空梯度
        output = model(input)      # 2. 前向传播
        loss = criterion(output, target) 
        loss.backward()            # 3. 反向传播计算梯度
        optimizer.step()           # 4. 更新权重
    
    scheduler.step()               # 5. 更新学习率 (通常以 Epoch 为单位)
~~~
##### 优化器的实例化与通用配置
实例化时，必须将模型的可学习参数（通常是 `model.parameters()`）传给优化器
~~~
# 1. SGD: 随机梯度下降，常配合 momentum 克服局部最优
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 2. Adam: 自适应矩估计，收敛快，初学者首选
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
~~~
###### 权重衰减 (Weight Decay)
它在数学上等同于 **L2 正则化**。
- **物理意义**：在更新梯度时，额外减去一部分权值。公式简写为：$w_{new} = w_{old} - \eta \cdot \nabla w - \lambda \cdot w_{old}$，其中 $\lambda$ 就是权重衰减系数。
- **目的**：**防止模型过拟合**。它会让权重不至于过大，从而使模型曲线更平滑，提高泛化能力。
- **配置**：在实例化优化器时通过 `weight_decay` 传入，通常取值很小（如 `1e-5` 或 `1e-4`）。
###### 参数组 (Parameter Groups)
默认情况下，优化器对传入的所有参数执行**统一**的学习率和策略。**参数组**允许将参数拆分，实施**差异化管理**。
- **应用场景**：
    - **微调（Fine-tuning）**：给预训练好的模型底层（特征提取层）设置极低的学习率，给新加的全连接层设置较高的学习率。
    - **特定正则化**：某些参数（如 `Bias` 或 `BatchNorm` 的参数）通常不需要权重衰减，可以单独分出一组将 `weight_decay` 设为 0。
~~~
# 将参数分为两部分：特征层和全连接层
optimizer = torch.optim.Adam([
    {'params': model.features.parameters(), 'lr': 1e-4}, # 底层学习率慢
    {'params': model.fc.parameters(), 'lr': 1e-3, 'weight_decay': 0} # 顶层快，且不加衰减
])
~~~

##### 梯度清零 zero_grad
如果没有进行梯度清零，这次循环计算出来的梯度会累加到参数的`.grad`上，因为Pytorch的设置是默认累加，并不是覆盖。如果忘记梯度清零会造成梯度越来越大，更新幅度失控。loss 不收敛，甚至直接爆炸。bug 很隐蔽，模型能跑但训练曲线异常，很难排查。
###### `.grad`
`.grad`是参数的一个成员变量。每个`requires_grad=True` 的 Tensor 都有一个 `.grad` 属性，初始值是 `None`，第一次 `backward()` 之后才会被创建成一个和自身形状相同的 Tensor。
~~~
w = torch.tensor([2.0], requires_grad=True)  
  
print(w.grad)  # None，还没反向传播  
  
loss = (w * 3).sum()  
loss.backward()  
  
print(w.grad)  # tensor([3.])，现在有值了
~~~
**所以`zero_grad` 实际上是在让`.grad`变成`None`**

##### 模型保存
~~~
torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
~~~

#### 验证循环
~~~
# 1. 切换到评估模式
model.eval() 

# 2. 禁用梯度计算环境
with torch.no_grad():
    for data, target in val_loader:
        # 数据搬运到设备
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)
        
        # 计算损失（可选）
        loss = criterion(output, target)
        
        # 计算准确率等指标
        # ... 逻辑处理 ...

# 3. 如果后续还要训练，记得切回训练模式
model.train()
~~~
- 如果漏掉 `eval()`：模型里的 Dropout 还在随机丢弃，导致同一个样本两次推理结果不一样，且指标变差。
    
- 如果漏掉 `no_grad()`：虽然结果正确，但程序会极其卡顿，且极易报 **Out of Memory (OOM)** 错误，因为显存被没用的梯度占满了。
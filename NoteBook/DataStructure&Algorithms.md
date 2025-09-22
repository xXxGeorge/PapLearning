---
tags:
  - ComputerScience
relation:
  - "[[Python学习]]"
teacher: 施章磊
---
## DATA STRUCTURE AND ALGORITHMS

### 复杂度分析(complexity)

我们讨论渐进复杂度分析

此时需要了解两个概念

###### 迭代(iteration)

迭代是一种从上而下的线性结构, 比较简单, 经典的迭代如循环等, 无需多言

###### 递归(recursion)

递归需要在函数体中调用自身, 如比较简单的求和

```python
def recursion(n: int) -> int:
  if n == 1:
    return 1
  return n + recursion(n-1)
```

##### 各种常见的时间复杂度

###### **<u>线性阶(Linear)</u>**

常见的迭代如循环就可以写出线性阶

```python
def Linear(n : int) -> void:
  for _ in range(n):
    print(1)
```

###### **<u>对数阶(Log)</u>**

常见的递归就可以写出对数阶

```Python
def LogRecursion(n: int) -> int:
  if n <=1:
    return 0
  return LogRecursion(n/2) + 1
```

当然迭代也可以写出对数阶的算法

```python
def LogIteration(n: int) -> int:
  sum = 0 # 用于求n, n/2...的和
  while n != 1:
    count += n
    n = n/2
  return sum
```

这两者的时间复杂度都是$ O(Logn)$

**注意⚠️!!**

在递归这里可以发现稍加改动就可以写出时间复杂度不一样的另一种方法

```python
def LogRecursion_fake(n: int) -> int:
  if n <=1:
    return 0
  return LogRecursion_fake(n/2) + LogRecursion_fake(n/2)
```

此时一共包含$ Log_2n$层, 而每层的操作都会比之前一层翻一倍. 所以总的操作数就是
$$
2^0 + 2^1 + 2^2 +...+2^{Log_2n} = 3 - 2n
$$
所以此时` LogRecursion_fake`这个算法的时间复杂度应该是$ O(n)$

###### **<u>线性对数阶(Linear*Log)</u>**

显然此时可以推测出一定有一种算法的时间复杂度是$ O(nLogn)$的形式, 而且不难推测出只需要将递推与迭代组合就可以写出类似满足要求的算法

```Python
def LinearLog(n: int) -> int:
  if n <= 1:
    return 0
  for _ in range(n):
    return LinearLog(n/2)
```

这是我一开始反应的线性对数阶, 然而显然这里面有许多问题. 首先语法上这个for循环里只有一个return能被用上就有问题了; 再者就算语法是允许的, 这里默认LinearLog在循环之后一次的运算量还是对数阶就有问题

事实上并不需要那么麻烦, 我们只需要在原先的算法中使得每一次都会有一个n的计算量即可

```python
def LinearLog(n: int, m: int) -> int: # m = n
  if n <=1:
    return 0
  for _ in range(m):
    print("")
  return LinearLog(n/2) + 1
```

这样就可以了. 然而我们这里引入了一个`m`, 是因为`LinearLog(n/2)`这里`n`的传入大小已经变成了二分之一, 所以得保证每层的计算量还是`n`个, 故引入了多出的`m`. 不过我们当然也可以通过两倍操作的方式保持每层计算量一定

```python
def linear_log_recur(n: int) -> int:
  """ 线性对数阶"""
  if n <= 1:
    return 1
  # 一分为二，子问题的规模减小一半
  count = linear_log_recur(n // 2) + linear_log_recur(n // 2)
  # 当前子问题包含 n 个操作
  for _ in range(n):
    count += 1
  return count
```

此处的`for`循环内的`count`其实只是随便写的, 只要有这个循环用于凑计算量即可. 我们以第二层为例, 传入`n/2`, 循环中只进行了`n/2`次操作, 但因为我们`return`了`linear_log_recur(n // 2) + linear_log_recur(n // 2)`, 所以这一层总的操作数为`(n/2)*2`



### 数据结构

#### 顺序表

#### 链表

##### 链表节点的定义

```cpp
typedef int LElemType;

typedef struct LNode{
    LElemeType data;
    LNode* next;
}* LList;
```

此处会发现在LNode的定义内部我们就写了LNode, 这是cpp的自引用, 详情可见以下Q&A:

> Q:
>
> 我懂了, 我只是需要一个指向LNode数据类型的指针, 而我写出结构体的那一起这个LNode的位置就已经存在了, 我只是需要一个指针指向他, 所以不需要知道内部结构是吗, 换句话说如果在这个结构题内部写的是LNode而不是LNode*, 这个程序就有问题了.
>
> A:
>
> 对！你理解得非常准！👏
>
> 我们来确认一下你说的这两点：
>
> **✅ 你说：**
>
> ​	“我只是需要一个指向 LNode 数据类型的指针，而我写出结构体的那一刻这个 LNode 的标识就已经存在了，所以不需要知道内部结构。”
>
> **完全正确！**
>
> 只要编译器看到 LNode*，它就只需要知道 “LNode 是个类型名”，并不会去深究它的大小或具体结构，因为：
>
> - 指针本身的大小是固定的（比如 4 个字节或 8 个字节），
>
> - 不管你指向的是啥，只要是个类型，编译器就能先给你安排一个指针位子，后面再填内存都行。
>
> **🚫 如果你写的是 LNode next; 而不是 LNode\* next;：**
>
> 那就真的是 **错误❗**
>
> 来看一下这个伪代码：
>
> ```cpp
> typedef struct LNode {
>     LElemType data;
>     LNode next;  // ❌ 错误！
> } *LList;
> ```
>
> 这一行 LNode next; 意味着 **当前结构体要完整地包含另一个 LNode 类型作为成员**。
>
> 但这个 LNode 又要再包含一个 LNode 成员… 就无限套娃了！编译器无法分配内存，也没办法结束这个定义。💥
>
> 所以这就会报错：
>
> ```
> error: field ‘next’ has incomplete type ‘LNode’
> ```
>
> **🧠 举个类比：**
>
> - LNode* next; 就像是你说 “我有一个朋友的电话📱”（你知道有这个人就够了，不需要知道他全身多重）。
>
> - LNode next; 就像你说 “我朋友住在我身体里面。” —— 然后他身体里又住着另一个人……这就爆炸了 🤯


### Tree&Forest

#### Practice
[[DS&A HW0301.jpg]]
##### Example 3.28
整体思路是利用二叉树几个量之间的关系简化程序, 双孩子节点数为dc, 单孩子为sc, 叶子节点为lf, 则我们只需要求出dc和lf即可得到sc
我们使用递归来编写
```cpp
void countRecursion(Bitree T){
	Bitree p = T; 
	int dc = 0, sc = 0, lf = 0;
	if(p->lc && p->rc == NULL || p->rc &&p->lc ==NULL){
	sc ++;
	}
	else if(p->lc == NULL && p->rc == NULL){
	lf ++;
	return;
	}
	countRecursion(p->lc, sc, lf);
	countRecursion(p->rc, sc, lf);
	dc = lf - 1;
	cout<<"there's"<<dc<<"dc,"<<sc<<"sc,"<<lf<<"lf."<<endl;
}
void countRecursion(Bitree p, int &sc, int &lf){
	if(p->lc && p->rc == NULL || p->rc &&p->lc ==NULL){
	sc ++;
	}
	else if(p->lc == NULL && p->rc == NULL){
	lf ++;
	return;
	}
	countRecursion(p->lc, sc, lf);
	countRecursion(p->rc, sc, lf);
}
```

修改后的版本为
```cpp
void countRecursion(Bitree T, int &dc, int &sc, int &lf) {
    if (T == nullptr) return;  // 如果当前节点为空，直接返回

    // 判断当前节点的度数
    if (T->lc != nullptr && T->rc != nullptr) {
        dc++;  // 度为2的节点
    } else if (T->lc == nullptr && T->rc == nullptr) {
        lf++;  // 叶节点
    } else {
        sc++;  // 度为1的节点
    }

    // 递归遍历左右子树
    countRecursion(T->lc, dc, sc, lf);
    countRecursion(T->rc, dc, sc, lf);
}
```
##### Example 3.29
要判断一个二叉树是否为严格二叉树, 只需判断是否存在度为一的节点
```cpp
bool ifStrict(Bitree T){
	if(!T) return true;
	if((T->lc != nullptr && T->rc == nullptr) || (T->lc == nullptr && T->rc != nullptr)){
		return false;
	}
	ifStrict(p->lc);
	ifStrict(p->rc);
}
```

但是这样写出来的函数出去特殊情况不会有返回值: 比如说在第三层函数结束了并返回了false, 但这个false被第二层函数接受到以后就无法继续传递了, 因为第二层函数的返回值都在递归之前.也就是说这个函数如果直接这么写的话, 前两句不return的话后面也就不会return了.
上一题是因为递归的每一层都会对dc, sc, lf变量留下影响, 不需要在第一层函数有返回值.
Example 3.29应该改为
```cpp
bool ifStrict(Bitree T){
	if(!T) return true;
	if((T->lc != nullptr && T->rc == nullptr) || (T->lc == nullptr && T->rc != nullptr)){
		return false;
	}
	bool leftResult = ifStrict(T->lc);
	bool rightResult =  ifStrict(T->rc);
	return leftResult&&rightResult;
}
```
##### Example 3.38
我觉得应该先使用层序遍历`void levelSearch`找到节点B, 如果没有找到返回0; 找到后使用递归`int countDepth`, 将递归的参数写为`int Rdepth, int Ldepth`, 两个递归之后比较左右的深度哪个更大.
```cpp
bool LevelOrderTraversal(Bitree T, int b, Bitree &position) {
	if (!T) return false;
	    LQueue Q;
	    QueueInit(Q);
	    EnQueue(Q, T); // 将根节点入队
	    while (Q.front != Q.rear) {
	        Bitree p;
	        QElemType temp;
	        DeQueue(Q, temp);
	        p = temp;
        
	        if (p->data == b){
	        position = p;
	        return true;
	        }
	        if (p->lc) EnQueue(Q, p->lc);
	        if (p->rc) EnQueue(Q, p->rc);
	    }
	    return false; // 整棵树遍历完没找到，返回 false
	}
int countDepth(Bitree T){
	    if(!T) return 0;
	    int leftDepth = 1 + countDepth(T->lc);
	    int rightDepth = 1 + countDepth(T->rc);
	    return leftDepth>rightDepth ? leftDepth:rightDepth;
}
```

[[DA&A HW0302.jpg]]
对于HW030
##### Example 3.48
```cpp
void ForestLevelOrder(Bitree T, void visit(TElemType)) {
    if (!T) {
        return;
    }

    LQueue Q;
    QueueInit(Q);
    EnQueue(Q, T);

    while (Q.front != Q.rear) {
        QElemType currentNode;
        DeQueue(Q, currentNode);
        if (currentNode) {
            visit(currentNode->data);
            
            if (currentNode->lc) {
                EnQueue(Q, currentNode->lc);
            }
            if (currentNode->rc) {
                EnQueue(Q, currentNode->rc);
            }
        }
    }
}
```


[[DS&A HW0401.jpg]]
##### Example 4.10
显然对于一个二叉树, 为了验证它是不是二叉排序树我们只需要验证对于每个节点左孩子是否小于右孩子, 又因为二叉排序树的子树一定是二叉排序树, 故我们显然可以用递归算法来减少代码量.
或是由二叉排序树的性质可知, 二叉排序树的中序遍历为严格递增序列.
```cpp
bool ifBiSort(Bitree T){
	if(! T) return true;
	if(T->rc == nullptr  && T->lc == nullptr) return true;
	if(T->rc != nullptr  && T->lc != nullptr){
		if( T->lc->data >= T->data || T->rc->data <= T->data){
			return false;
		}
	}
	else if(T-> lc == nullptr){
		if(T-> rc->data <= T->data) return false;
	}
	else{
		if(T->lc->data  >= T->data) return false;
	}
	
	return ifBiSort(T->rc) && ifBiSort(T->lc);
}
```

这个代码有一个问题在于, 我们只判断了节点的左右孩子与节点的数据大小关系. 然而二叉排序树的定义是左子树中的数据都小于根节点, 右子树中的数据都大于根节点, 所以可能存在一种情况:左子树中的某一个节点的右孩子小于其根节点, 然而这个值却大于整个二叉树的根节点.
所以我们有两种方式可以解决这个问题: 
- 在每一次迭代时多加入一个范围参数, 用来限制孩子的范围
- 利用二叉排序树的性质, 检验中序遍历是否为严格递增序列

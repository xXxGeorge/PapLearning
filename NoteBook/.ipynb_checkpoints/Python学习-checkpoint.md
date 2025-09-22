# Python

使用过的课件:

>  [第1章 Python3概述.pdf](../THEORY/Python 小学期/第1章 Python3概述.pdf) 
>
>  [第2章 基本语法.pdf](../THEORY/Python 小学期/第2章 基本语法.pdf) 
>
>  [第3章 流程控制.pdf](../THEORY/Python 小学期/第3章 流程控制.pdf)  
>
>  [第4章 组合数据类型.pdf](../THEORY/Python 小学期/第4章 组合数据类型.pdf)  
>
>  [第5章 字符串与正则表达式.pdf](../THEORY/Python 小学期/第5章 字符串与正则表达式.pdf) 
>
>  [第6章 函数.pdf](../THEORY/Python 小学期/第6章 函数.pdf)
>
>  [第7章 模块.pdf](../THEORY/Python 小学期/第7章 模块.pdf) 
>
>  [第8章 类和对象.pdf](../THEORY/Python 小学期/第8章 类和对象.pdf) 
>
>  [第9章 异常.pdf](../THEORY/Python 小学期/第9章 异常.pdf)  
>
>  [第10章 文件操作.pdf](../THEORY/Python 小学期/第10章 文件操作.pdf) 
>
>  [第11章 访问mySQL数据库.pdf](../THEORY/Python 小学期/ch11 访问mySQL数据库.pdf)  
>
>  [第12章 访问SQLite数据库.pdf](../THEORY/Python 小学期/ch12 访问SQLite数据库.pdf) 
>
>  [第13章 Numpy-快速数据处理.pdf](../THEORY/Python 小学期/ch13 Numpy-快速数据处理.pdf) 
>
>  [第14章 Matplotlib数据可视化.pdf](../THEORY/Python 小学期/第14章 Matplotlib数据可视化.pdf)  
>
>  [第15章 Pandas数据分析.pdf](../THEORY/Python 小学期/第15章 Pandas数据分析.pdf)  [未命名.md](../THEORY/Python 小学期/未命名.md) 

----

----





## 字符串与正则表达式

 [第5章 字符串与正则表达式.pdf](../THEORY/Python 小学期/第5章 字符串与正则表达式.pdf)

### re模块

#### re模块几种函数辨析

**1. re.match(pattern, string)**

- **特点：**从字符串的开头开始匹配模式。
- **用法：**用于检查字符串是否以某个模式开头。
- **区别：**只从字符串的开头进行匹配，如果开头不符合模式，就不会继续检查后面的部分。
- **返回值：**如果匹配成功，返回一个匹配对象 MatchObject；如果失败，返回 None。

**2. re.search(pattern, string)**

- **特点：**在字符串中搜索整个字符串，直到找到第一个匹配的位置。
- **用法：**用于查找字符串中是否存在匹配的模式，不限于开头。
- **区别：**可以匹配字符串的任何位置，不局限于开头。
- **返回值：**如果匹配成功，返回一个匹配对象 MatchObject；如果失败，返回 None。

**3. re.finditer(pattern, string)**

- **特点：**在字符串中搜索整个字符串，返回所有的匹配结果。
- **用法：**用于查找字符串中所有匹配的模式。
- **区别：**返回所有匹配结果的迭代器，每个元素是一个 MatchObject。
- **返回值：**返回一个迭代器，可以使用 for 循环或者 next() 来获取每一个匹配对象。

**4. re.findall(pattern, string)**

- **特点：**在字符串中搜索整个字符串，返回所有的非重叠匹配。
- **用法：**用于获取字符串中所有匹配的模式的字符串列表。
- **区别：**返回的是匹配的字符串列表，不包括 MatchObject。
- **返回值：**返回一个列表，列表中的每个元素是一个匹配的字符串。

**总结：**

- re.match 和 re.search 都是单次匹配，match 从开头匹配，search 在整个字符串中查找。
- re.finditer 返回所有匹配的迭代器，可以逐个访问每个 MatchObject。
- re.findall 返回所有匹配的字符串列表，不包括 MatchObject。
- 如果需要逐行匹配可以使用split先分割字符串,然后在循环中逐行匹配



## 类和对象

 [第8章 类和对象.pdf](../THEORY/Python 小学期/第8章 类和对象.pdf) 

### 对象

python显著特点即是其为面向对象的编程语言,他可以直接描述对象. 类比而言, class可以用于指代一类动物, 在这个类别下有一系列的特征, 如颜色, 形态,动作等. 其中颜色和形态这样的就相当于class 中的属性, 而动作这样的动态行为就是其中的方法.

### 类的定义

在类中, 有一些典型的结构:

```python
class MyFirstDemo:
    #定义类
    def __init__(self):  #  构造方法/构造函数
        #这个部分中的self会在具体的实例出现后被赋予为实例,如demo = MYFIrstDemo()后,self即被赋予demo
        #如果有其他参数就需要赋给self.param, 如self.param = param
    def method():
        --snip--  #  在此之后的方法就没有什么特殊的,直接当作函数写就行
```

在class中, 属性就相当于参数, 而方法就相当于函数.

### 专有方法

其中有一部分方法被称为专有方法, 如:

> - ` __init__()`
>
> - ` __del__()` 
>
> - ` __str__()` 
>
> - ` __ep__()` 
>
> 	…….

这些方法的特点在于名称不可改,且会自动调取

e.g.:

```python
class MyFirstDemo:
    def __init__(self, param_1 = 0, param_2 = 0):
        self.param_1 = param_1
        self.param_2 = param_2
    def __str__(self):
        return (f'param_1 = {param_1}, param_2 = {param_2}')
    def __del__(self):
        --snip--
```

如果此时有一个实例1.` objet = MyFirstDemo()`, 且同时有2.` print(objet)`以及3.` del object`, 则会出现以下情况:

1. 此时object作为类的实例被自动传入self

  > [Python self用法详解](https://c.biancheng.net/view/2266.html) 

  - 创建的实例可以用列表等保存, 如 `student = [Student(1), Student(2), Student(3)]`此时相当于

  ```python
  student[0] = Student(1913,'Javier', 'male')
  student[1] = Student(1934, 'Ella', 'female') 
  student[2] = Student(1978, 'Pepe', 'male')
  ```

  即实例为列表中的元素

2. 此时自动调用` __str__()`, 以字符串形式输出object, 否则将会是包含内存地址以及实例名的一长串

3. 使用del语句时, 析构方法是对象被删除前执行的最后一个程序

	- 同时因为python会自动管理内存, 程序运行结束后析构方法就会被自行调用

## 文件处理

### 文件读写

主要是用` with open() as :`这样就不用特意去` function.close()`去关闭文件了

#### 读取文件

-  `read() `		读取所有文件
-  `readline()`	   逐行读取
-  `readlines()`	  逐行读取所有行并存为一个列表



### CSV文件读写





## Numpy处理和使用

### 创建ndarray

常用` np.array()`函数进行创建. 可以将这个函数理解为强制转换

```python
import numpy as np
tup = (1, 2, 3, 4)
lst = list(tup)  #  这里将元组强制转化成列表
arr = np.array(lst)  #  这里将列表(以及元组之类)的转化成数组
```

因为是强制转换,所以列表有的特性他也会有一点,比如说可以用多层中括号表示多维数组, ~~将其中的一部分换成小括号, 小括号内就是数组的元素~~  且优先级很高. ==我们可以通过改变` dtype` 的方式改变维数, 但只能改变小括号的意义==

也可以通过` np.arange()`创立一个等差的一维数组, 相当于python中的` range()`. 然后再用` np.reshape()`函数将其变成想要的格式

```python
import numpy as np
arr = np.arange(10).reshape(2, 5)
print(arr)
```

此时的输出是

```
[[0 1 2 3 4]
 [5 6 7 8 9]]
```

可以发现, reshape里面为位置参数, 分别对应着从零开始的各个轴(axis).



#### 调整ndarray中的参数(用` dtype`影响数组的维度)

其中比较重要的是`dtype`的定义, 即数组元素的数据类型. 有基础的` int 8`, ` int64`, ` float8`, ` S20`等等,但有时我们不止希望在数组中存储简单的数字或字符, 而是一些结构数据.

```python
import numpy as np
arr = np.array([(1,2,3), (4,5,6)])
print(arr, arr.ndim)
```

此时可以得到输出为

```
[[1 2 3]
 [4 5 6]] 2
```

即程序将小括号看作和中括号一样的效果,将数组创建为二维数组

而如果为程序添加` dtype`便会产生差异

```python
import numpy as np
arr_type = np.dtype([('first', 'i1'), ('second', 'i2'), ('third', 'i4')])
arr = np.array([(1,2,3), (4,5,6)], dtype = arr_type)
print(arr, arr.ndim)
```

此时输出

```
[(1, 2, 3) (4, 5, 6)] 1
```

程序知道了数组内的元素一定是三种组成的结构数据,所以将()内看作一个元素

此时还可以看出[]的优先级之高

```python
import numpy as np
arr_type = np.dtype([('first', 'i1'), ('second', 'i2'), ('third', 'i4')])
arr = np.array([[1,2,3], [4,5,6]], dtype = arr_type)
print(arr, arr.ndim)
```

输出为

```
[[(1, 1, 1) (2, 2, 2) (3, 3, 3)]
 [(4, 4, 4) (5, 5, 5) (6, 6, 6)]] 2
```

此时因为都是中括号, 所以维度是2, 但是因为dtype确定了每个元素内部一定是三类数据, 所以程序自动将缺少的两个数据补全了

## Matplotlib处理和使用

#### Matplotlib 中子图的创建

可以使用语句` plt.subplot(x, y, z)`来创建子图, 其中x是横向有几个子图, y为纵向有几个子图, z为当前对哪一个子图进行操作

我们可以使用`matplotlib`的`pyplot`接口或者`API`接口进行操作, 两个稍有不同

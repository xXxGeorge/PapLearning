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



## Docstring写法

将docstring写在要解释的事物内部第一行, 使用三个双引号. 查看时使用` help(functionName)`或者` print(functionName.__doc__)`

## 函数

函数后加()表示调用此函数, 不加表示这个对象

### 函数参数类型

函数定义的时候要定义参数列表,此时有位置参数和关键字参数两种.

#### 位置参数

位置参数在函数中以位置赋值,同时对位置参数来说不可以用关键字赋值的方法进行赋值;

```python
def my_function(param1, param2):
    print(param1, param2)
#  my_function(10, 20)  # 输出：10 20 (调用)
```

#### 关键字参数

关键字参数需要以等式赋值,不赋值则是默认值

```python
def my_function(param1, param2, param3="default", param4="default"):
    print(param1, param2, param3, param4)
# my_function(10, 20)  # 输出：10 20 default default
# my_function(10, 20, param3="not default")  # 输出：10 20 not default default
# my_function(10, 20, param4="another value")  # 输出：10 20 default another value
```

#### 仅使用关键字参数

```python
def my_function(*, param1, param2):
    print(param1, param2)
# my_function(param1=10, param2=20)  # 输出：10 20
# my_function(10, 20)  # 会报错：TypeError: my_function() takes 0 positional arguments but 2 were given
```

#### 混合型

位置参数必须在关键值参数前赋值



### 特殊函数类型(生成器)



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



## 模块







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

## 异常

### 异常的基本语法

#### try…except…else…finally

在程序中, 倘若其中一段有可能出现错误, 我们应该加上try语句. except语句后面写上对应的异常类型和处理方法. 需要注意的是我们应该尽量具体的写出异常的类型. 有时我们不清楚异常的具体类型, 可以利用try…except语句是一对多的特性详细排查.

```python
try:
    # 可能会抛出异常的代码
except IOError as e:
    print("IOError occurred:", e)
except ValueError as e:
    print("ValueError occurred:", e)
except TypeError as e:
    print("TypeError occurred:", e)
except Exception as e:
    print("Unexpected error occurred:", e)
```

常见异常有` IOeError, ValueError, TypeError, StopIteration`等等. ` StopIteration`是迭代器超出了范围时的报错, 但是因为for循环会自动处理这种异常, 所以一般在` next()`配合的迭代中才需要使用
` else`语句是如果没有异常时执行的语句. ` finally`显然就是最后无论如何都会执行的语句, 常用于确保资源的释放或清理操作.

#### raise语句

主动抛出异常. 常和自定义异常放在一起使用, 需要使用except语句接受.

#### 自定义异常

自定义异常是异常类` Exception`的子类, 写的时候如下

```python
class ErrorName:
    def __init__(self, sentence):
        self.sentence = sentence
    def __str__(self):
        return f'{self.sentence}'
```

使用的时候

```python
try:
    if ...:
        raise ErrorName(‘...’)  #  括号内写上你此时想要定义的异常
    except ErrorName as e:
        print(e)
```



## 文件处理

### 文件读写

主要是用` with open() as :`这样就不用特意去` function.close()`去关闭文件了

#### 读取文件

-  `read() `		读取所有文件
-  `readline()`	   逐行读取
-  `readlines()`	  逐行读取所有行并存为一个列表

上述是通用的文件读取方法,而对于一些常用的文件, 我们还有相应的类来读写文件

#### CSV文件读写

对于CSV文件,我们需要先导入CSV库, 然后进行读写

##### 以字典方式

首先我们知道CSV文件是一种类似于表格的形式,所以我们可以以字典的形式将其读出, 以表头作为其键

```python
import csv
dicts = []
with open('file.csv', 'r', encoding = 'utf-8-sig') as f:
    reader_ = csv.DictReader(f)
    dicts.append(next(reader_))
    for dict in reader_:
        dicts.append(dict)
```

DictReader()会产生一个生成器, 内容是键值对形成的列表, 以表头作为键. 如果没有表头, 可以认为添加代码

```python
header = ['first', 'second',...]
reader_ = csv.DictReader(f, fieldnames = header)
```

也可以将字典列表形式的内容保存至CSV文件中, 使用` csv.DictWriter()`

```python
import csv
headers = [1, 2]
dict = [{1:23, 2:24}, {1:56, 2:45}, {1:26, 2:89}]
with open('file.csv', 'w', encoding = 'utf-8-sig', newline = '') as f:
    writer = csv.DictWriter(f, fieldnames = headers)
    writer.writeheader()
    writer.writerows(dict)
```

> 具体来说，`csv.DictWriter()` 是 `csv` 模块中的一个类，用于创建一个对象，该对象可以写入以字典为单位的数据到 CSV 文件中。当你使用 `csv.DictWriter()` 创建了一个对象后，你可以通过它的方法将字典数据写入到一个已经打开的 CSV 文件中。
>
> - `csv.DictWriter()` 返回的是一个 `DictWriter` 对象，该对象在实例化时传入了一个文件对象 `csvfile` 和字段名列表 `fieldnames`。
> - `DictWriter` 对象 `writer` 可以调用方法 `writeheader()` 写入表头（即字段名），以及 `writerow()` 方法逐行写入每个字典数据。
>
> 因此，`csv.DictWriter()` 返回的是一个 `DictWriter` 类的实例对象，你可以通过这个对象来操作和写入 CSV 文件。



##### 以嵌套列表方式

###### 以迭代器形式读取

```python
import csv
header = []
rows = []
with open(file, 'r', encoding = 'utf-8') as f:
    csv_reader = csv.reader(f)
    header = next(csv_reader)
    for row in csv_reader:
        rows.append(csv_reader)
```

###### 以列表形式读取

```python
import csv
rows = []
with open(file, 'r', encoding = 'utf-8') as f:
    csv_reader = list(csv.reader(f))
    rows = csv_reader
```

###### 以列表形式写入

```python
import csv
import random
row = [random.randint(1, 10) for x in range(10)]
rows = [row for x in range(5)]
with open(fole, 'w', encoding = 'utf-8', newline = '') as f:
    writer = csv.writer(f)
    writer.writerow(rows[0])
    writer.writerows(rows[1:])
```

***CSV模块方法总结***

​	函数有` reader(), writer(), DictReader(), DictWriter()`, 前两种对应列表, 后两种对应表格. ` DictReader`返回一个迭代器, 内容是键值对列表, 键是表头, 值是对应的内容. 如果没有表头就在后面加上` fieldnames`参数以提供表头. ` DictWriter()`创造了一个对象, 创立时需要传入一个参数` fieldnames = header`也就是提供表头, 然后其中的` writehead()`方法就会写入表头, ` writerows(rows)`方法就会根据键一行行写入内容

​	` reader()`返回一个以每一行为内容的迭代器. ` writer()`实现的方法就是` writerow()`&` writerows()`, 一个写入单行一个写入多行.



## Numpy处理和使用

### 创建ndarray

常用` np.array()`函数进行创建. 可以将这个函数理解为强制转换

```python
import numpy as np
tup = (1, 2, 3, 4)
lst = list(tup)  #  这里将元组强制转化成列表
arr = np.array(lst)  #  这里将列表(以及元组之类)的转化成数组(arry内部只需要是一个序列, 直接用range()对象也行)
```

因为是强制转换,所以列表有的特性他也会有一点,比如说可以用多层中括号表示多维数组, ~~将其中的一部分换成小括号, 小括号内就是数组的元素~~  且优先级很高. ==我们可以通过改变` dtype` 的方式改变维数, 但只能改变小括号的意义==

除此之外, 还有一个类似的函数` asarray()`, 二者只在对` ndarray`的处理上出现差别, 对其他数据类型的时候统一将元数据复制, 而对` ndarray`操作的时候, ` asarray`的操作是不复制, 直接对原数组进行操作, 所以对原数组的操作也会反映到`asarray`处理后的对象身上

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

### ndarray的向量特性

#### 简单的运算

如果使用乘法,加法等运算, ndarray中会处理成向量式的一一对应计算. 倘若是单值对数组赋值或是数组不匹配, 会先进行广播再计算

#### 用布尔数组进行筛选

数组的比较运算也是向量化的, 我们可以使用布尔索引来进行筛选操作

```python
import numpy as np
nd = np.arange(12).reshape(3,4)
print(nd%2 == 0)
print(nd[nd%2 == 0])
```

` nd%2 == 0`作为向量运算对数组内的元素一一处理, 得到` True` 或` False`的结果形成了一个新的数组, 同时这个数组也可以作为一个布尔索引来索引原数组

#### 通用函数(ufunc)

- `numpy.squrt`
- ` numpy.exp`
- ` numpy.maximum` 
- ` numpy.add`

后两个接受两个数组作为对象.

通用函数就是一种执行元素级运算的函数. 他将简单的算术计算封装成向量性的函数

## Matplotlib处理和使用

#### Matplotlib 中子图的创建

可以使用语句` plt.subplot(x, y, z)`来创建子图, 其中x是横向有几个子图, y为纵向有几个子图, z为当前对哪一个子图进行操作

我们可以使用`matplotlib`的`pyplot`接口或者`API`接口进行操作, 两个稍有不同



## Pandas数据分析

### Series的使用

Siries对象会为输入的序列生成一个索引, 可以自己设置索引, 否则默认生成从零开始的顺序数列.使用时可以直接用` pd.Series(target, index = list_2)`生成.  其中的target可以是列表,字典等等. target为字典的时候, 以键值为索引, 此时在index里面继续写键值的列表可以将Series对象的序列按列表顺序排列, 列表中多出来的元素对应NaN, 缺少的数据就没了. 使用属性` .index`可以直接更改索引值(==DataFrame不行==)

Series内容和序列对应两个属性index和value, 可以使用` .index`类的形式打印出来

### DataFrame的使用

` df = pd.DataFrame()`来创建, 内部可导入

- 二维ndarray
	数组,列表,元组之类的组成的字典

	- 数据矩阵，还可以传入行标和列标, 每个序列会变成DataFrame的一列。所有序列的长度

		必须相同

- NumPy的结构化/记录数组

	由Series组成的字典

	- 类似于由数组组成的字典, 每个Series会成为一列。如果没有显式指定索引，则各Series的索引会被合并成结果的行索引

- 由字典组成的字典
	- 各内层字典会成为一列。键会被合并成结果的行索引，跟“由Series组成的字典”的情况一样
- 字典或Series的列表
	- 各项将会成为DataFrame的一行。字典键或Series索引的并集将会成为DataFrame的列标
- 由列表或元组组成的列表
	- 类似于二维ndarray



简单来说, 基础格式即` obj = pd.DataFrame(data, index = pattern, columns = pattern_2)`

可以用` df.columns.name`来定义列索引的名称,列索引也一样

读取CSV文件的时候参数` index_col`也可以指定行索引, ` index_col = [0,1]`意为指定第一列和第二列为行索引, 同时两列的第一行为索引名; 用` index_col = ['year']`这样的列名也可以,此时的列名被作为索引名

```python
import pandas as pd
df = pd.read_csv('medicare.csv', index_col=[0])
df.columns.name = 'Measure'
print(df)
```

结果

```
Measure                Classification  ... Total Payment
City, State                            ...              
ANCHORAGE, AK    Alcohol and Drug Use  ...     68,641.00
JUNEAU, AK       Alcohol and Drug Use  ...        93,951
ANNISTON, AL     Alcohol and Drug Use  ...     44,087.00
BIRMINGHAM, AL   Alcohol and Drug Use  ...     42,433.00
FLORENCE, AL     Alcohol and Drug Use  ...     46,057.00
...                               ...  ...           ...
SANTA ROSA, CA     Circulatory System  ...       105,413
FRENCH CAMP, CA    Circulatory System  ...     80,469.00
STOCKTON, CA       Circulatory System  ...     60,006.00
MANTECA, CA        Circulatory System  ...     87,589.00
STOCKTON, CA       Circulatory System  ...    223,126.00

[10000 rows x 16 columns]
```

如果没有` index_col = [0]`, 就会自动生成行索引

#### DataFrame中数据的存取

- **[]**方法

	- 对于列而言,无法进行切片操作, 除此以外无论是按序列还是按列名都可以操作
	- 对于行而言,只能进行切片操作(除去布尔类型的), 也就是说对行的存取无法去除` columns`
	- 对于布尔类型来说,可以对整体操作, 如` print(df[df > a])`,各个位置如果对应` True`值即被输出,否则为NaN; 也可以对行筛选, 如` print(df[df.c1 > a])`, 此时输出符合条件的行, 显然筛选条件为列.

- **loc**方法/**iloc**方法

	- `df.loc[a, b]`. 每个位置对应一个轴, 从第0轴开始不过a, b都==只能写索引名==. 支持切片操作, 如果需要存取某一列则需要写成` df.loc[:, b]`, 也支持多个索引, 需要加括号` df.loc[:, [b1, b2]]`

	> 需要注意不管是行还是列, 都是以一个Series形式输出的, 也就是说出来的时候是一列
	>
	> 如果想用索引位置来存取的话用iloc()

	- 如果有多个行索引, 用小括号逐一括起` df.loc[(1a, 2a), b]`, 如果只对某一个行索引有要求可以用` slice(None)`, 表示无要求的行索引



## 迭代器,生成器,可迭代对象

#### 迭代器

- 迭代: 逐个获取元素的方式

	在其中要实现` __iter__()`方法和` __next__()`方法, ` __iter__()`方法要返回一个可迭代对象, 因为现在我们定义的类是为了迭代的, 其中的self本身就是可迭代对象, 所以可以写为` def __iter__(self): return self`. 此时再在` __next__()`方法中定义具体的迭代方法, 比如说现在要定义一个类似于` range` 的迭代器, 我们就可以写成

``` python
class Range_copy:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __iter__(self):
        return self
    def __next__(self):
        if current > self.end:
            raise stopIterration
        current = self.start
        self.start += 1
        return current
```



这个篇章只是因为有很多函数的返回值并不是我们想象中的列表或直接放在迭代中无法运行, 所以特别取出一个板块归纳一下

> 1. map() 函数:会返回一个 map 迭代器,需要用 list() 转换为列表。
> 2. filter() 函数:会返回一个 filter 迭代器,需要转换为列表。
> 3. zip() 函数:返回一个zip迭代器,可以聚合多个列表。
> 4. enumerate() 函数:在迭代器中同时获取索引和值,返回enumerate迭代器。
> 5. dict.keys() 方法:返回字典键的迭代器。
> 6. dict.values() 方法:返回字典值的迭代器。
> 7. dict.items() 方法:返回字典键值对的迭代器。
> 8. set 和 dictionary 的 popitem() 方法:可以返回一个项的迭代器。
> 9. reversed() 函数:返回一个反向迭代器。
> 10. file 对象:文件对象支持迭代器接口,可以直接在文件上迭代读取。
> 11. generator 函数:含有 yield 的函数会返回一个生成器迭代器。
> 12. csv.DictReader()函数: 读取csv文件的时候常用,其结果是一个迭代器

上述是常见的返回值是迭代器的函数, 对于这样的函数咬调用其结果要么循环, 要么将其强制转化成列表

#### 生成器

- 生成器使用更简洁，因为它们可以像编写普通函数一样编写代码，并使用 `yield` 逐个生成值。
- 生成器可以通过生成器表达式或者生成器函数创建。
- 生成器在需要时惰性生成值，节省了内存和计算时间。

```python
# 使用生成器表达式
generator_expr = (x for x in range(5))
print(next(generator_expr))  # 输出 0
print(next(generator_expr))  # 输出 1

# 使用生成器函数
def generator_func():
    for i in range(5):
        yield i

generator = generator_func()
print(next(generator))  # 输出 0
print(next(generator))  # 输出 1
```



生成器函数有点像是将`print`或者`return`换成`yield`, 用`print`理解可以更方便的理解循环输出, 不过是被拆分开来, 用`return`理解更方便理解返回值的方面.

#### ***==生成器的特点实例==***

```python
list = [lambda : x for x in range(5)]
for i in list:
    print(i())
```

```python
list = (lambda : x for x in range(5))
for i in list:
    print(i())
```

上面两个例子展现了生成器的特点, 逐个生成所需的值. 对于例一, 列表里生成的都是函数对象, 返回的都是同一个对象n, 且同时存储在列表里, 所以会随着n的变化而变化. 但对于例二来说, 每次输出时才会产生一个对象, 且产生后就被print输出, 不会被后面所影响.例一的输出是

```
4
4
4
4
4
```

而例二是

```
0
1
2
3
4
```

由此可以更直观的感受到列表和生成器的区别.同时虽然例二的输出结果和`[x for x in range(5)]`一样, 但我们要知道其中的不同. 而这三个例子中虽然变量都是统一的, 但只有前面两个中是引用传递, 第三个中的赋值只是值传递(类似, 毕竟此处没有函数, 不存在参数传递). 



## 零碎知识

```python
list_ = [x for x in range(1, 16)]
list_alter = [list_[0]]
list_alter.extend(filter(lambda x: x%2 == 0, list_[1:]))
print(list_alter)
```

` extend()`内需要写入一个列表, 但是通过这个题目能发现写入一个迭代器也行, 也就是说只要是一个可迭代对象就可以. ` extend()`是操作, 而不是返回新列表

注意在第二行, 如果直接写为` list_alter = list_[0]`的话, `list_alter`就无法被定义为一个列表, 所以只要我们想将一个东西作为元素加入新列表, 要么用` append()` , 要么用`[]`来定义列表.

` XXXX*N`这样的式子均表示复制其引用,相当于浅复制, 并非是面向对象本身复制, 而是将引用的id复制了一遍

除此之外, 我们还可以讨论` id()`操作的处理对象是什么, 究竟是对象本身还是引用
```python
a = [1]
b = [1]
print(id(a), id(b))
#输出
4374907264 4374909056
```


因为列表是可变对象, 所以一经创建基本可以肯定不会再发生改变, 所以每个列表的id地址显然都是唯一的, 而此时他们所指对象的数值却是相等的, 由此可见, 如果` id()`操作是针对对象的话就会输出一样的结果, 而现实中输出了两个不一样的结果, 结论就是` id()`操作是针对引
用的.

```python
a = 1
b = 1
print(id(a), id(b))

#输出
4402780320 4402780320
```


这里为什么又是一样的呢, 推测可能是对于不可改变对象, 如果两个对象的内容完全一样就不会在内存中产生多余的损耗, 而是直接使用同一个引用
验证:

```python
a = (1,)
b = (1,)
print(id(a), id(b))

#输出
4375216032 4375216032
```

由此可见, 这个结论是对不可变对象普遍适用的.

/Users/horatius/eclipse/java-2024-063

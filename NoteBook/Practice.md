#  Wrong Practice

#### 5-1.10

```python
my_list = [[1, 2, 3]]*2  
my_list[0][0] = 'hack'
print my_list
```

结果

```
[[hack, 2, 3], [hack, 2, 3]]
```

` *2`操作事实上是将子列复制了一次, 两个子列的地址是一样的, 因为其复制的是引用而不是值. 所以此时当` my_list[0][0]`的引用被修改之后, 两个子列都被修改了

#### 5-1.8

```python
list = [lambda : x for x in range(5)]
for i in list:
    print(i())
```

这里我们知道`lambda`函数的结果是一个函数对象而不是返回值或函数结果, 所以此时`list`的结果是一个由函数组成的列表, 所以输出时`i`要写成`i()`以表示输出. 而这个列表的中的函数返回值都是`n`, 所以列表里的值一定都是一样的, 每经过一个新的循环, `n`的指向就发生一次变化, 所以最后`n`变为4, 列表里所有的元素都变成返回值为4的匿名函数
此外同样的例子如果将` []`换成` ()`, 结果将变成0到4的序列. 因为此时对象是一个生成器, 只有被调用时才会产生结果. n = 0时它以0被单个输出, n = 1时它以1被输出

而此时与定义函数又有差别. 在一般` def`定义的函数中, 函数内部的不可变对象是对值的传递, 所以不会不会对原本的对象产生影响, 也就是创立了一个新对象, 而对可变对象来说则是引用传递 

#### Experiment

为考试排座位并将程序封装为一个函数

```python
def seat_decision(file):
    import pandas as pd
    import random
    with open(file, 'r', encoding='utf-8-sig') as f:
        obj = pd.read_csv(f, index_col=[0])
        obj.columns.name = '信息'
    list_ = [x for x in range(1, len(obj.index))]
    random.shuffle(list_)
    list_1 = pd.Series(list_)
    obj['座位'] = list_1
    print(obj)
    obj.to_csv('try.csv', sep=',', header=True, index=True, encoding='utf-8-sig')
file = '考生名单.CSV'
seat_decision(file)
```

文件如下:  [考生名单.CSV](../PRACTICE/PycharmProjects/WALLET/考生名单.CSV)

同时有小阮的示范:

```python
# 
# @Time : 2024/7/13 9:07
# @Author : 阮宗利
# @File : 考试座位随机排列.py
# @Software: PyCharm
import csv
import os
import random
from typing import List


def rand_seating_arrangements(names_csvfile, out_csvfile):
    """
    给定考生名单，随机安排座位
    @param names_csvfile: 考生名单文件
    @param out_csvfile: 含随机座位的名单输出文件名
    @return: None
    """

    '''
    输入格式：
    序号，学号，姓名 
    输出格式：
    序号，学号，姓名，座位号
    '''
    # 读取所有名单到一个字符串列表
    with open(names_csvfile, 'r') as fp:
        reader: List[str] = csv.reader(fp)  
        header: List[List[str]] = next(reader)  # 表头列表
        data = list(reader)  # 考生名单嵌套列表
    header.append('座位号')  # 增加座位号列的表头
    numbers = len(data)  # 考生人数

    # 按考生人数不重复抽取
    seat_ids = random.sample(range(1, numbers + 1), numbers)

    # 验证是否重复
    assert len(seat_ids) == len(set(seat_ids)), "有重复！！！"

    # 将随机座位号添加到data列表末尾
    for record, s_id in zip(data, seat_ids):
        record.append(str(s_id))

    # data写入csv文件
    with open(out_csvfile, 'wt', newline='', encoding='utf-8-sig') as fp:
        writer = csv.writer(fp)
        writer.writerow(header)  # 写入表头
        writer.writerows(data)  # 写入所有名单数据行

    print(f'考试座位随机排列完成，已保存至文件{os.getcwd()}\\{out_csvfile}')


if __name__ == '__main__':
    in_file = '考生名单.csv'
    out_file = '考生名单_排座.csv'
    rand_seating_arrangements(in_file, out_file)
```



**Tips**

其中几个之前没有注意过的写法:

1. header: List[str] = … : 在header = …的基础上加了一些修饰, 方便他人理解
2. assert: 断言. 异常处理的内容, 多用与调试和测试阶段, 如果后面的布尔表达式为False, 则输出后面的内容.

3. ` zip()`函数. 聚合多个列表形成迭代器, 对应关系是按照位置一一对应









##### 旅行商问题/TSP(Hamilton圈)
website: 大学生在线
对城市网络$D = (V, A, D)$, $V$为城市集合, 城市个数为$n$, $A$为交通道路集合, $D$为城市距离矩阵, 若$(i, j) \in A$, $d_{ij} \in D$ 时$i$,$j$之间的距离, $x_{ij} = 0$或$1$表示是否走过$i$到$j$的路. 则有
$$
min \sum_{i, j = 1}^{n}x_{ij}d_{ij}
$$
[TSP问题收集](http://www.math.uwaterloo.ca/tsp/index.html)
[TSP算法收集](https://github.com/kellenf/TSP_collection)


### 智能算法

#### 禁忌搜索算法(Tabu Search Algorithm, TS)
主要用于解决多峰值的优化问题(跳出经典算法中多峰问题中局部最优解的陷阱)
陷入局部最优解的时候, 将局部最优解存入禁忌表, 抹除此处的数据继续进行搜索. 直到禁忌表收集了区间内所有的局部最优解, 在禁忌表内寻找最优解.

# JAVA 学习

### 第一个小实验（经典程序 HelloWorld）

首先在记事本中写下如下程序：

```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("hello world");
    }
}
```

然后打开 Terminal，使用` cd`命令跳转到此程序所在的目录（ 如果不清楚可以将程序拖入 Terminal），我们将文件放在了` \Users\horatius\Desktop`下，所以` cd \Users\horatius\Desktop `， 然后使用` javac`来编译程序并生成` .class`文件，最后直接使用` java HelloWorld`打印出内容

注意将文件后缀改为` .java`，并且文件名称和类的名称应该保持一致。



### Eclipse

尼玛的臭逼煞笔Eclipse中英版用词和布局还不一样

中文版的/窗口/首选项对应的是英文版的` /Eclipse/settings`,中英文互译是吧,首选项–Preference–设置–settings真是太直接了.

### 变量

#### 字段修饰词static

表示这个变量属于类.所有根据类创立的实例共享一个副本.*所以如果有一个实例改变了`static`变量的值,则在这之后的实例被创建的时候就会以新的值创建*

#### 字段修饰词final

表示这个变量只能赋值一次,此后不能再修改其内容.如果在构造方法之前,定义的时候直接初始化则与`static final`变量区别不大,不过在内存分配上还是有区别.

### 类和对象

java 也是面向对象编程的语言，他的变量创建和 C 语言类似,需要声明类型以及可见性,其中声明类型包括自定义类.

java是一种纯面向对象的编程语言,强调封装性,所有的内容必须放在类中进行.这意味着无法像c语言一样直接声明函数,或者说方法.在java中如果想声明一个方法,但是调用的时候不想对对象调用或是不想创建对象,可以使用` static class`,其中的` static`只是沿用C++中的用法,其实也可以叫做类方法,也就是说不需要对对象调用,可以直接对类使用. e.g.

```java
public class example{
    public static outPut{
		System.out.println("This is a test class");
    }
    public static void main(String[] args) {
        // 调用静态方法
        example.outPut();
    }
}
```

在这里我们通过定义一个静态方法实现没有实例直接调用方法.

#### 方法参数

方法参数的调用分为值传递(call by value)和应用传递(call by reference), 在java中方法参数都是值传递.

其中分为两种情况:

- 基本数据类型: 值传递且函数体中对参数的改变不会影响原本的变量
- 对象引用: 值传递,传递的是对于引用的复制.其中对于引用的改变不会对外部产生影响,而通过引用改变所指变量的内部的改变会被保留. e.g.

```java
class test{
    private int a = 1;
    public void setA(int newa){
        a = newa;
    }
}
class useful{
    public static usefulfunc(test a){
        a.setA(5);
        System.out.println(a);//其实应该用toString方法,这里偷懒一下就这样表示
	}
    public static unusefulfuc(test a, test b){
        test c;
        c = a;
        a = b;
        b = c;
    }
}
public class Test{
    public static void main(String[] args){
        test exam_1 = new test();
        test exam_2 = new test();
        exam_2.setA(7);
        useful.usefulfunc(exam_1);
        useful.unusefulfunc(exam_1, exam_2);
	}
}
```

在这个程序中,` usefulfunc`是有效的,因为传入的参数是对引用的复制,这个复制和原本的引用都指向一个地址,所以改变这个地址内的变量值还可以被保留的;而` unusefuluc`中的操作是交换两个引用指向的对象,而此时交换的并不是原本的引用,而是对引用的复制,所以不会对原本的引用造成影响.

### IO流与文件读写

#### File类

File类可以表示一个文件或目录对象,然后可以对他进行创建,删除,改名等操作.

```java
package learningProcess;
import java.io.*;
public class FileClass {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		File doc = new File("~/horatius/AnimationCN");
		System.out.println(doc.getAbsolutePath()); ///Users/horatius/eclipse-workspace/Day1/~/horatius/AnimationCN
	}
}
```

我原本期望输出`Users/horatius/AnimationCN`,但不知为何是这个输出.

解决:java不会识别~为主目录.

```java
package learningProcess;

import java.io.*;

public class FileClass {
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		File doc = new File("/Users/horatius/AnimationCN");
		System.out.println(doc.isDirectory());
		if(doc.exists()) {
			File[] A = doc.listFiles();
			for(int i = 0; A[i] != null; i++) {
				System.out.println(A[i].getName());
				System.out.println(A[i].isDirectory());
			}
		}
	}
}
```

```
true
.DS_Store
false
DanDaDan
true
Books
true
[AnimeRep] 蓦然回首   Look Back [1080p][简中内嵌][AI翻译润色].mp4
false
BlueBox
true
CyberPunk
true
Exception in thread "main" java.lang.ArrayIndexOutOfBoundsException: Index 6 out of bounds for length 6
	at learningProcess.FileClass.main(FileClass.java:12)
```

从这个结果可以看出来我实现了我的目的但是也暴露出一些问题:

- 这个地方本来应该写成` if(doc.exist() && doc.isDirectory())`但是因为我调试过程中不确定是否是目录因此分开写了
- 这个地方用`for`循环会导致超出`range`,因为在处理完`A[5]`之后数组中已经没有内容了,但是为了判断`A[i] != null`必须出现`A[6]`才能进行判断.这里用`for-each`比较妥当.
- 从中我们可以判断出什么是目录什么是文件:有下级结构即可称之为目录.



### 多线程

#### Thread类以及其子类

创建新的线程需要调用start()方法, 而不是调用run()方法,这样才能开启一个新的线程,若是调用run()方法则是在原线程中运行.

#### 线程状态

New, ready, running, sleep, block, wait等多种状态,其中可以用图解释.
有意思的一点是ready和running状态之间是双箭头,可以理解为ready就是线程的排队状态, 正常情况下线程不会一次性运行完,而是不同线程轮流运行,而等待时间中线程就处于ready状态

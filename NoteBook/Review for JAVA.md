## Reviewing Java

[[Java 学习]]

### File and IO stream

#### File 类

##### 用File类来实现对目标结构的查找(解析子文件以及子目录)

```java
package fileDealing;

import java.io.File;

class filetry{
	public static void getinfo(File dir) {
		String innerDirs = "", innerFiles = "";
		File[] innerStructure = dir.listFiles();
		for(File f:innerStructure) {
			if(f.isFile()) {
				innerFiles += f.getName() + '\n';
			}
			else {
				innerDirs += f.getName() + '\n';
			}
		}
		System.out.println("Files are: \n" + innerFiles);
		System.out.println("Dirs are: \n" + innerDirs);
	}
}
public class getInformation {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		File doc = new File("/Users/horatius/Desktop/Theory");
		filetry.getinfo(doc);
	}

}
```

这里需要注意创建新的File对象的语法, 应为` File Name = new File(filePath)`, 而不是` File Name = new File(name, filePath)`

### some details

#### 方法引用

**什么是方法引用？**

方法引用是 Lambda 表达式的一种简写形式。它允许你直接引用现有的方法，而不是使用 Lambda 表达式来调用它。方法引用使用 :: 语法。

**类型**

方法引用有四种主要类型：

​	1.	**静态方法引用**：`ClassName::staticMethodName`

​	2.	**实例方法引用**：`instance::instanceMethodName`

​	3.	**特定对象的实例方法引用**：`ClassName::instanceMethodName`

​	4.	**构造方法引用**：`ClassName::new`

**示例**

	1.	**静态方法引用**：

```java
Function<Integer, String> intToString = String::valueOf;
String result = intToString.apply(123); // 输出 "123"
```

	2.	**实例方法引用**：

```java
String str = "Hello";
Supplier<Integer> stringLength = str::length;
int length = stringLength.get(); // 输出 5
```

	3.	**特定对象的实例方法引用**：

```java
Function<String, Integer> stringToLength = String::length;
int length = stringToLength.apply("Hello"); // 输出 5
```

	4.	**构造方法引用**：

```java
Supplier<List<String>> listSupplier = ArrayList::new;
List<String> list = listSupplier.get(); // 创建一个新的 ArrayList 实例
```



## 考试复习资料

**建立两个类及并进行相应测试：**

**① 建立一个学生类Student，数据成员包括学号、姓名、性别，成绩等，方法成员包括构造方法、set/get方法、toString( )方法等。**

**②** **建立一个管理学生对象的类StudentManager。在其中定义向量（或列表）成员，用于存储学生对象；提供若干方法，分别用于向量（或列表）进行学生对象的****插入、移除、修改、排序（按成绩升序）、浏览、查找（按姓名）、统计人数****等操作****。请对这些方法进行必要的测试，其中，查找到某个学生后将其成绩修正为99分。**

```java
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
 
class Student implements Comparable<Student>{
    private String id;
    private String name;
    private String gender;
    private double grade;
 
    // 构造方法
    public Student(String id, String name, String gender, double grade) {
        this.id = id;
        this.name = name;
        this.gender = gender;
        this.grade = grade;
    }
 
    // get 方法
    public String getId() {
        return id;
    }
 
    public String getName() {
        return name;
    }
 
    public String getGender() {
        return gender;
    }
 
    public double getGrade() {
        return grade;
    }
 
    // set 方法
    public void setId(String id) {
        this.id = id;
    }
 
    public void setName(String name) {
        this.name = name;
    }
 
    public void setGender(String gender) {
        this.gender = gender;
    }
 
    public void setGrade(double grade) {
        this.grade = grade;
    }
    public String toString() {
        return "Student{" +
                "id='" + id + '\'' +
                ", name='" + name + '\'' +
                ", gender='" + gender + '\'' +
                ", grade=" + grade +
                '}';
    }
    public int compareTo(Student other) {
        return Double.compare(this.grade, other.grade);
    }
}
class StudentManager {
    private List<Student> students;
 
    public StudentManager() {
        students = new ArrayList<>();
    }
 
    public void add(Student stu) {
        students.add(stu);
    }
 
    public Student remove(int i) {
        if (i >= 0 && i < students.size()) {
            return students.remove(i);
        }
        return null;
    }
 
    public boolean remove(Student stu) {
        return students.remove(stu);
    }
 
    public Student get(int i) {
        if (i >= 0 && i < students.size()) {
            return students.get(i);
        }
        return null;
    }
 
    public void set(int i, Student stu) {
        if (i >= 0 && i < students.size()) {
            students.set(i, stu);
        }
    }
 
    public void sort() {
        Collections.sort(students);
    }
 
    public void display() {
        for (Student student : students) {
            System.out.println(student);
        }
    }
 
    public Student search(String name) {
        for (Student student : students) {
            if (student.getName().equals(name)) {
                return student;
            }
        }
        return null;
    }
 
    public int size() {
        return students.size();
    }
 
}
public class SchoolActivity {
 
    public static void main(String[] args) {
        // TODO Auto-generated method stub
        StudentManager manager = new StudentManager();
        // 添加学生
        manager.add(new Student("2024001", "Alice", "Female", 90.5));
        manager.add(new Student("2024002", "Bob", "Male", 85.0));
        manager.add(new Student("2024003", "Charlie", "Male", 92.0));
 
        // 显示学生列表
        System.out.println("学生列表:");
        manager.display();
 
        // 按成绩排序
        manager.sort();
        System.out.println("\n按成绩排序后的学生列表:");
        manager.display();
 
        // 查找学生并修改成绩
        Student student = manager.search("Bob");
        if (student != null) {
            student.setGrade(99.0);
        }
        System.out.println("\n修改成绩后的学生列表:");
        manager.display();
 
        // 移除学生
        manager.remove(1);
        System.out.println("\n移除学生后的学生列表:");
        manager.display();
 
        // 统计学生人数
        System.out.println("\n学生人数: " + manager.size());
    }
 
}
```



1、UTF8编码的文本文件[sportman.txt](https://learn.upc.edu.cn/meol/common/ckeditor/openfile.jsp?id=DBCPDDDBDFDADBDGDECPHDHAGPHCHEGNGBGOCOHEHIHE)（点击链接可下载文件）的内容如下图所示：

![image.png](https://learn.upc.edu.cn/meol/common/ckeditor/openfile.jsp?id=DBCPDDDBDFDADBDGDBCPGJGNGBGHGFCOHAGOGH)

每行数据中包括运动员姓名、身高和体重信息，各数据项用空白分隔。

（1）设计一个表示运动员的Sportman类，具有构造方法、setter、getter和toString方法；

（2）读取该文件，并将每个运动员的数据以Sportman对象的形式存入一个向量中；

（3）按体重升序输出向量中运动员数据，格式如下（各数据项用一个Tab符分隔）：

![image.png](https://learn.upc.edu.cn/meol/common/ckeditor/openfile.jsp?id=DBCPDDDBDFDADBDFDACPGJGNGBGHGFCOHAGOGH)

```java
package b;
 
// 定义 Sportman 类，用于表示运动员
public class Sportman {
    private String name; // 运动员的名字
    private double height; // 运动员的身高（单位：cm）
    private double weight; // 运动员的体重（单位：kg）
 
    // 构造方法，初始化运动员的名字、身高和体重
    public Sportman(String name, double height, double weight) {
        this.name = name;
        this.height = height;
        this.weight = weight;
    }
 
    // 获取运动员名字的方法
    public String getName() {
        return name;
    }
 
    // 设置运动员名字的方法
    public void setName(String name) {
        this.name = name;
    }
 
    // 获取运动员身高的方法
    public double getHeight() {
        return height;
    }
 
    // 设置运动员身高的方法
    public void setHeight(double height) {
        this.height = height;
    }
 
    // 获取运动员体重的方法
    public double getWeight() {
        return weight;
    }
 
    // 设置运动员体重的方法
    public void setWeight(double weight) {
        this.weight = weight;
    }
 
    // 重写 toString 方法，用于输出运动员的基本信息
    @Override
    public String toString() {
        return name + "\t" + height + "\t" + weight; // 用制表符分隔运动员的名字、身高和体重
    }
}
package b;
 
import java.io.*; // 导入输入输出相关的类
import java.util.*; // 导入集合框架相关的类
 
// 定义 SportmanManager 类，用于管理运动员信息
public class SportmanManager {
    public static void main(String[] args) {
        List<Sportman> sportmen = new ArrayList<>(); // 创建一个列表来存储运动员信息
        String filePath = "D://sportman.txt"; // 设置文件路径，确保路径正确
 
        // 读取文件中的运动员数据
        try (BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), "UTF-8"))) {
            String line;
            boolean isFirstLine = true; // 用于标记是否为文件的第一行
 
            // 逐行读取文件内容
            while ((line = br.readLine()) != null) {
                // 如果是第一行，跳过（表头）
                if (isFirstLine) {
                    isFirstLine = false;
                    continue; // 跳过这一行
                }
 
                // 按空白字符分割每一行的数据
                String[] data = line.trim().split("\\s+");
                // 确保数据格式正确，长度为3
                if (data.length == 3) {
                    try {
                        String name = data[0]; // 获取运动员的名字
                        double height = Double.parseDouble(data[1]); // 获取身高并转换为 double
                        double weight = Double.parseDouble(data[2]); // 获取体重并转换为 double
                        // 将运动员信息添加到列表中
                        sportmen.add(new Sportman(name, height, weight));
                    } catch (NumberFormatException e) {
                        System.err.println("无效数据格式: " + line); // 输出格式错误的行
                    }
                } else {
                    System.err.println("数据格式错误: " + line); // 输出行数据格式错误的信息
                }
            }
        } catch (IOException e) {
            System.err.println("读取文件时发生错误: " + e.getMessage()); // 输出文件读取错误信息
        }
 
        // 按照体重升序排序运动员列表
        sportmen.sort(Comparator.comparingDouble(Sportman::getWeight));
        // 输出表头
        System.out.println("序号\t姓名\t身高(cm)\t体重(kg)");
 
        // 输出每位运动员的信息，并添加序号
        for (int i = 0; i < sportmen.size(); i++) {
            System.out.println((i + 1) + "\t" + sportmen.get(i));
        }
    }
}
```

2、编写static boolean fileCopy(String src, String dest)方法，其中使用IO流实现任意文件的复制。

要求及提示如下：

（1）基本步骤：

​    a) 打开源文件，创建目标文件；

​    b) 从源文件读取数据块，并写入目标文件；

​    c) 关闭源文件和目标文件。

（2）不用关心文件的内容具体含义，所有的内容都以0、1序列即字节对待即可。

（3）复制成功（未发生异常）则返回true，否则返回false。

（4）关键技术：

​     a) 使用FileInputStream和FileOutputStream实现文件的读与写；

​     b) 为提高效率，使用BufferedInputStream和BufferedOutputStream对输入输出字节流进行缓冲。

```java
package c; // 定义包名
 
import java.io.*; // 导入输入输出流相关的类
 
public class FileUtil {
 
    // 静态方法，复制文件
    public static boolean fileCopy(String src, String dest) {
        // 使用 try-with-resources 语法来自动关闭资源，确保在使用后释放文件流
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(src));
             BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(dest))) {
 
            byte[] buffer = new byte[1024]; // 创建一个大小为1024字节的缓冲区
            int bytesRead; // 用于存储每次读取的字节数
 
            // 循环读取源文件数据并写入目标文件
            while ((bytesRead = bis.read(buffer)) != -1) {
                // 将读取的数据写入目标文件，buffer中存储的是读取的内容
                bos.write(buffer, 0, bytesRead); // 写入的字节数是bytesRead
            }
 
            return true; // 如果文件复制完成，返回true表示成功
        } catch (IOException e) {
            e.printStackTrace(); // 输出异常信息，便于调试
            return false; // 如果发生异常，返回false表示失败
        }
    }
 
    public static void main(String[] args) {
        // 定义源文件路径和目标文件路径
        String sourceFile = "D://周杰伦 - 青花瓷.txt"; // 源文件路径
        String destFile = "D://目标文件.txt"; // 目标文件路径
 
        // 调用文件复制方法并输出结果
        boolean result = fileCopy(sourceFile, destFile);
        System.out.println("文件复制成功: " + result); // 输出复制结果
    }
}
```

3、（选做）编写一个程序，用于加密或解密文件，算法可以采用简单的字节取反或字节中某个比特或若干位取反。 

​    提示: X = (byte) (~X) 实现字节取反，

​         X = (byte)(X^0x80) 实现字节的最高（即第1个）比特翻转，

​         其中，X是byte变量。

```java
package d;
import java.io.*;
 
public class FileEncryptor {
 
    // 加密或解密文件
    public static boolean encryptDecryptFile(String src, String dest, boolean encrypt) {
        // 使用 try-with-resources 语法来自动关闭资源
        try (BufferedInputStream bis = new BufferedInputStream(new FileInputStream(src));
             BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(dest))) {
 
            int byteData; // 用于存储每次读取的字节
 
            // 循环读取源文件的每一个字节
            while ((byteData = bis.read()) != -1) {
                // 根据选择的操作（加密或解密）处理字节
                if (encrypt) {
                    // 加密：字节取反
                    byteData = ~byteData;
                } else {
                    // 解密：字节取反
                    byteData = ~byteData;
                }
 
                // 写入处理后的字节到目标文件
                bos.write(byteData);
            }
 
            return true; // 如果文件处理完成，返回true表示成功
        } catch (IOException e) {
            e.printStackTrace(); // 输出异常信息，便于调试
            return false; // 如果发生异常，返回false表示失败
        }
    }
 
    public static void main(String[] args) {
        // 定义源文件路径和目标文件路径
        String sourceFile = "D://源文件.txt"; // 源文件路径
        String destFile = "D://加密后的文件.txt"; // 加密后的文件路径
 
        // 调用加密方法并输出结果
        boolean encryptResult = encryptDecryptFile(sourceFile, destFile, true);
        System.out.println("文件加密成功: " + encryptResult);
 
        // 定义解密后的文件路径
        String decryptedFile = "D://解密后的文件.txt"; // 解密后的文件路径
 
        // 调用解密方法并输出结果
        boolean decryptResult = encryptDecryptFile(destFile, decryptedFile, false);
        System.out.println("文件解密成功: " + decryptResult);
    }
}
```

3、在第1题的基础上，修改界面和程序，进一步丰富文本编辑器功能，要求：

​     1）加上菜单栏，其中文件菜单包括新建、打开、保存、另存为、退出等菜单项（提示：可以设置一个窗体变量filename，打开一个文件后filename存储文件绝对路径，新建文件时filename赋值为null，保存文件时filename为null则先弹出保存文件对话框，然后保存，否则直接保存，“另存为”则总是显示保存对话框后保存。只要显示了保存文件对话框，filename就更新为新的名字。）

   2）打开/保存文件时，提供打开/保存文件对话框。

   3）关闭窗体时，如果文档没有保存，提示用户是否要退出程序。

   4）尽你的能力，尝试提供更多的其他一些功能（例如字体设置、字号缩放、颜色设置、查找替换等）

```java
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import javax.swing.text.Document;
 
import java.io.*;
 
public class HW extends JFrame {
 
    private JPanel contentPane;
    private JTextArea ta;
    private JScrollPane scrollPane;
    private String filename = null;
    private boolean isTextModified = false;
    private final static String APP_NAME = "简易文本编辑器";
 
    public static void main(String[] args) {
        EventQueue.invokeLater(() -> {
            try {
                HW frame = new HW();
                frame.setVisible(true);
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }
 
    public HW() {
        super(APP_NAME);
        setDefaultCloseOperation(JFrame.DO_NOTHING_ON_CLOSE);
        setBounds(100, 100, 800, 600);
        addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent e) {
                exitProgram();
            }
        });
 
        contentPane = new JPanel();
        contentPane.setBorder(new EmptyBorder(5, 5, 5, 5));
        contentPane.setLayout(new BorderLayout(0, 0));
        setContentPane(contentPane);
 
        // Create a menu bar
        JMenuBar menuBar = new JMenuBar();
        setJMenuBar(menuBar);
 
        // File menu
        JMenu fileMenu = new JMenu("文件");
        menuBar.add(fileMenu);
 
        JMenuItem newItem = new JMenuItem("新建");
        newItem.addActionListener(e -> newFile());
        fileMenu.add(newItem);
 
        JMenuItem openItem = new JMenuItem("打开");
        openItem.addActionListener(e -> openFile());
        fileMenu.add(openItem);
 
        JMenuItem saveItem = new JMenuItem("保存");
        saveItem.addActionListener(e -> saveFile());
        fileMenu.add(saveItem);
 
        JMenuItem saveAsItem = new JMenuItem("另存为");
        saveAsItem.addActionListener(e -> saveFileAs());
        fileMenu.add(saveAsItem);
 
        fileMenu.addSeparator();
 
        JMenuItem exitItem = new JMenuItem("退出");
        exitItem.addActionListener(e -> exitProgram());
        fileMenu.add(exitItem);
 
        // Edit menu
        JMenu editMenu = new JMenu("编辑");
        menuBar.add(editMenu);
 
        JMenuItem copyItem = new JMenuItem("复制");
        copyItem.addActionListener(e -> ta.copy());
        editMenu.add(copyItem);
 
        JMenuItem pasteItem = new JMenuItem("粘贴");
        pasteItem.addActionListener(e -> ta.paste());
        editMenu.add(pasteItem);
 
        JMenuItem cutItem = new JMenuItem("剪切");
        cutItem.addActionListener(e -> ta.cut());
        editMenu.add(cutItem);
 
        JMenuItem findItem = new JMenuItem("查找");
        findItem.addActionListener(e -> findText());
        editMenu.add(findItem);
 
        editMenu.addSeparator();
 
        JMenuItem clearItem = new JMenuItem("清空");
        clearItem.addActionListener(e -> ta.setText(""));
        editMenu.add(clearItem);
 
        // Settings menu
        JMenu settingsMenu = new JMenu("设置");
        menuBar.add(settingsMenu);
 
        JMenuItem fontItem = new JMenuItem("字体设置");
        fontItem.addActionListener(e -> setFontSettings());
        settingsMenu.add(fontItem);
 
        JMenuItem colorItem = new JMenuItem("字体颜色");
        colorItem.addActionListener(e -> setFontColor());
        settingsMenu.add(colorItem);
 
        // Text area with scroll pane
        ta = new JTextArea();
        ta.setFont(new Font("宋体", Font.PLAIN, 16));
        ta.setLineWrap(true);
        ta.setWrapStyleWord(true);
 
        // 添加鼠标滚轮事件监听器
        ta.addMouseWheelListener(e -> {
            if (e.isControlDown()) {   // 检查 Ctrl 键是否按下
                Font currentFont = ta.getFont();
               // 获取文本区域当前的字体对象。
                int currentSize = currentFont.getSize();
                // 获取当前字体的字号。
                int newSize = e.getWheelRotation() < 0 ? currentSize + 1 : currentSize - 1; // 滚轮向上放大，向下缩小
                // 根据鼠标滚轮的滚动方向计算新的字号：
                // 如果滚轮向上滚动（e.getWheelRotation() < 0），字号加1；
                // 如果滚轮向下滚动，字号减1。
                newSize = Math.max(newSize, 10); // 防止字号过小
                ta.setFont(new Font(currentFont.getName(), Font.PLAIN, newSize));
            }
        });
 
        scrollPane = new JScrollPane(ta);
        contentPane.add(scrollPane, BorderLayout.CENTER);
 
        Document doc = ta.getDocument();
        doc.addDocumentListener(new DocumentListener() {
            @Override
            public void insertUpdate(DocumentEvent e) {
                markAsModified();
            }
 
            @Override
            public void removeUpdate(DocumentEvent e) {
                markAsModified();
            }
 
            @Override
            public void changedUpdate(DocumentEvent e) {
                markAsModified();
            }
        });
    }
 
    private void newFile() {
        if (confirmSaveIfModified()) {
            ta.setText("");
            filename = null;
            isTextModified = false;
            setTitle(APP_NAME);
        }
    }
 
    private void openFile() {
        if (!confirmSaveIfModified()) return;
 
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("打开文件");
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                ta.read(reader, null);
                filename = file.getAbsolutePath();
                isTextModified = false;
                setTitle(APP_NAME + " - " + filename);
                markAsModified();
            } catch (IOException e) {
                JOptionPane.showMessageDialog(this, "无法打开文件: " + e.getMessage(), "错误", JOptionPane.ERROR_MESSAGE);
            }
        }
    }
 
    private void saveFile() {
        if (filename == null) {
            saveFileAs();
        } else {
            saveToFile(new File(filename));
        }
    }
 
    private void saveFileAs() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setDialogTitle("另存为");
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            File file = fileChooser.getSelectedFile();
            saveToFile(file);
        }
    }
 
    private void saveToFile(File file) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            ta.write(writer);
            filename = file.getAbsolutePath();
            isTextModified = false;
            setTitle(APP_NAME + " - " + filename);
            JOptionPane.showMessageDialog(this, "文件已保存", "提示", JOptionPane.INFORMATION_MESSAGE);
        } catch (IOException e) {
            JOptionPane.showMessageDialog(this, "无法保存文件: " + e.getMessage(), "错误", JOptionPane.ERROR_MESSAGE);
        }
    }
 
    private boolean confirmSaveIfModified() {
        if (isTextModified) {
            int choice = JOptionPane.showConfirmDialog(this, "文档已修改，是否保存？", "提示", JOptionPane.YES_NO_CANCEL_OPTION);
            if (choice == JOptionPane.CANCEL_OPTION) return false;
            if (choice == JOptionPane.YES_OPTION) saveFile();
        }
        return true;
    }
 
    private void exitProgram() {
        if (confirmSaveIfModified()) {
            System.exit(0);
        }
    }
 
    private void markAsModified() {
        if (!isTextModified) {
            setTitle(APP_NAME + " - " + (filename == null ? "未命名" : filename) + " (*)");
            isTextModified = true;
        }
    }
 
    private void setFontSettings() {
        Font currentFont = ta.getFont();
        String fontName = JOptionPane.showInputDialog(this, "请输入字体名称：", currentFont.getFontName());
        String fontSizeStr = JOptionPane.showInputDialog(this, "请输入字号：", currentFont.getSize());
        if (fontName != null && fontSizeStr != null) {
            try {
                int fontSize = Integer.parseInt(fontSizeStr);
                ta.setFont(new Font(fontName, Font.PLAIN, fontSize));
            } catch (NumberFormatException e) {
                JOptionPane.showMessageDialog(this, "无效的字号", "错误", JOptionPane.ERROR_MESSAGE);
            }
        }
    }
 
    private void setFontColor() {
        Color chosenColor = JColorChooser.showDialog(this, "选择字体颜色", ta.getForeground());
        if (chosenColor != null) {
            ta.setForeground(chosenColor);
        }
    }
 
    private void findText() {
        String searchTerm = JOptionPane.showInputDialog(this, "请输入要查找的文本：");
        if (searchTerm != null && !searchTerm.isEmpty()) {
            String content = ta.getText();
            int index = content.indexOf(searchTerm);
            if (index >= 0) {
                ta.setCaretPosition(index);
                ta.select(index, index + searchTerm.length());
                ta.requestFocusInWindow();
            } else {
                JOptionPane.showMessageDialog(this, "未找到指定文本", "提示", JOptionPane.INFORMATION_MESSAGE);
            }
        }
    }
}
```


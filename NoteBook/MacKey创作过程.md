### MacKey创作过程

程序源码

```python
import os
import threading
from pynput import keyboard
from playsound import playsound
import sys

# 获取正确的音频文件路径（支持 PyInstaller 打包）
def get_sound_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS  # PyInstaller 打包后文件所在目录
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, "1.wav")

# 设置音效文件路径
sound_file = get_sound_path()

def play_sound():
    try:
        playsound(sound_file)
    except Exception as e:
        print(f"Error playing sound: {e}")

def on_press(key):
    # 使用线程播放音效
    threading.Thread(target=play_sound).start()

def main():
    # 创建一个监听器
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

if __name__ == "__main__":
    main()

```

终端安装pyinstaller, 然后将音效文件放在源码同一目录下之后先cd进目录, 然后在终端运行

```
pyinstaller --onefile --add-data "1.wav:." bodyFile.py
```

就可以在目录里找到exe文件了


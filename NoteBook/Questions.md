# Problems In Soft&HardWare

## Vidhub使用

- 使用时应先建立好文件夹并梳理好文件关系
- 下载的`.torrent`文件会被自动分类,但是是按名称来进行划分.比如同样的蓝箱第一季第五集,名字叫做` Nekomoe kissaten&LoliHouse] Ao no Hako - 05 [WebRip 1080p HEVC-10bit AAC ASSx2].mkv`时被单独排在外面,但是仿照前面几个字幕组的命名方法改为` KitaujiSub] Ao no Hako [04][WebRip][HEVC_AAC][CHS_JP&CHT_JP].mkv`后就直接进入同一目录了.
- 如果不小心将影片从Media Center删除了只需要重命名一下就可以回来(左上角有刷新键).
- 有时会出现报错` Playback failed, network connection failed / unsupported format / broken file.`只需要重新导入`animation`文件夹即可.





## FDM(Free Download Machine)的使用

速度慢的时候可以去` Github`复制一点` traker`粘贴进来,以提供更多的服务器.



## 如何在设置中设置允许任何软件安装

在新版本中MacOS已经将“anywhere”选项给隐藏, 由此需要在terminal中输入

`````
sudo spctl --master-disable
`````

 然后再打开设置找到相应位置,就会显示被隐藏的“anywhere”

## IPad配对的蓝牙键盘失灵情况的解决
IPad连接的蓝牙键盘可能会与assistanttouch起冲突，这个时候只需要关闭这一项，删除蓝牙键盘的配对，重启一下IPad再重新连接一般就好了。
还有一个情况是Esc键会被改为home键以弥补无法回到桌面的问题，所以我们可以在Hardware Keyboard设置里面将opt的Modifier Keys为Esc（既然平时不怎么用）

## MacBook上使用rar进行压缩和解压缩
使用homebrew 下载rar:
```
brew install rar
```
然后等到下载完成之后即可在terminal中使用rar进行压缩(解压之前下载的the unarchiver就可以了), 具体做法为(也可以在terminal输入`rar`查看用法):
- 首先我们在桌面建立一个文件夹, 并使用cd命令进入此文件夹(方便找到压缩后的文件)
- 将文件聚合成一个压缩包:
	- 进入文件夹目录之后输入命令`rar a 压缩包名称.rar 源文件名称` 如果有多个文件直接每个之间隔一个空格即可.
- 将已有的文件夹压缩为压缩包
	- 进入文件夹以后输入命令`rar a 压缩包名称.rar 源文件夹名称/`.(必须有这个/, 否则无法成功压缩)


## 使用HandBrake
使用homebrew下载handbrake:
```
brew install --cask handbrake
```
然后进行配置:
先进入summary选项卡, 将格式调整为mp4, 然后勾选web optimised, align A/V start. 还有一个选项叫做passthru common metadata, 勾选的话可以保留原视频的一些其他信息, 比如title, subtitle之类的, 没什么用, 可以不勾选
![[handbrake Config1.png]]
然后进入dimensions, 将cropping改为none(或者custom, 然后将四周改为0, 应该是一样的效果(?)). 再根据视频的dimension改相关参数. 
![[handbrake Config2.png]]
最后到video, 在constant quality压缩, 一般调到22或35就够了, 数字越大体积越小, 不过损耗也会变大. 然后调节encoding option, 将preset调为slow, 这一点教程上说是压缩速度的快慢, 我查了一下应该是指编码器的编码速度, slow的编码速度慢一点但是体积更小.
最后可以将这些都保存为preset, 不过这些都不重要就是了
![[handbrake Config3.png]]
## 使用PillOCR
使用doubao-1.5-vision-lite-250315, 这个名字不能错, 否则无法调用API会显示红色pill

Account:`api-key-20250515163818`
Password:`235ac232-2f5d-4ca2-90bc-ba9f9253c64e`
还有一点就是我之前将截屏的保存位置设为了`\Desktop\Grocery\ScreenShot`, 所以不会直接进入剪贴板. 快捷键多按一个`control`可以让截图或截屏只进入剪贴板而不保存.

## 如何通过编辑HTML修改网页(粘贴)
- 进入inspect
- 定位文本框
- 尝试直接插入文本(在标签之间)
	- 若成功
		- 结束
	- 若失败
		- 记下id, 进入console操作
		- 使用指令
```
document.getElementById("q1_0").value = `copy text`;
```
也可以使用“, 但是不能换行

### 如何配置使用im-select
im-select依赖于fcitx-remote-for-osx, 可以使用homebrew下载.
下载后terminal输入`brew list fcitx-remote-for-osx`可以得到下载得到的内容以及路径.然后配置PATH变量:
```
echo '/opt/homebrew/Cellar/fcitx-remote-for-osx/0.4.0/bin/fcitx-remote' >> ~/.zshrc

source ~/.zshrc
```
最后再加个别名, 这样就可以方便的调用了
```
echo 'alias im-select="/opt/homebrew/Cellar/fcitx-remote-for-osx/0.4.0/bin/fcitx-remote"' >> ~/.zshrc

source ~/.zshrc
```



# TECHNIC PROBLEMS IN LEARNING

## 解决SSL证书问题

#### 起因

写爬虫的时候下载网页的时候总是报错

```java
import urllib.request as request

def download(url : str):
    response = request.urlopen(url)
    result = response.read()
    return result
url = "https://fanyi.youdao.com/#/TextTranslate"
htmltext = download(url)
print(htmltext)
```

这是报错

```
Traceback (most recent call last):
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/urllib/request.py", line 1344, in do_open
    h.request(req.get_method(), req.selector, req.data, headers,
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 1331, in request
    self._send_request(method, url, body, headers, encode_chunked)
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 1377, in _send_request
    self.endheaders(body, encode_chunked=encode_chunked)
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 1326, in endheaders
    self._send_output(message_body, encode_chunked=encode_chunked)
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 1085, in _send_output
    self.send(msg)
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 1029, in send
    self.connect()
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/http/client.py", line 1472, in connect
    self.sock = self._context.wrap_socket(self.sock,
                ==^^====^^====^^====^^====^^====^^==
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/ssl.py", line 455, in wrap_socket
    return self.sslsocket_class._create(
           ==^^====^^====^^====^^====^==
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/ssl.py", line 1042, in _create
    self.do_handshake()
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/ssl.py", line 1320, in do_handshake
    self._sslobj.do_handshake()
ssl.SSLCertVerificationError: [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/Users/horatius/Desktop/experiment/PycharmProjects/pythonProject/data acquisitation/chzero.py", line 8, in <module>
    htmltext = download(url)
               ==^^====^^==^
  File "/Users/horatius/Desktop/experiment/PycharmProjects/pythonProject/data acquisitation/chzero.py", line 4, in download
    response = request.urlopen(url)
               ==^^====^^====^^==^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/urllib/request.py", line 215, in urlopen
    return opener.open(url, data, timeout)
           ==^^====^^====^^====^^====^^==^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/urllib/request.py", line 515, in open
    response = self._open(req, data)
               ==^^====^^====^^==^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/urllib/request.py", line 532, in _open
    result = self._call_chain(self.handle_open, protocol, protocol +
             ==^^====^^====^^====^^====^^====^^====^^====^^====^^==^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/urllib/request.py", line 492, in _call_chain
    result = func(*args)
             ==^^====^==
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/urllib/request.py", line 1392, in https_open
    return self.do_open(http.client.HTTPSConnection, req,
           ==^^====^^====^^====^^====^^====^^====^^==^^^^
  File "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/urllib/request.py", line 1347, in do_open
    raise URLError(err)
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)>
```

ChatGPT的解释是我的Python环境试图通过HTTP协议访问URL时SSL证书验证失败,具体原因是无法获取本地发行者证书

后来实验发现是因为我的系统(或者Python?)安装缺少必要的SSL证书文件导致的.

随后尝试在terminal更新SSL证书文件:

```
/Applications/Python\ 3.12/Install\ Certificates.command
```

`/Applications/Python\ 3.12`是Python的安装地址

```
 -- pip install --upgrade certifi
Requirement already satisfied: certifi in /Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages (2024.6.2)
WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)'))': /simple/certifi/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)'))': /simple/certifi/
WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)'))': /simple/certifi/
WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)'))': /simple/certifi/
WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)'))': /simple/certifi/
Could not fetch URL https://pypi.org/simple/certifi/: There was a problem confirming the ssl certificate: HTTPSConnectionPool(host='pypi.org', port=443): Max retries exceeded with url: /simple/certifi/ (Caused by SSLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1000)'))) - skipping
 -- removing any existing file or link
 -- creating symlink to certifi certificate bundle
 -- setting permissions
 -- update complete
```

随后报错这样,显示Python无法验证SSL证书, 于是尝试手动下载安装证书.

- 访问` https://pypi.org/simple/certifi/`并下载最新的` .whl`文件
- 进入终端, 进入下载到的那个目录: ` cd ~/Downloads`
- 使用pip安装` .whl`文件: ` pip install pip install certifi-1.0.1-py2.py3-none-any.whl`
- 后来我重看的时候发现我应该是有SSL证书认证的,因为有一段终端是这样的: 

```
Processing ./certifi-1.0.1-py2.py3-none-any.whl
Installing collected packages: certifi
  Attempting uninstall: certifi
    Found existing installation: certifi 2024.6.2
    Uninstalling certifi-2024.6.2:
      Successfully uninstalled certifi-2024.6.2
```

- 下载了新的,但是提示不满足requests库的要求
- 于是升级: ` pip install --upgrade certifi`
- 升级后的结果: 

```
Successfully uninstalled certifi-1.0.1
Successfully installed certifi-2024.8.30
```

- 也就是说我最后相当于绕了一圈更新了certifi库? 真的也是十分无语了

不过最后圆满解决!

Look!

```java
import urllib.request as request

def download(url : str):
    response = request.urlopen(url)
    result = response.read()
    return result
url = "https://fanyi.youdao.com/#/TextTranslate"
htmltext = download(url)
print(htmltext)
```

new result:

```
b'<!DOCTYPE html><html lang="" class="light"><head><meta charset="utf-8"><meta http-equiv="X-UA-Compatible" content="IE=edge"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="keywords" content="\xe5\x9c\xa8\xe7\xba\xbf\xe5\x8d\xb3\xe6\x97\xb6\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81\xe5\x85\x8d\xe8\xb4\xb9AI\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81\xe6\x96\x87\xe6\xa1\xa3\xe6\x96\x87\xe6\x9c\xac\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81\xe4\xba\xba\xe5\xb7\xa5\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81\xe7\xbd\x91\xe9\xa1\xb5\xe7\xbf\xbb\xe8\xaf\x91."><meta name="description" content="\xe6\x9c\x89\xe9\x81\x93\xe7\xbf\xbb\xe8\xaf\x91\xe6\x8f\x90\xe4\xbe\x9b\xe5\x8d\xb3\xe6\x97\xb6\xe5\x85\x8d\xe8\xb4\xb9\xe7\x9a\x84\xe4\xb8\xad\xe6\x96\x87\xe3\x80\x81\xe8\x8b\xb1\xe8\xaf\xad\xe3\x80\x81\xe6\x97\xa5\xe8\xaf\xad\xe3\x80\x81\xe9\x9f\xa9\xe8\xaf\xad\xe3\x80\x81\xe6\xb3\x95\xe8\xaf\xad\xe3\x80\x81\xe5\xbe\xb7\xe8\xaf\xad\xe3\x80\x81\xe4\xbf\x84\xe8\xaf\xad\xe3\x80\x81\xe8\xa5\xbf\xe7\x8f\xad\xe7\x89\x99\xe8\xaf\xad\xe3\x80\x81\xe8\x91\xa1\xe8\x90\x84\xe7\x89\x99\xe8\xaf\xad\xe3\x80\x81\xe8\xb6\x8a\xe5\x8d\x97\xe8\xaf\xad\xe3\x80\x81\xe5\x8d\xb0\xe5\xb0\xbc\xe8\xaf\xad\xe3\x80\x81\xe6\x84\x8f\xe5\xa4\xa7\xe5\x88\xa9\xe8\xaf\xad\xe3\x80\x81\xe8\x8d\xb7\xe5\x85\xb0\xe8\xaf\xad\xe3\x80\x81\xe6\xb3\xb0\xe8\xaf\xad\xe5\x85\xa8\xe6\x96\x87\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81\xe7\xbd\x91\xe9\xa1\xb5\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81\xe6\x96\x87\xe6\xa1\xa3\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81PDF\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81DOC\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81PPT\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81\xe4\xba\xba\xe5\xb7\xa5\xe7\xbf\xbb\xe8\xaf\x91\xe3\x80\x81\xe5\x90\x8c\xe4\xbc\xa0\xe7\xad\x89\xe6\x9c\x8d\xe5\x8a\xa1\xe3\x80\x82"><meta property="og:image" content="https://ydlunacommon-cdn.nosdn.127.net/322e3486ea5a517c2cba3b76864dadba.png"><link rel="icon" href="https://ydlunacommon-cdn.nosdn.127.net/31cf4b56e6c0b3af668aa079de1a898c.png"><script src="//bat.bing.com/bat.js" async=""></script><script src="//bat.bing.com/bat.js" async=""></script><script>(function (w, d, t, r, u) {\n        var f, n, i;\n        (w[u] = w[u] || []),\n          (f = function () {\n            var o = { ti: "97136777", enableAutoSpaTracking: true };\n            (o.q = w[u]), (w[u] = new UET(o)), w[u].push("pageLoad");\n          }),\n          (n = d.createElement(t)),\n          (n.src = r),\n          (n.async = 1),\n          (n.onload = n.onreadystatechange =\n            function () {\n              var s = this.readyState;\n              (s && s !== "loaded" && s !== "complete") ||\n                (f(), (n.onload = n.onreadystatechange = null));\n            }),\n          (i = d.getElementsByTagName(t)[0]),\n          i.parentNode.insertBefore(n, i);\n      })(window, document, "script", "//bat.bing.com/bat.js", "uetq");</script><script>function uet_report_conversion(keyfrom) {\n        window.uetq = window.uetq || [];\n        window.uetq.push("event", "log in", {"event category": keyfrom});\n      }</script><script>(function (w, d, t, r, u) {\n        var f, n, i;\n        (w[u] = w[u] || []),\n          (f = function () {\n            var o = { ti: "97136779", enableAutoSpaTracking: true };\n            (o.q = w[u]), (w[u] = new UET(o)), w[u].push("pageLoad");\n          }),\n          (n = d.createElement(t)),\n          (n.src = r),\n          (n.async = 1),\n          (n.onload = n.onreadystatechange =\n            function () {\n              var s = this.readyState;\n              (s && s !== "loaded" && s !== "complete") ||\n                (f(), (n.onload = n.onreadystatechange = null));\n            }),\n          (i = d.getElementsByTagName(t)[0]),\n          i.parentNode.insertBefore(n, i);\n      })(window, document, "script", "//bat.bing.com/bat.js", "uetq");</script><title>\xe6\x9c\x89\xe9\x81\x93\xe7\xbf\xbb\xe8\xaf\x91_\xe6\x96\x87\xe6\x9c\xac\xe3\x80\x81\xe6\x96\x87\xe6\xa1\xa3\xe3\x80\x81\xe7\xbd\x91\xe9\xa1\xb5\xe3\x80\x81\xe5\x9c\xa8\xe7\xba\xbf\xe5\x8d\xb3\xe6\x97\xb6\xe7\xbf\xbb\xe8\xaf\x91</title><script defer="defer" src="https://shared.ydstatic.com/dict/translation-website/0.4.6/js/chunk-vendors.92ee7c9a.js" type="module"></script><script defer="defer" src="https://shared.ydstatic.com/dict/translation-website/0.4.6/js/app.54cddb16.js" type="module"></script><link href="https://shared.ydstatic.com/dict/translation-website/0.4.6/css/chunk-vendors.6fc89b85.css" rel="stylesheet"><link href="https://shared.ydstatic.com/dict/translation-website/0.4.6/css/app.72125a92.css" rel="stylesheet"><script defer="defer" src="https://shared.ydstatic.com/dict/translation-website/0.4.6/js/chunk-vendors-legacy.d3cded41.js" nomodule=""></script><script defer="defer" src="https://shared.ydstatic.com/dict/translation-website/0.4.6/js/app-legacy.69746be8.js" nomodule=""></script><link rel="stylesheet" type="text/css" href="https://shared.ydstatic.com/dict/translation-website/0.4.6/css/853.481de091.css"><link rel="stylesheet" type="text/css" href="https://shared.ydstatic.com/dict/translation-website/0.4.6/css/233.cb6f39bc.css"><link rel="stylesheet" type="text/css" href="https://shared.ydstatic.com/dict/translation-website/0.4.6/css/aiTranslateV2.f15dbb76.css"><script src="https://bat.bing.com/p/action/97136779.js" type="text/javascript" async="" data-ueto="ueto_fa7f27eb88"></script><script src="https://bat.bing.com/p/action/97136777.js" type="text/javascript" async="" data-ueto="ueto_ea3742aa33"></script></head><body><noscript><strong>We\'re sorry but translation-website doesn\'t work properly without JavaScript enabled. Please enable it to continue.</strong></noscript><div id="app" data-v-app=""><div data-v-fa956cd2="" class="index isPrerender aiBorder os_Mac"><div data-v-23fdaeb7="" data-v-fa956cd2="" class="web-frame-container less-than-min-view-width" style="--abd7f146:1080; --7f70d552:108; --15e1316b:800;"><div data-v-23fdaeb7="" class="top-banner-outer-container"></div><div data-v-23fdaeb7="" class="web-frame-content-container"><div data-v-2d62dbe7="" data-v-23fdaeb7="" class="header-outer-container"><div data-v-2d62dbe7="" class="header-container"><div data-v-2d62dbe7="" class="logo-container"><i data-v-2d62dbe7="" class="icon logo"></i></div><div data-v-2d62dbe7="" class="header-inner-container"><div data-v-2d62dbe7="" class="header-content"><div data-v-2d62dbe7="" class="actions-container" style="position: absolute; top: 0px; right: 16px;"><div data-v-2d62dbe7="" class="action-item"><i data-v-2d62dbe7="" class="icon icon-aiboke"></i><span data-v-2d62dbe7="" class="text color_text_0">AI\xe6\x92\xad\xe5\xae\xa2</span><div data-v-2d62dbe7="" class="tag-container">\xe9\x99\x90\xe6\x97\xb6\xe5\x85\x8d\xe8\xb4\xb9</div></div><div data-v-2d62dbe7="" class="action-item"><i data-v-2d62dbe7="" class="icon icon-echo"></i><span data-v-2d62dbe7="" class="text color_text_0">AI\xe5\xa4\x96\xe6\x95\x99</span><div data-v-2d62dbe7="" class="more-info-container echo"><img data-v-2d62dbe7="" class="client" src="https://ydlunacommon-cdn.nosdn.127.net/a0f731b873f33744e9c1320bf2891a24.png"><div data-v-2d62dbe7="" class="client-click-area"></div></div></div><div data-v-2d62dbe7="" class="action-item"><i data-v-2d62dbe7="" class="icon icon-pc-download"></i><span data-v-2d62dbe7="" class="text color_text_0">\xe6\xa1\x8c\xe9\x9d\xa2\xe7\xab\xaf</span><div data-v-2d62dbe7="" class="tag-container">\xe5\x85\x8d\xe8\xb4\xb9\xe4\xb8\x8b\xe8\xbd\xbd</div><div data-v-2d62dbe7="" class="more-info-container"><img data-v-2d62dbe7="" class="client" src="https://ydlunacommon-cdn.nosdn.127.net/210979446755a4ba850e1db3d8b99860.png"><div data-v-2d62dbe7="" class="client-dowload-area"></div></div></div><div data-v-2d62dbe7="" class="action-item"><i data-v-2d62dbe7="" class="icon icon-enterprise"></i><span data-v-2d62dbe7="" class="text color_text_0">\xe4\xbc\x81\xe4\xb8\x9a\xe7\x89\x88</span></div><div data-v-2d62dbe7="" class="action-line"></div><div data-v-2d62dbe7="" class="action-item open-vip-dialog"><i data-v-2d62dbe7="" class="icon icon-vip"></i><span data-v-2d62dbe7="" class="text">\xe5\xbc\x80\xe9\x80\x9a\xe4\xbc\x9a\xe5\x91\x98</span></div><div data-v-2d62dbe7="" class="action-item open-login-page"><i data-v-2d62dbe7="" class="icon icon-user-avatar"></i><span data-v-2d62dbe7="" class="text color_text_0">\xe7\x99\xbb\xe5\xbd\x95</span></div></div></div></div></div></div><div data-v-23fdaeb7="" class="web-frame-inner-container"><div data-v-ba8b1018="" data-v-23fdaeb7="" class="sidebar-container"><div data-v-ba8b1018="" class="sidebar-inner-container"><div data-v-0295b654="" data-v-ba8b1018="" class="sidebar-menus-container"><div data-v-0295b654="" class="menus"><div data-v-0295b654="" class="menu-item active"><i data-v-0295b654="" class="icon icon-menu-text-translate"></i><div data-v-0295b654="" class="text">\xe6\x96\x87\xe6\x9c\xac\xe7\xbf\xbb\xe8\xaf\x91</div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-ai-translate"></i><div data-v-0295b654="" class="text">\xe6\xb6\xa6\xe8\x89\xb2\xe6\x94\xb9\xe5\x86\x99</div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-document-translate"></i><div data-v-0295b654="" class="text hot-desc" data-tips="\xe7\xa7\x92\xe7\xbf\xbbPDF">\xe6\x96\x87\xe6\xa1\xa3\xe7\xbf\xbb\xe8\xaf\x91</div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-ai-write"></i><div data-v-0295b654="" class="text">AI\xe5\x86\x99\xe4\xbd\x9c</div></div><div data-v-0295b654="" class="menu-item has-hover-tip"><div data-v-452c3873="" data-v-0295b654="" class="menu-item-trigger" style="--16733cba:154.5;"><i data-v-0295b654="" class="icon icon-menu-aippt"></i><div data-v-0295b654="" class="text hot-desc" data-tips="\xe5\x85\x8d\xe8\xb4\xb9\xe6\xa8\xa1\xe7\x89\x88">AI PPT</div><div data-v-452c3873="" class="sidebar-tip-container"><div data-v-650ba91c="" data-v-0295b654="" class="aippt-guide-tip-container"><div data-v-650ba91c="" class="aippt-guide-tip-inner-container"><img data-v-650ba91c="" class="client" src="https://ydlunacommon-cdn.nosdn.127.net/f4d2cfa629fc213821abf299eb32a5f4.png"></div></div></div></div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-arxiv"></i><div data-v-0295b654="" class="text">arXiv\xe8\xae\xba\xe6\x96\x87\xe7\xbf\xbb\xe8\xaf\x91</div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-web-page"></i><div data-v-0295b654="" class="text">\xe7\xbd\x91\xe9\xa1\xb5\xe7\xbf\xbb\xe8\xaf\x91</div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-file-format"></i><div data-v-0295b654="" class="text">PDF\xe8\xbd\xacWORD</div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-study"></i><div data-v-0295b654="" class="text">\xe6\x9c\x89\xe9\x81\x93\xe5\xad\xa6\xe6\x9c\xaf</div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-summary"></i><div data-v-0295b654="" class="text">\xe6\x9c\x89\xe9\x81\x93\xe9\x80\x9f\xe8\xaf\xbb</div></div><div data-v-0295b654="" class="menu-item has-hover-tip human-translate-menu-item"><div data-v-452c3873="" data-v-0295b654="" class="menu-item-trigger" style="--16733cba:352;"><i data-v-0295b654="" class="icon icon-menu-human-translate"></i><div data-v-0295b654="" class="text">\xe4\xba\xba\xe5\xb7\xa5\xe7\xbf\xbb\xe8\xaf\x91</div><div data-v-452c3873="" class="sidebar-tip-container"><div data-v-210af69a="" data-v-0295b654="" class="human-translation-guide-tip-container"><div class="human-translation-guide-tip" data-v-210af69a=""><div class="guide-tip-title" data-v-210af69a=""><span class="highlight" data-v-210af69a="">\xe7\xbd\x91\xe6\x98\x93\xe8\x87\xaa\xe8\x90\xa5</span><span data-v-210af69a="">\xe4\xba\xba\xe5\xb7\xa5\xe7\xbf\xbb\xe8\xaf\x91\xe6\x9c\x8d\xe5\x8a\xa1\xef\xbc\x8c\xe4\xb8\x93\xe4\xb8\x9a\xe3\x80\x81\xe7\xb2\xbe\xe5\x87\x86\xe3\x80\x81\xe5\x9c\xb0\xe9\x81\x93\xef\xbc\x81</span></div><div class="guide-tip-desc" data-v-210af69a=""> \xe7\xb2\xbe\xe9\x80\x89\xe5\x85\xa8\xe7\x90\x83\xe4\xb8\x8a\xe4\xb8\x87\xe5\x90\x8d\xe8\xaf\x91\xe5\x91\x98\xe5\xae\x9e\xe6\x97\xb6\xe5\xbe\x85\xe5\x91\xbd\xef\xbc\x8c\xe4\xb8\x93\xe5\xae\xb6\xe5\xae\xa1\xe6\xa0\xa1\xef\xbc\x8c\xe6\xaf\x8d\xe8\xaf\xad\xe6\xb6\xa6\xe8\x89\xb2\xe3\x80\x82\xe4\xb8\xa5\xe6\xa0\xbc\xe8\xb4\xa8\xe9\x87\x8f\xe6\x8a\x8a\xe6\x8e\xa7\xef\xbc\x8c\xe5\xa4\x9a\xe9\x87\x8d\xe5\x94\xae\xe5\x90\x8e\xe4\xbf\x9d\xe9\x9a\x9c\xef\xbc\x8c\xe8\xae\xa9\xe6\x82\xa8\xe7\xbf\xbb\xe8\xaf\x91\xe6\x97\xa0\xe5\xbf\xa7\xef\xbc\x81 </div><div class="guide-tip-intro-form" data-v-210af69a=""><div class="intro-form-item" data-v-210af69a=""><div class="intro-form-item-title" data-v-210af69a="">\xe5\xbf\xab\xe9\x80\x9f\xe7\xbf\xbb\xe8\xaf\x91</div><div class="line" data-v-210af69a=""></div><div class="intro-form-item-desc" data-v-210af69a="">\xe6\x9c\x80\xe5\xbf\xab1\xe5\x88\x86\xe9\x92\x9f<br data-v-210af69a="">\xe7\xab\x8b\xe7\xad\x89\xe5\x8f\xaf\xe5\x8f\x96</div></div><div class="line" data-v-210af69a=""></div><div class="intro-form-item" data-v-210af69a=""><div class="intro-form-item-title" data-v-210af69a="">\xe6\x96\x87\xe6\xa1\xa3\xe7\xbf\xbb\xe8\xaf\x91</div><div class="line" data-v-210af69a=""></div><div class="intro-form-item-desc" data-v-210af69a="">\xe7\xb2\xbe\xe5\x87\x86\xe5\x8c\xb9\xe9\x85\x8d\xe8\xaf\x91\xe5\x91\x98<br data-v-210af69a="">\xe5\xa4\x9a\xe9\x87\x8d\xe8\xb4\xa8\xe9\x87\x8f\xe6\x8a\x8a\xe6\x8e\xa7</div></div><div class="line" data-v-210af69a=""></div><div class="intro-form-item" data-v-210af69a=""><div class="intro-form-item-title" data-v-210af69a="">\xe6\xaf\x8d\xe8\xaf\xad\xe6\xb6\xa6\xe8\x89\xb2</div><div class="line" data-v-210af69a=""></div><div class="intro-form-item-desc" data-v-210af69a="">\xe6\xb6\xb5\xe7\x9b\x96\xe4\xb8\x8a\xe5\x8d\x83\xe5\xad\xa6\xe7\xa7\x91<br data-v-210af69a="">\xe6\xaf\x8d\xe8\xaf\xad\xe4\xb8\x93\xe5\xae\xb6\xe6\xb6\xa6\xe8\x89\xb2</div></div></div></div></div></div></div></div><div data-v-0295b654="" class="menu-item has-hover-tip echo-menu-item"><div data-v-452c3873="" data-v-0295b654="" class="menu-item-trigger" style="--16733cba:270;"><i data-v-0295b654="" class="icon icon-menu-echo"></i><div data-v-0295b654="" class="text">AI\xe5\x8f\xa3\xe8\xaf\xad\xe9\x99\xaa\xe7\xbb\x83</div><div data-v-452c3873="" class="sidebar-tip-container"><div data-v-042087d7="" data-v-0295b654="" class="echo-guide-tip-container"><div data-v-042087d7="" class="echo-guide-tip-inner-container"><img data-v-042087d7="" class="client" src="https://ydlunacommon-cdn.nosdn.127.net/2c99001b8878e302dbb456438a24a3e8.png"><div data-v-042087d7="" class="client-click-area"></div></div></div></div></div></div><div data-v-0295b654="" class="menu-item"><i data-v-0295b654="" class="icon icon-menu-translate-api"></i><div data-v-0295b654="" class="text">\xe7\xbf\xbb\xe8\xaf\x91API</div></div><div data-v-0295b654="" class="menu-item vip-menu-item"><i data-v-0295b654="" class="icon icon-menu-vip"></i><div data-v-0295b654="" class="text">\xe8\xb6\x85\xe7\xba\xa7\xe4\xbc\x9a\xe5\x91\x98</div></div></div></div></div><div data-v-ba8b1018="" class="sidebar-footer-container"><div data-v-ba8b1018="" class="feedback-container"><i data-v-ba8b1018="" class="icon icon-feedback"></i><span data-v-ba8b1018="" class="text">\xe6\x84\x8f\xe8\xa7\x81\xe5\x8f\x8d\xe9\xa6\x88</span></div><div data-v-ba8b1018="" class="collapsed-btn"><i data-v-ba8b1018="" class="icon icon-collapsed"></i><div data-v-ba8b1018="" class="collapse-hover-tip">\xe6\x94\xb6\xe8\xb5\xb7\xe4\xbe\xa7\xe8\xbe\xb9\xe6\xa0\x8f</div><!----></div></div></div><div data-v-23fdaeb7="" class="web-frame-content web-frame-content-ai"><div data-v-23fdaeb7="" class="web-frame-content-scroll-container ai-bg"><div data-v-23fdaeb7="" class="web-frame-inner-full-content"><div data-v-72d80a04="" data-v-fa956cd2="" class="tab-header tab-header-ai"><div data-v-72d80a04="" class="tab-left"><div data-v-72d80a04="" class="tab-item color_text_3"><span data-v-72d80a04="">\xe7\xbf\xbb\xe8\xaf\x91</span><!----><!----></div><div data-v-72d80a04="" class="tab-item active color_text_1 tab-item-ai gradient-item"><span data-v-72d80a04="">AI \xe7\xbf\xbb\xe8\xaf\x91</span><!----><!----></div><div data-v-72d80a04="" class="tab-item color_text_3"><span data-v-72d80a04="">\xe6\x96\x87\xe6\xa1\xa3\xe7\xbf\xbb\xe8\xaf\x91</span><!----><div data-v-72d80a04="" class="tip-text">\xe7\xa7\x92\xe7\xbf\xbbPDF</div></div><div data-v-72d80a04="" class="tab-item color_text_3"><span data-v-72d80a04="">PDF\xe8\xbd\xacWORD</span><!----><!----></div><div data-v-72d80a04="" class="tab-item color_text_3"><span data-v-72d80a04="">\xe4\xba\xba\xe5\xb7\xa5\xe7\xbf\xbb\xe8\xaf\x91</span></div><div data-v-72d80a04="" class="tab-item color_text_3"><span data-v-72d80a04="">AI+\xe4\xba\xba\xe5\xb7\xa5\xe7\xbf\xbb\xe8\xaf\x91</span></div></div><div data-v-72d80a04="" class="tab-right"></div></div><div data-v-d74f9006="" data-v-fa956cd2="" class="translate-tab-container ai-content ai-content-sidebar-unfold ctrl-icon-nav" style="width: calc(100% - 0px);"><div data-v-e96cb4ec="" data-v-d74f9006="" class="tab-header tab-header-ai"><div data-v-e96cb4ec="" class="tab-right"><!----><!----><!----></div></div><div data-v-d74f9006="" class="tab-body color_bg_3"><div data-v-d96f32dc="" data-v-d74f9006="" class="ai-v3"><div data-v-d96f32dc="" class="dialog-wrap"><div data-v-27cf4eac="" data-v-d96f32dc="" class="dialog-index-container dialog-index-wrap"><!----><div data-v-69bb50fe="" data-v-27cf4eac="" class="guide-box"><p data-v-69bb50fe="" class="guide-title">\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x8c\xe6\x88\x91\xe6\x98\xaf\xe6\x9c\x89\xe9\x81\x93\xe7\xbf\xbb\xe8\xaf\x91AI\xe5\x8a\xa9\xe6\x89\x8b</p><!----><div data-v-69bb50fe="" class="guide-examp"></div><span data-v-69bb50fe="" class="hidden-guide color_text_3">\xe9\x9a\x90\xe8\x97\x8f\xe7\xa4\xba\xe4\xbe\x8b</span></div></div><div data-v-d96f32dc="" class="prompt-index-wrap"><!----></div><!----></div><div data-v-d96f32dc="" class="history-list-wrap"><img data-v-65a9fde4="" class="svip-guide" src="https://shared.ydstatic.com/dict/translation-website/0.4.6/img/ic_ai_history_svip_guide.ee329c2a.png"><p data-v-65a9fde4="" class="new-convers disable-new"><img data-v-65a9fde4="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAAXNSR0IArs4c6QAAAARzQklUCAgICHwIZIgAAADaSURBVFiF7ZU9CsJAEIXfS+1hAmqvt1hMRL1AoidJ9AKJ+ENuob0KHiZ1xiJWkgFZQ4KwXzVs8eaDYWYBh+OfMfM4M0Gc/ZJB++abEaS61SneuDgkd5scz1YAIoPGujOBlnACTqB3AfUOzBbRUCp9vSqhTzAFAIGsPcpTbeKxPO93j68FTBBnIJaqtg2CvDhtV5/PzSOgWF9IFSVTbVSfWn0EIvBJSeqaGxLqCECW2qm2/wvC9QSQyztmWhzTq01O71vgBJzAHwuQZWPdJSaMchNGeS/NHY62eAHfyj8u82drzQAAAABJRU5ErkJggg=="><span data-v-65a9fde4="">\xe6\x96\xb0\xe5\xbb\xba\xe5\xaf\xb9\xe8\xaf\x9d</span></p><div data-v-65a9fde4="" class="ai-history" id="ai-history"><!----></div></div></div></div></div><!----><!----><!----></div></div></div></div></div></div><!----></div><div class="sticky-sidebar"><div data-v-e6a8c40e="" class="backToTopButton" style="display: none;"></div></div><!----></div><div id="YOUDAO_SELECTOR_WRAPPER" bindto="inputOriginal" style="display: none;\n        z-index: 101;\n        margin: 0;\n        border: 0;\n        padding: 0;\n        width: 376px;\n        height: 222px;"></div><script>var _rlog = _rlog || [];\n      // \xe6\x8c\x87\xe5\xae\x9a product id\n      _rlog.push(["_setAccount", "fanyiweb"]);\n\n      var screenWidth = window.screen.width,\n        screenHeight = window.screen.height;\n      _rlog.push(["_addPost", "screen", screenWidth + "*" + screenHeight]);</script><script>var protocol = window.location.protocol;\n      if (\n        (!window.__PRERENDER_INJECTED ||\n          !window.__PRERENDER_INJECTED.isPrerender) &&\n        protocol == "http:"\n      ) {\n        window.location.href = window.location.href.replace("http", "https");\n      } else {\n        window.alert = function () {};\n      }</script><script>const dom = document.querySelector(".isPrerender");\n      if (/^#\\/AITranslate(\\?|$)/.test(window.location.hash) && dom)\n        dom.style.visibility = "visible";</script></div><div class="popup"></div><div id="batBeacon656837486477" style="width: 0px; height: 0px; display: none; visibility: hidden;"><img id="batBeacon724074260649" width="0" height="0" alt="" src="https://bat.bing.com/action/0?ti=97136777&amp;Ver=2&amp;mid=9c0e24a2-5b74-40cb-9145-de31645b30fc&amp;bo=1&amp;sid=e46ab0b09b1611ef9a32c95457ee5d49&amp;vid=e46afd709b1611ef8fa6f5f9acb4e825&amp;vids=1&amp;msclkid=N&amp;pi=0&amp;lg=en-US&amp;sw=1440&amp;sh=900&amp;sc=30&amp;nwd=1&amp;tl=%E6%9C%89%E9%81%93%E7%BF%BB%E8%AF%91_%E6%96%87%E6%9C%AC%E3%80%81%E6%96%87%E6%A1%A3%E3%80%81%E7%BD%91%E9%A1%B5%E3%80%81%E5%9C%A8%E7%BA%BF%E5%8D%B3%E6%97%B6%E7%BF%BB%E8%AF%91&amp;kw=%E5%9C%A8%E7%BA%BF%E5%8D%B3%E6%97%B6%E7%BF%BB%E8%AF%91%E3%80%81%E5%85%8D%E8%B4%B9AI%E7%BF%BB%E8%AF%91%E3%80%81%E6%96%87%E6%A1%A3%E6%96%87%E6%9C%AC%E7%BF%BB%E8%AF%91%E3%80%81%E4%BA%BA%E5%B7%A5%E7%BF%BB%E8%AF%91%E3%80%81%E7%BD%91%E9%A1%B5%E7%BF%BB%E8%AF%91.&amp;p=http%3A%2F%2F127.0.0.1%3A8000%2F%23%2FAITranslate&amp;r=&amp;lt=561&amp;evt=pageLoad&amp;sv=1&amp;cdb=AQAQ&amp;rn=451898" style="width: 0px; height: 0px; display: none; visibility: hidden;"></div><div id="batBeacon496539796670" style="width: 0px; height: 0px; display: none; visibility: hidden;"><img id="batBeacon176443592459" width="0" height="0" alt="" src="https://bat.bing.com/action/0?ti=97136779&amp;Ver=2&amp;mid=4e7c0e51-78cb-4fa9-8843-d2eaa18fd4d4&amp;bo=1&amp;sid=e46ab0b09b1611ef9a32c95457ee5d49&amp;vid=e46afd709b1611ef8fa6f5f9acb4e825&amp;vids=0&amp;msclkid=N&amp;pi=0&amp;lg=en-US&amp;sw=1440&amp;sh=900&amp;sc=30&amp;nwd=1&amp;tl=%E6%9C%89%E9%81%93%E7%BF%BB%E8%AF%91_%E6%96%87%E6%9C%AC%E3%80%81%E6%96%87%E6%A1%A3%E3%80%81%E7%BD%91%E9%A1%B5%E3%80%81%E5%9C%A8%E7%BA%BF%E5%8D%B3%E6%97%B6%E7%BF%BB%E8%AF%91&amp;kw=%E5%9C%A8%E7%BA%BF%E5%8D%B3%E6%97%B6%E7%BF%BB%E8%AF%91%E3%80%81%E5%85%8D%E8%B4%B9AI%E7%BF%BB%E8%AF%91%E3%80%81%E6%96%87%E6%A1%A3%E6%96%87%E6%9C%AC%E7%BF%BB%E8%AF%91%E3%80%81%E4%BA%BA%E5%B7%A5%E7%BF%BB%E8%AF%91%E3%80%81%E7%BD%91%E9%A1%B5%E7%BF%BB%E8%AF%91.&amp;p=http%3A%2F%2F127.0.0.1%3A8000%2F%23%2FAITranslate&amp;r=&amp;lt=561&amp;evt=pageLoad&amp;sv=1&amp;cdb=AQAQ&amp;rn=294990" style="width: 0px; height: 0px; display: none; visibility: hidden;"></div><!----></body></html>'
```





## 安装并使用 msedgedriver

### 下载软件

找到浏览器当前的版本号，要求至少前三个数一样

还要注意是否与电脑版本相恰

### installation

在终端中配置环境变量

- 打开zsh 文件 ` nano ~/.zshrc`, 然后在尾行输入 msedriver 所在的文件夹位置，注意是 exe 文件所在的位置而不是整个解压后的文件夹：`export PATH=$PATH:/usr/local/bin`

- Ctrl + O保存，然后确认文件名后 enter，再按 ctrl + X 退出

- 重开一个窗口输入` maedgedriver --version`看是否出现版本号，出现则成功

	

## selenium 网页交互作业总结

作业中的几个点：

- 一开始网页打开后有时间延迟，所以后续操作无法连续上会导致报错；一开始使用` drive.implicitly_waite(30)`, 但是由于网页中有动态内容，查阅资料后改为显式等待
- 一直使用` by = 'id'`时因为网页结构太过于复杂而不便于分析，后来发现使用` By.CEE_SELECTOR`进行搜索可以更便捷的确定要点击的区域，因为可以直接在inspect 中对所需部位右键然后copy css selector
- 一开始以为直接在网页中下载下载链接即可下载文档，但是 print 出来以后发现点击不了，这时候发现下载下来的链接其实是网页中的文档。注意到下载时左下角的运行链接显示的网址是主网页加上下载的链接，于是结合 selenium 中的` .get(url)`方法， 直接进入下载的链接下载打牌默认的下载文件夹实现爬虫下载动画资源

PS：因为网页选择不善，进入网页后会显示不恰当的广告内容，所以提交作业的时候在视频中增加了贴纸遮挡广告



```python
import random
import time
from urllib.parse import urlparse
from bs4 import BeautifulSoup

import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

'''程序使用 selenium 与浏览器进行交互，登入动画资源下载网站，从文本文件读取用户名以及密码并进行登陆，随后搜索动画蓝箱并下载其资源列表'''

drive = webdriver.Edge()
drive.get('https://mikanani.me/')
WebDriverWait(drive, 15).until(EC.element_to_be_clickable((By.CSS_SELECTOR, 'a.text-right:nth-child(2)'))).click()

# 读取用户名和密码
with open('password.txt', 'r', encoding="UTF-8") as f:
    content = f.read()

Usernamecache = re.search(r'Username:\s*(\S+)', content)
Passwordcache = re.search(r'Password:\s*(\S+)', content)

if Usernamecache and Passwordcache:
    Username = Usernamecache.group(1)
    Password = Passwordcache.group(1)
    print(f"Username: {Username}, Password: {Password}")
else:
    print("there's no user information in the file")

# 登录操作
WebDriverWait(drive, 15).until(EC.visibility_of_element_located((By.ID, 'login-popover-input-username'))).send_keys(Username)
WebDriverWait(drive, 15).until(EC.visibility_of_element_located((By.ID, 'login-popover-input-password'))).send_keys(Password)
WebDriverWait(drive, 15).until(EC.element_to_be_clickable((By.ID, 'login-popover-submit'))).click()

# 搜索操作
WebDriverWait(drive, 15).until(EC.visibility_of_element_located((By.ID, 'header-search'))).send_keys('蓝箱')
drive.find_element(by='id', value='header-search').send_keys(Keys.ENTER)

WebDriverWait(drive, 15).until(EC.visibility_of_element_located((By.CSS_SELECTOR, '.an-text'))).click()

# 切换至新窗口
handles = drive.window_handles
drive.switch_to.window(handles[-1])

# 获取当前网页的网址并输入爬虫的程序
url = drive.current_url
print("destination url is: " + url)


# 爬取数据所需代码
class Throttle:
    def __init__(self, delay):
        self.delay = delay
        self.domains = {}

    def wait(self, url):
        domain = urlparse(url).netloc
        last_accessed = self.domains.get(domain)
        if self.delay > 0 and last_accessed is not None:
            sleep_secs = self.delay - (time.time() - last_accessed)
            if sleep_secs > 0:
                time.sleep(sleep_secs)
        self.domains[domain] = time.time()

class Downloader:
    def __init__(self, delay=5, user_agent='wswp', proxies=None, cache={}):
        self.throttle = Throttle(delay)
        self.user_agent = user_agent
        self.proxies = proxies
        self.cache = cache

    def __call__(self, url, num_retries=2):
        try:
            result = self.cache[url]
            print('Loaded from cache:', url)
        except KeyError:
            result = None

        if result and 500 <= result['code'] <= 600:
            result = None

        if result is None:
            self.throttle.wait(url)
            proxies = random.choice(self.proxies) if self.proxies else None
            headers = {'User-Agent': self.user_agent}
            result = self.download(url, headers, proxies, num_retries)
            if self.cache is not None:
                self.cache[url] = result

        return result['html']

    def download(self, url, headers, proxies, num_retries=2):
        print('Downloading:', url)
        response = None
        try:
            response = requests.get(url=url, headers=headers, proxies=proxies, timeout=10)
            response.encoding = response.apparent_encoding
            html = response.text
            if response.status_code != 200:
                print('Error code:', response.status_code)
                html = None
                if num_retries > 0 and response.status_code >= 400:
                    delay = 5
                    print(f'Pause for {delay} seconds.')
                    time.sleep(delay)
                    print('Retry to download.')
                    return self.download(url, headers, proxies, num_retries - 1)
            else:
                code = response.status_code
        except Exception as e:
            print('Download error:', e)
            html = None
            code = 404 if response is None else response.status_code
        return {'html': html, 'code': code}

downloader = Downloader(delay=1, user_agent='wswp', proxies=None, cache={})
result = downloader(url)
if result:
    html_content = result  # 从下载器返回的 HTML 内容
    soup = BeautifulSoup(html_content, 'html.parser')
    download_links = []

    # 因为都是 torrent 文件，检查后缀名下载链接
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith(('.torrent')):
            downloadlink = "https://mikanani.me" + href
            download_links.append(downloadlink)
            print("downloading: " + href)

else:
    print("Failed to retrieve the page")

# 因为文件太多，所以下载五个文件以检验程序
for element in download_links[:5]:
    drive.get(element)
```




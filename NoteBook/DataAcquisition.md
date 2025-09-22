---
tags:
  - DataScience
relation:
  - "[[ExplorativeDataAnalysis]]"
  - "[[Python学习]]"
teacher: 阮宗利
---
## 大数据搜集与可视化

### 下载网站

我们通过URL作为入口下载网站的HTML内容

常用的库有` urllib`, ` requests`

- ` urllib.request.urlopen()` : 要求其中的参数是一个请求对象或是URL, 请求对象中可以包含代理等更多信息.返回值为一个HTML对象,需要用` .read()`变成字节或者字符.
- ` urllib.request.Request()` : 向其中放入一个URL,将其转换成请求对象.
- ` .add_header()` : 其中有两个位置参数key和value,用于向请求对象rq中添加请求头.键名为` "User-Agent"`. 注意这是个添加行为而不是返回新的对象

其内在的逻辑为，初始的`urllib.request.urlopen()`会直接以默认的代理来进行爬虫访问。而当我们需要更换代理的时候只需要在 `request` 的界面更换代理，然后再使用 `urlopen()`来进行访问。而 `urlopen()`既可以处理 url 也可以处理 request对象，所以先生成一个使用已有代理的 request 对象再放入 `urlopen()`。

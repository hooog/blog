---
title: hexo+github+mac+coding搭建hexo-next主题个人博客详解
date: 2018-03-13 13:43:53
update: 
comments: true
categories:
  - Hexo
tags: 
  - Github
  - Coding
---

**- 预览我的[博客](http://www.hoooge.top)**
**- 直接上步骤：**
### 一、环境配置
#### 安装Node.js
在[官网](https://nodejs.org/en/)下载最新版本，安装即可，我们用它来生成静态网页做测试。
#### 安装Hexo
- [官网](https://hexo.io/docs/)
- 在这里我们直接在终端输入命令安装即可：
```powershell
sudo npm install -g hexo
```
### 二、生成Hexo主目录，生成静态资源
- 这里我把他放在用户主目录下，也可以放在去他目录但要注意权限问题。
```powershell
mkdir ~/hexo
cd ~/hexo
hexo init 
# 这时就会在hexo生成一些所需文件但是还不够，继续！
sudo npm install
-----------------------------------------
hexo g 
hexo s
```
- 这一步成功后，在浏览器输入localhost：4000，即可看到效果。

### 三、关联Github
- 此刻我默认各位看官已经注册好了Github
- 在github上新建个仓库，名为yourname.github.io,yourname是github的用户名，这个规则不能变.然后新建一对ssh的key,将公钥添加到github,请参考https://help.github.com/articles/connecting-to-github-with-ssh/,添加SSH keys之后，就可以使用git为后缀的仓库地址，本地push的时候无需输入用户名和密码.
注意:考虑到大家不止一个github，此处如果不这样处理，使用https的仓库地址，再接下来部署时往往会出现不让输入github用户名和密码的问题!

- 编辑本地hexo目录下的_config.yml文件,搜索deploy关键字，然后添加如下三行:
```powershell
deploy:
  type: git
  repo: https://github.com/yourname/yourname.github.io.git
  branch: master  
```
- **注意:每个关键字的后面都有个空格.**
这样就关联好了github，接下来需要在hexo目录下做两件事:
```powershell
 hexo g
 hexo d
```
- (hexo generate的缩写)生成md文件的静态页面.md文件在hexo/source/_posts目录下。如果新写的md文件，复制到此即可. 生成的静态文件在public目录下，我们将其push到github就ok了。
- hexo d(hexo depoly的缩写)将public文件夹东西推到仓库上，此时浏览器输入https://yourname.github.io就能看到自己博客啦!
还有个命令hexo clean作用是清除缓存db.json和public文件夹.

### 四、关联域名
- [英文好的童鞋可以参考这里](https://help.github.com/articles/about-supported-custom-domains/)
- 楼主自己用的阿里云，可以自行搜索下github怎么关联域名这里就不多废话了
### 五、使用Next主题
- 默认的主题不太好看，我们使用点赞最高的[next主题](http://theme-next.iissnan.com/getting-started.html#stable)。 
- 在博客根目录(hexo)下执行：
```
git clone https://github.com/iissnan/hexo-theme-next themes/next
```
- 然后执行cp themes/landscape/source/CNAME themes/next/source先把CNAME文件拷贝过来。然后在博客根目录下的_config.yml下的theme的名称landscape修改为next即可。 如果找不到这个文件就直接在themes/next/source目录创建个CNAME文件，输入你购买的域名即可，如：[www.hoooge.top](www.hoooge.top) 即可！
然后执行:
```
hexo clean
hexo g
hexo s #输入localhost:4000预览效果，没毛病的话执行下一句
hexo d #提交到public
```
- 这时候我们的主题就配置成功了，可以直接在浏览器输入你自己的域名访问博客了。

### 六、怎么发布日志到博客
- 考虑到有的童鞋可能安装好后也不知道怎么用，这里说明下：
	- 新建日志文件
```
hexo new 'MyFirstDiary'
```
	- 打开编辑文件
```
title: MyFirstDiary
date: 2017-03-29 22:37:23
categories:
  - 日志
  - 二级目录
tags:
  - hello
---

摘要部分

<!--more-->

正文部分
```
	- 发布日志
```
hexo d -g
```
- Over Done!
### 七、利用Coding进行线路优化
- 因为github服务器在国外，博客放上边访问的话有时速度较慢，大家都懂得。幸好，https://coding.net也提供了Pages服务，每个人也是1G空间。所以可以通过线路解析，国内访问coding国外访问github.我们需要三步操作:

- **coding上建仓库**
	- 首先，注册个https://coding.net，用户名假设为username,然后新建个仓库(可以是私有哦)，名为username,即建立个跟用户名同名的仓库.然后跟github一样，把本地的ssh公钥添加进去。然后点击项目，在代码--Pages 服务，在部署来源里选中master 分支。
	- 这样在博客推上去之后，可以通过http://username.coding.me/username访问博客。在下面可以看到自定义域名,假设域名为myhost.me,这里你需要绑定两个域名：myhost.me和www.myhost.me.如此，coding上的设置就一切ok了! 
注意:域名设置一定不能少，只在域名提供商如万网上设置解析是不起作用的!!!
- **hexo本地设置**
```
deploy:
  type: git
  repository:
      github: git@github.com:username/username.github.io.git
      coding: git@git.coding.net:username/username.git
  branch: master
```
**注意：一定不要设置错了!!! **
在博客根目录下的source路径下新建文件:touch Staticfile  #名字必须是Staticfile 
然后hexo g，再hexo d就可以把博客放到coding和github了!

- **域名设置**
	- 给github的线路设为海外，给coding的设为默认，添加两条记录就ok了。coding添加的时候，地址写成username.coding.me

到这里所有操作基本告一段落了，有问题的同学欢迎访问我的博客跟我讨论。
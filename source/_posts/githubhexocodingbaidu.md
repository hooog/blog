---
title: 使用Github和Coding共同托管hexo博客解决无法被百度收录的问题
date: 2018-04-5 13:05:59
password:
top:
categories:
  - Hexo
tags:
  - Coding
  - Github
---
<!--more-->



1、之前在阿里云注册的`.top`域名一直不受百度待见，遂重新申请了`.cn`域名。更换域名大致流程如下：
- 在新申请的域名下添加域名解析：主机记录添加`@`和`www`两种
![png](/images/hexo/1.png)
- 修改`CNAME`文件写入新域名`www.ihoge.cn`
- 在`Github Pages`中更新新域名并`save`

2、更换域名发现仍不能被收录，原因是`Github`嫌弃百度蜘蛛爬取太频繁，屏蔽了百度。作为国内的小伙伴无非是一种很麻烦的事。这里我选择用`Coding`和`Github`双线托管，默认解析到`Github`,并给百度设置专线解析到`Coding`

3、使用`Coding`作为`hexo`博客的第二托管平台。
- 登陆https://coding.net并注册账户，这里需要简单的操作免费升级下账户才能使用`Coding`的`Pages`服务。
- 在`Coding`上新建项目`bolg`
- 将`hexo`博客同步到新创建的仓库中
	- 第一次使用`Coding`需要使用`ssh`，方法和`Github`上一样
	- 按照官方格式修改本地`hexo`根目录下的配置文件
![png](/images/hexo/2.png)
- 完成之后在`hexo`的根目录输入：
`ssh -T git@git.coding.net`
- 上一步执行没有报错就说明成功了。然后重新部署就可以将代码同时上传到`Github`和`Coding`上了。

4、设置`Coding`的`Pages`服务
- 部署来源选择master分支
- 绑定购买的域名
- 放置`Hosted by Coding Pages`文字版到首页提交审核（方便后面的seo优化）方法是找到`themes\next\layout\_partials\footer.swig`添加代码如下：
![png](/images/hexo/5.png)

- 这里要注意上传的`CNAME`文件，上面填入购买的域名

5、绑定域名并指定百度专线：
![png](/images/hexo/3.png)

6、由于之前用`Github`托管造成百度站长抓取网站一直失败，显示服务器被拒绝。后来利用给百度指定`Coding`专线的方式才解决该问题。
![png](/images/hexo/4.png)

7、将网站链接提交到百度
百度提供三种验证方式，以Html标签为例，在themes\next\layout\_partials\head.swing中添加验证代码：
```<meta name="baidu-site-verification" content="s8Pe1TBqyy" />```
同理将链接提交到Google和搜狗

8、给站点添加sitemap

- hexo安装sitemap
```
npm install hexo-generator-sitemap --save #sitemap.xml适合提交给谷歌搜素引擎
npm install hexo-generator-baidu-sitemap --save #baidusitemap.xml适合提交百度搜索引擎
```

- 在主题配置文件`_config.yml`中找到`sitemap`添加以下代码
```
# 自动生成sitemap
sitemap:
	path: sitemap.xml
baidusitemap:
	path: baidusitemap.xml
```

- 修改站点`_config`
```
# URL
## If your site is put in a subdirectory, set url as 'http://yoursite.com/child' and root as '/child/'
url: http://www.ihoge.cn
```

9、添加蜘蛛协议robots.txt
新建robots.txt文件，添加以下文件内容，把robots.txt放在hexo站点的source文件下。
```
User-agent: *
Allow: /

Sitemap: http://www.ihoge.cn/sitemap.xml
Sitemap: http://www.ihoge.cn/baidusitemap.xml
```

hexo d -g 提交后到百度站长平台找到Robots检测并更新查看是否生效。

10、keywords 和 description
在\scaffolds\post.md中添加如下代码，用于生成的文章中添加关键字和描述。
```
keywords: 
description: 
```
在\themes\next\layout\_partials\head.swig有如下代码，用于生成文章的keywords。暂时还没找到生成description的位置。
```
{% if page.keywords %}
  <meta name="keywords" content="{{ page.keywords }}" />
{% elif page.tags and page.tags.length %}
  <meta name="keywords" content="{% for tag in page.tags %}{{ tag.name }},{% endfor %}" />
{% elif theme.keywords %}
  <meta name="keywords" content="{{ theme.keywords }}" />
{% endif %}
```

然后在\themes\next\layout\_macro\post.swig中找到并去掉以下代码，否则首页的文章摘要就会变成文章的description。
```
{% if post.description %}
  {{ post.description }}
  <div class="post-more-link text-center">
    <a class="btn" href="{{ url_for(post.path) }}">
      {{ __('post.read_more') }} &raquo;
    </a>
  </div>
```
（这里未操作）

11、修改文章链接
HEXO默认的文章链接形式为`domain/year/month/day/postname`，默认就是一个四级url，并且可能造成url过长，对搜索引擎是十分不友好的，我们可以改成 domain/postname 的形式。编辑站点`_config.yml`文件，修改其中的`permalink`字段改为`permalink: :year/:title/`即可。








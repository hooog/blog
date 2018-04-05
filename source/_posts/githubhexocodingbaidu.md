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

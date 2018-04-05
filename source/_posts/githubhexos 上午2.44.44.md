---
title: github 创建分支位作为云端备份解决hexo博客多PC间同步的问题
date: 2018-04-5 17:05:59
password:
top:
categories:
  - Hexo
tags:
  - Github
---
<!--more-->


1、如果`git master`显示红色的`x`，需要删除next目录及根目录下的`.git`,（或者修改添加你不需要上传到远程的部分）同时删除bolg根目录下的`.gitignore`。这里我选择在主目录做了备份。

2、现在回到bolg根目录分别执行：

- `git init`  初始化本地仓库
- `git checkout -b hexo` 创建并切换hexo分支
- `git remote add origin https://github.com/hooog/hooog.github.io.git`给hexo分支关联远程映射
- `git add .`  添加blog目录下所有文件，注意有个`.`（`.gitignore`声明过的文件不包含在内)
- `git commit -m '添加描述'`
- `git push origin hexo` 将hexo分支上传到远程仓库 
- 如果加错了的话执行`git rm -r --cached .`
到这里，云端备份就完成了。
同理也可以把分支备份到`Coding`的私密仓库上。（如果源代码涉及保密信息的话）

3、将远程仓库的内容拷贝到新`PC`端端本地：
```
git init
git remote add origin https://github.com/hooog/hooog.github.io.git
git fetch --all
git reset --hard origin/master
```
`fetch`是将云端所有内容拉取下来。`reset`则是不做任何合并处理，强制将本地内容指向刚刚同步下来的云端内容（正常pull的话需要考虑不少冲突的问题，比较麻烦。）

4、更新文章后的同步操作：
假设在B电脑写完了文章，也`hexo d -g`发布成功，这时候需要将新文章的md文件更新上去。

`git add .`

这时候可以使用`git status`查看下状态，一般只显示刚刚更改过的文件状态。

然后执行：
`git commit -m "更新信息"`
`git push origin hexo`

在回到A电脑上的时候，只需要
`git pull`
即可同步更新

5、日常改动
平时我们对源文件有修改的时候记得先pull一遍代码，再将代码push到Hexo分支，就和日常的使用git一样~ 
依次执行：
`git add .`
`git commit -m “…”`
`git push origin hexo`指令将改动推送到GitHub（此时当前分支应为hexo）；然后执行：
`hexo g -d`发布网站到master分支上。
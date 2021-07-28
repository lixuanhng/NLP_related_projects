## 各种情形的使用方式

+ 推送自己代码

  使用本地分支，检查本地分支是否为lixh_dev

  <font color=red>git branch</font>

  *lixh_dev

  rc

  master

  从远程下拉同样分支的代码

  <font color=red>git pull origin lixh_dev</font>

  在本地对代码进行修改，修改完毕后，提交并推送

  <font color=red>git add .</font>

  <font color=red>git commit -m 'update'</font>

  <font color=red>git push origin lixh_dev</font>

  切到rc环境

  <font color=red>git checkout rc</font>

  从远程拉下来

  <font color=red>git pull</font>

  本地合并分支

  <font color=red>git merge lixh_dev</font>

  推送到rc环境

  <font color=red>git push origin rc</font>

+ 从master中切出一个新的分支

  切换到本地master分支

  <font color=red>git checkout master</font>

  更新远程master分支到本地

  <font color=red>git pull origin master</font>

  创建并切换到newBranch分支

  <font color=red>git checkout -b newBranch</font>

  推送到远程并与远程的newBranch关联

  <font color=red>git push -u origin newBranch</font>

+ 更新版本后，git不能使用，需要重新安装xcode

  <font color=red>xcode-select --install</font>
  
+ 文件被复写之后，不能切换分支

  + error: Your local changes to the following files would be overwritten by checkout:

  + Please commit your changes or stash them before you switch branches.

    可以使用 <font color=red>git stash</font> 将已经修改的部分添加到暂存区，然后就可以切换分支了

    并且可以使用 <font color=red>git stash pop</font> 将暂存区内的数据删除

    最后强制重置 FETCH_HEAD，使用 <font color=red>git reset --hard FETCH_HEAD</font>, 这样就可以使用 git pull 将远程代码拉下来


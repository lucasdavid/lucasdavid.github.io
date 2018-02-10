---
layout: post
title: Code Versioning
excerpt: The basics on GIT and code storage
date: 2017-10-2 00:01:00
color: blue lighten-1
---

## Introduction

<div class="row">
  <div class="col-md-7">
    Versioning is one of the most basic ideas in software development, but is treated with such contempt that lots of people only start to actually use it after too much manual effort. Someone somewhere at some point said: "the idea of a human doing a task that could be automated by a machine just makes me sad". I think this sentence sums up pretty much everything about versioning: you can develop without it, sure, but it would be just painful and, considering the formality of computers, it would just result in a shittier job, most likely.
  </div>
  <div class="col-md-5">
    <figure>
      <img src="/assets/git.svg" alt="Git's basic flow">
    </figure>
  </div>
</div>

## What are versioning control softwares

Imagine that you work within a group of coders. All of you are doing concurrent/parallel work. Now, the dummy way to do it is to ask each one of the group members to implement a functionally/class/module and, at the end, put all of it together and pray for it to work. You will most likely end up with a hundred files named as `["project_" + str(i) for i in range(100)]`, which sucks.

The idea of versioning control softwares is to provide a platform that will integrate possibly adversarial coding, generating the minimal collision as possible. I.e., a program that will merge code that can be merged, alert where there is overlapping work and organize everything into versions that can be revised.

There are [many versioning control softwares](http://www.smashingmagazine.com/2008/09/18/the-top-7-open-source-version-control-systems/) out there. I encourage you to explore the possibilities to find out which one is the best suited for you. However, the goal of this post is to provide you guys a start. So, Git, I choose you!!! (Ash's voice)

## Git

Git is a great versioning control software, given its versioning policy being shaped as a digraph. A repository (of code obviously) can be forked/branched, changed, updated, rolled-back and merged again. There is no "master repository", as in SVN. This grants users the complete freedom of customization, whereas it still keeps records of all changes.

If you're already familiar with Git, you've probably read the statement above and thought "well, that was a grotesque simplification". It's true, but I'm going for the basics here... Anyhow, if that's not the case and you're still interest, you can check out [its awesome website](http://git-scm.com/).

Git will version your code based on concepts such as changes, commits, branches, merges etc. Again, you can find all about it in [Git's website](http://git-scm.com/).

## Actually using Git

Let's make a toy example. Install Git from its website. If you're a Debian/Ubuntu user, you can simply run:
```shell
sudo apt-get install git
```

You need to create a folder for your project now. I usually keep all my projects in one folder, inside the **~** (because no one else can see if), called **Repositories**. Consider the project will be named **doberman**. Therefore:
```shell
mkdir -p ~/Repositories/doberman
cd ~/Repositories/doberman
git init
```

Now add a file called `doberman.py` and and some content, such as:
```python
class Doberman(object):
    def run(self):
        print('Doberman has started!')


    if __name__ == '__main__':
        Doberman().run()
```

Now, go back to the terminal and run:

```shell
git status
```

You will see the file that you've just created on the <b>untracked files</b> section. That means that file isn't being versioned. Add it with:
```shell
git add doberman.py
```

> **#1** `git add` actually accepts regular expressions. I.e.: `git add *` would add all untracked files to the list for the next commit. `git add doberman.*` would add all files with the name "doberman." followed by anything.

You can now make your first commit. That records a version of your code that can be revised and, possibly, reverted.
```shell
git commit -m "Add Doberman class"
```

Simply typing `git commit` also works. That will open the default text editor, where you can type a more detailed commit message. To continue, simply save the file and close the editor.

You can now mess up as much as you want. Suppose that this code were tremendously complex and you have reached a point where you just don't know what you have done and this shit is broken. You could simply hit:
```shell
git stash
```

And all files would be reverted to their original state in the beginning of the commit. No harm done.

Finally, let's say that you'd like to try something, but you aren't really sure if it's going to work or not. What you do know is that it's going to take a quite amount of steps to see some results, and you would like to commit each step, just to make sure that stash isn't going to reset ALL changes down to this point. Situations like this require you to branch your repository, where a branch is a path of commits.
```shell
git branch test_bark
```

Check the branches locally available through...
```shell
git branch
```

And finally switch from mater to test_doberman_bark:
```shell
git checkout test_bark
```

Do all the work you want. Let's say, three commits are enough for you to complete the functionality that you were looking for, and everything looks great.
```shell
git checkout master
git merge test_bark
git branch -D test_bark
```

With this: (a) the repository goes back to master, where the code goes back to that previous set of commits, (b) merges test_bark branch on master (push all commits from test_bark to master set of commits) and (c) deletes test_bark branch.

Finally, in order to share your repository with other people, you can host it in a existing Git server, such as [Github](https://github.com) or [bitbucket](http://bitbucket.org/).

This was just the tip of the iceberg, from which I hope you can take some interest. For a much more detailed tutorial, you can access [Gittutorial docs](git-scm.com/docs/gittutorial).

## Short version for lazy loaders

```shell
sudo apt-get install git       # install git
mkdir ~/Repositories/doberman  # creates the repository folder
cd ~/Repositories/doberman     # navigates to it.
git init                       # initiates a git repository
vi doberman.py                 # creates doberman.py. You need to fill this document with code!
git add doberman.py            # adds doberman.py to the tracking list
git commit -m "Add Doberman class" # commits changes
...
git stash                      # resets all changes to head of previous commit.
git checkout -b test_bark      # creates a branch "test_bark" and switch to it.
...
git checkout master            # goes back to master.
git merge test_bark            # merges test_bark's commit list onto master.
git branch -D test_bark        # deletese test_bark branch.
```

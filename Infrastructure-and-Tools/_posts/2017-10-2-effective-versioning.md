---
layout: post
title: Effective Code Versioning
excerpt: Digging Deep in Code Versioning with GIT
first_p: |-
  Some weeks ago, I've started this blog with this very subject. Today, as requested by a reader, I'll go back and try to talk a little bit more about versioning focusing on GIT, it's problems and how to avoid/fix them.
toc: true
date: 2017-10-2 00:08:00
lead_image: /assets/images/posts/git-advanced.svg
tags:
  - DevOps
---

<span>Some</span>
weeks ago, I've started this blog with this [very subject](https://ycoding.wordpress.com/2015/05/03/14/). Today, as requested by a reader, I'll go back and try to talk a little bit more about versioning (focusing on [GIT](http://git-scm.com/)), it's problems and how to avoid/fix them.

Versioning is great. If you're coding alone, it provides you cool features, like saving, branching, rolling back, reverting, peeking previous states etc. In the other hand, if you're working with a team, it becomes even more important:, as it's a way to automate what can be automated, stop wasting so much time on team management and focus on your domain problem.

<figure>
  <img src="{{site.baseurl}}/assets/images/posts/git-advanced.svg" alt="An advanced flow using Git" class="figure-img img-fluid rounded" />
    <figcaption>An advanced flow using Git. Available at: <a href="https://www.benmarshall.me/git-rebase/">benmarshall.me</a></figcaption>
</figure>

## First, some important GIT concepts

 * **Commit change list**: the list of files to be committed - `git add {file-name}` to add files to the list and `git reset HEAD {file-name}` to remove them.
 * **Commit**: a well defined state of your repository that's saved and can be peaked or restored - `git commit -m "{your-message}"`
 * **Branch**: a path of commits. It starts as a fork of another branch and can be merged again to any other branch - `git checkout -b {new-branch-name}`.
 * **Pull**: operation which retrieves all commits made on a specific branch from a remote server to your local GIT repository - `git pull`.
 * **Push**: operation which uploads all commits made on a specific branch, in your local GIT repository, to a remote server - `git push`.
 * **Merging**: process in which modifications in two different branches are merged. If there's not conflicts, the entire process will be automatic - `git merge {branch-to-be-merged-name}`.
 * **Conflict**: a state in which the repository enters after the merging process, if the former was unable to automatic merge all the changes. You can't create new commits until the conflicts have been solved.

### How conflicts "magically" appear
I hear way to many people complaining how hard it's to fix GIT conflicts. Before talking about how to reduce them, let's understand the merging process: GIT will compare both versions and merge them optimally. Some regions of the file, though, cannot be merged. That happens when the same region is **modified by two different commits**, where **one of them is NOT an ancestor of the other** and the **two versions are now different**. In this case, GIT can't conservatively infer which modification should be preserved and which should be discarded, so it simply informs you that a conflict happened and you must fix it.

The image bellow illustrates how GIT represents a conflict. To solve it, you must manually re-write the entire region/delete one of the versions and erase the lines 1, 4 and 6. Finally, it'll allow you to commit.

![A conflict example](https://help.github.com/assets/images/mac/changes/merge_conflict_sample.png)

> **#1** Please, keep in mind that GIT didn't create conflicts. They exist whether you're using a versioning tool or simply grabbing your mates codes and manually merging them. Seriously. IT IS NOT easier to keep swapping a hundred different .zip files through email.

> **#2** For more information on workflow, there is this [nice article](http://stackoverflow.com/questions/273695/git-branch-naming-best-practices) from Atlassian.

## What is our problem?
Although GIT helps us a lot, there are many situations that are a real pain in the ass. In my opinion, these are the worse:

* Recurrent conflicts, pissing you off as you have to fix them all pulls/pushes/merges.
* Constant modifications are being made on regions that you directly depend, constantly breaking up your code.
* Changes that weren't made by you, which have broken the entire repository.
* Modifications that were not there when you left, and you just can't seem to understand them.
* All branches are broken, which means there isn't a single version of your project that's stable.

## How to solve it.
We don't have the whole day to fight GIT and our code mates over this. Plus, we're here to concentrate on our domain problem: the code that is contained on the repository. Naturally, we want to minimize, or rather eliminate these situations.

### Work in short cycles
Be organized. Make small changes and commit these right away. Don't modify the entire file and just then remember to commit. Yes, `git commit -a` is an abomination.

Before starting to code, always `pull` from the remote repository. After some commits, `push`. Don't leave commits locally, as these can create conflicts that could so easily be avoided.
```shell
git pull
git push --all
```

### Separate work that isn't related
You should be able to pick up from were you left, not have to understand dozens of new lines that don't relate to your work at all every time you `pull`. If you're working onto something, create a new branch and do your thing. Commit, experiment and revert without any worry of breaking someone else's code. After all, they'll be on their own branches.

For branch naming, I was taught to name it after an issue. I find this particularly useful, as both [GitHub](github.com) and [Bitbucket](bitbucket.org) maintain an issue tracker and their issues can contain lots of information! Additionally, the branch name becomes quite simple. As an toy example: imagine that you've created an issue `Adding Google oauth2 authentication alternative` in your Bitbucket's repository and it was assigned the **id #45** to that issue. Go to your repository, create and move to a new branch `iss45`:
```shell
git checkout -b iss45 # create and switch to branch iss45
```

You can now implement the usage of Google oauth2 service, free from other people bugging your work. When you're done, you can go back to `master`, merge `iss45` and delete it:
```shell
git checkout master # go back to master
git merge iss45     # merge iss45 -> master
git branch -D iss45 # delete iss45
```

> **#3** Don't be an douchebag and break other people's code. Make sure you've tested the code in your branch before merging it to `master`. Run the tests again after you've merged. Overall, try to keep the project on your main branch working.

> **#4** If you didn't like this naming style, here is some [reference](http://stackoverflow.com/questions/273695/git-branch-naming-best-practices) that might interest you. This one involves given meaningful names to branches, such as `feature/new/google-oauth2`.

### Continuous stability
Make sure your repository has at least on branch that contains stable code. Always be careful when merging to that branch, and be sure that you've tested it thoroughly before doing so.
```shell
git branch # list existing branches
git branch release # create release
```

When you're certain that `master` is stable, merge it:
```shell
git checkout release # go to release
git merge master     # merge master
git checkout master  # go back to master
```

In opposite of having a branch `release`, you can also use `master` as the "stable brach" and create a branch called `dev`. You should then work mainly at `dev`. I prefer the first option, though, as GitHub doesn't close issues automatically when you commit with the message **fix #{issue id}** in branches that are not the main.

## Short version for lazy-loaders
```shell
git pull   # retrieve modifications from remote
git branch # list existing branches

git checkout -b iss45 # create and switch to branch iss45

... # do your thing and, when you're sure it's working...

git checkout master # go back to master
git merge iss45     # merge iss45 -> master
git branch -D iss45 # delete iss45
git push # push modifications to remote

git checkout -b release # create and switch to the release branch
git merge master        # merge master -> release

git checkout master     # go back to master
```

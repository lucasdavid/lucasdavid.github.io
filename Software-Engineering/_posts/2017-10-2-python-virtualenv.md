---
layout: post
title: Python virtualenv and Dependency Management
excerpt: The basics project's dependency management using virtualenv
first_p: |-
    If you like Java or C#, you most likely have had contact with Maven or NuGet. These are both amazing utilities that can help you managing your projects and their dependencies. These tools will basically retrieve packages from a repository and install them somewhere, making them easily <code>[imported|used|updated|removed]</code> (read this as a regular expression). They can also modify or create structures or files inside your project, providing you a starting point for doing something. There is much stuff here, check out their web-pages to see what I'm talking about.
toc: true
date: 2017-10-2 00:02:00
tags:
  - DevOps
  - Python
  - virtual environments
---

<span class="display-6">If</span> you like Java or C#, you most likely have had contact with [Maven](https://maven.apache.org/) or [NuGet](https://www.nuget.org/). These are both amazing utilities that can help you managing your projects and their dependencies. These tools will basically retrieve packages from a repository and install them somewhere, making them easily `[imported|used|updated|removed]` (read this as a [regular expression](http://en.wikipedia.org/wiki/Regular_expression)). They can also modify or create structures or files inside your project, providing you a starting point for doing something. There is much stuff here, check out their web-pages to see what I'm talking about.

Let's focus this discussion on Python. Python, of course, has a similarly cool feature. It's called [pip](https://pypi.python.org/pypi/pip).

> **#1** If you're a Linux user, try to locate pip in your bin folder (where python is most likely installed):
> ```shell
> ls /usr/bin
> ```
> Hahaha it's just humanly impossible, right? There is just so many packages... `ls /usr/bin/pip` will locate it, though.

See, pip will install your dependencies in common places, as `/usr/bin/` or some default library folder, because it knows that those places are usually on the users' `path` or `source root`, making them easier to use.

> **#2** Let's make a quick exercise: open the terminal and install [uwsgi](https://uwsgi-docs.readthedocs.org/en/latest/), a very good runner of wsgi modules used by HTTP servers.
> ```shell
> pip install uwsgi
> ```

There is a small "problem" with pip. Can you see it? As it install packages on this *global* space, it could fill these folders with stuff that you don't need/use, contaminating all the environment with its new commands. Additionally, it could easily override something that is important or that is being used by some other program.


Another adversarial situation is when you have multiple projects, each one depending on a different version of a same library. For example, I rather use Python3 whenever I can, which means I tend to run `virtualenv env -p /usr/bin/python3` often. However, one simply can't install `MySQL-python` on Python3.4, in oposite to Python2.7. That shit is just broken. Python programmers are passing through a rough transition, period.


## Virtualenv FTW

We know our problem: the package "leaking" that pip will make. How can we solve it? Simple: what if we could encapsulate the environment in a local structure, and, at some point, switch to that environment and work on it? That is the idea of [virtualenv](https://virtualenv.pypa.io/en/latest/).

Firstly, you have to install virtualenv:
```shell
pip install virtualenv
```

To use it, you can simply type `virtualenv DEST_DIR [-p python_exec_path]`.
For example:

```shell
mkdir ~/Repositories/django-new-project
cd ~/Repositories/django-new-project
virtualenv env
```
> **#3** You don't necessarily have to put `env` inside your project's folder. You could have a separate directory that holds virtualenvs. However, I believe it's better to keep them inside the project folder, since different projects might have different packages' versions dependencies and, given a great number of projects, you might get lost on which virtualenv is the right one for each project.

Done! The default python interpreter, as well as pip have been copied to `env` directory. You can now switch environments:

```shell
source env/bin/activate # for linux users
envScriptsactivate    # for windows users
```

Perfect! You will notice that the terminal will display `(env)` behind the current's directory cursor. You can now install all packages that you'd like without messing with any other programs dependencies.

```shell
pip install django django-filters djangorestframework # ... and many others.
```

All these packages will be installed under the `env` directory, preventing collision with the global packages.
> **#4** If you want to exit the virtual environment, type `exit` and hit `Enter`.

Sometimes, you want to install what's not yet installed and update what is installed:
```shell
$ pip install flask flask-restful flask-scripts mongoengine --upgrade
```

Finally, a project can have many dependencies, which makes it really hard to keep track of all of them. You can
easily solve this by creating a text file that will hold all dependencies names and simply giving this file to pip.
So, create a file `requirements.txt`:

```python
django
djangorestframework
django-filters
rest_condition
markdown
MySQL-python
python-memcached
drf-extensions
django-oauth-toolkit
uwsgi
requests
enum
```

And finally give it to pip:
```shell
pip install -r requirements.txt
```

## Short version for lazy-loaders
```shell
mkdir ~/Repositories/cool-python-project  # creates the project folder.
cd ~/Repositories/cool-python-project     # navigates to it.
virtualenv env                            # creates the virtual environment.
source env/bin/activate                   # switchs to local virtualenv.
vi requirements.txt                       # creates a file requirements.txt (you have to fill it with the wanted references)
pip install -r requirements.txt --upgrade # selective installs and updates packages.
```

Well, I guess this is it. Please let me know if there's anything wrong or that could be improved! Your oppinion is highly appreciated!

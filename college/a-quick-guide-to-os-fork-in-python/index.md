---
layout: post
nav-menu: false
title: "A quick guide to os.fork in Python"
date: "2017-10-24"
categories: 
  - "python"
tags: 
  - "fork"
  - "parallel"
  - "python"
---

Paralellization usually is pretty tricky in python, however there's a super easy way to implement pretty straightforward parallelization using the built-in `os.fork()` functionality.

Let's talk about what `os.fork()` actually does. In short, it simply creates an additional copy (referred to as the "child process") of the running program at the same(ish) spot as the "parent process". These are separate processes, so they have completely different memory allocated for them (and _cannot_ share variables).

This lack of internal communication means that this is best suited for having a single piece of code concurrently spin up a separate task that has some external communication method (files, network, etc.) or one that doesn't have any communication requirements whatsoever.

When `os.fork()` is called, it returns back a different number depending on whether or not it's the child or parent process. For the child, the result of `os.fork()` is the number `0`. For the parent the result is the `pid` of the child process.

```
import os, signal

pid = os.fork()
if pid == 0:
    # have the child do a thing
else:
    # have the parent do a thing
    os.kill(pid, signal.SIGINT)
```

I use this for spinning up a webserver and then running integration tests against it in the same codebase without having to shell out. A good example is the [Boulder Python Meetup](http://www.boulderpython.org) website. This code [is available on github](https://github.com/boulder-python/boulderpython.org) but here's the relevant code snippet:

```
def main(args: argparse.Namespace) -> int:
    configure(os.getenv('FLASK_CONFIG') or 'default')

    status = True
    if args.test is True:
        pid = os.fork()
        if pid == 0:
            status = run_server(mode='debug')
        else:
            try:
                status = pytest.main(['tests'])
            except:  # trust me this is ok
                pass
            os.kill(pid, signal.SIGINT)
    elif (args is not None) and (args.debug or app.debug):
        status = run_server(mode='debug')
    else:
        status = run_server(mode='prod')
    return status
```

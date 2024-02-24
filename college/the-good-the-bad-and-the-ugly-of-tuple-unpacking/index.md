---
title: "The Good, the Bad, and the Ugly of Tuple Unpacking"
layout: post
nav-menu: false
date: "2017-10-28"
categories: 
  - "python"
tags: 
  - "python"
  - "style"
  - "tuples"
---

Python has this neat feature of unpacking tuples during item assignment. Here's a general example:

```
a, b, c = (a, b, c)
```

or if you have a function that returns multiple items:

```
a, b, c = foo()
```

But what if (for some reason) you have a function that returns some large number of variables (or a smaller number of long-named variables) and need to unpack all of them? How do we style this for PEP8 compliance and maintaining readability?

# The Ugly

This is the naive approach, and also breaks line-length style.

```
first_var, second_var, third_var, fourth_var, fifth_var, sixth_var, seventh_var, eighth_var, ninth_var, tenth_var, eleventh_var, twelfth_var = foo()
```

# The Good

This is probably the best style for this sort of thing (indentation style-dependent)

```
(first_var,
 second_var,
 third_var,
 fourth_var,
 fifth_var,
 sixth_var,
 seventh_var,
 eighth_var,
 ninth_var,
 tenth_var,
 eleventh_var,
 twelfth_var) = foo()
```

# The Bad

Don't do this. No matter how tempted you are. This is bad.

```
first_var, *_ = foo()
second_var, *_ = _
third_var, *_ = _
fourth_var, *_ = _
fifth_var, *_ = _
sixth_var, *_ = _
seventh_var, *_ = _
eighth_var, *_ = _
ninth_var, *_ = _
tenth_var, *_ = _
eleventh_var, twelfth_var = _
```

# The Worse (Halloween Edit!)

Here's a followup, even worse way to do things...

Let's assume we have some function called `foo` that returns some number of variables. For testing purposes we'll set this function to just return all of its arguments.

```
def foo(*args):
    return args 
```

We can call this and bind each of the returned variables to a custom variable in the `locals()` or `globals()` namespace (or both if we really want).

```
for i, arg in enumerate(foo(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)):
    locals()[f'variable_{i}'] = arg                        
```

Which works great!

```
>>> locals()
{...
 'variable_0': 1,
 'variable_1': 2,
 'variable_2': 3,
 'variable_3': 4,
 'variable_4': 5,
 'variable_5': 6,
 'variable_6': 7,
 'variable_7': 8,
 'variable_8': 9,
 'variable_9': 10}
>>> variable_0
1
```

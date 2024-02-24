---
layout: post
nav-menu: false
title: "An Introduction to Python's Types"
date: "2017-10-25"
categories: 
  - "python"
tags: 
  - "programming"
  - "python"
  - "types"
  - "typing"
---

In this post we'll talk about python's types, how to use them, how they're treated, and what we can do with typing.

This is aimed at beginners who have heard the words "Python" and "Types" but haven't quite nailed down what they have to do with each other.

This is Part 1 of a multi-part series on Python Typing.

# Python's Native Types

The best place to start is just by listing each of the types that Python uses natively so that way we can work through examples later. As with most core-python functionality, the best place to learn more is the documentation, specifically [this page](https://docs.python.org/3/library/stdtypes.html).

- Boolean - `bool`
- Integer - `int`
- Float - `float`
- Complex
- Iterator
- Generator
- List - `list`
- Tuple - `tuple`
- Range - `range`
- String - `str`
- Bytes
- Byte Array
- Memory View
- Set - `set`
- Frozenset - `frozenset`
- Dictionary - `dict`
- Context Manager
- Modules
- Functions
- Methods
- Code Objects
- Type Objects
- Null - `None`
- Ellipses
- NotImplemented

Whew, that's a ton. Let's focus on the more important ones for now.

## Booleans

Truth values, these can take on one of two values - `True` or `False`.

```
In [1]: type(True)
Out[1]: bool

In [2]: type(False)
Out[2]: bool

In [3]: type(5 == 5)
Out[3]: bool

In [4]: type(None is None)
Out[4]: bool
```

## Integers

Basic numeric type, these are whole numbers.

```
In [5]: type(5)
Out[5]: int

In [6]: type(6 * 10)
Out[6]: int

In [7]: type(-1)
Out[7]: int
```

## Floats

Another numeric type, these are decimal numbers.

```
In [8]: type(0.0)
Out[8]: float

In [9]: type(3.14159)
Out[9]: float

In [10]: type(2**3.1)
Out[10]: float
```

## Lists

An ordered collection of items. These items don't have to be the same type, and there's no limit on the length of the list. Lists are _mutable_, which we'll talk about later.

```
In [11]: type([1, 2, 3])
Out[11]: list

In [12]: type([])
Out[12]: list
```

## Tuples

Another ordered collection of items, which also don't have to be the same type, and there's also no limit on the length. However, the big difference between lists and tuples is that tuples are _not mutable_.

```
In [13]: type((1, 2, 3))
Out[13]: tuple

In [14]: type((1,))
Out[14]: tuple
```

## Ranges

Ranges are ordered sequences of numbers created with the `range` keyword.

```
In [15]: list(range(0, 11, 2))
Out[15]: [0, 2, 4, 6, 8, 10]

In [16]: type(range(10))
Out[16]: range

In [17]: type(range(0, 11, 2))
Out[17]: range
```

## Strings

Strings are an ordered collection of "characters".

```
In [18]: type('a')
Out[18]: str

In [19]: type('this is a string')
Out[19]: str
```

## Sets

Sets are an unordered collection of items. These are similar to lists, but are _not mutable_ and _not ordered_.

```
In [20]: type({1, 2, 3})
Out[20]: set
```

## Dictionaries

Dictionaries are mappings of Key to Value pairings. The keys can be any _immutable_ thing, and the values can be anything.

```
In [21]: foo = {'a': 5, 'b': 10}

In [22]: foo['a']
Out[22]: 5

In [23]: type(foo)
Out[23]: dict

In [24]: {[1, 2, 3]: 5}  # can't have mutable keys
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-22-c0fedc33c691> in <module>()
----> 1 {[1, 2, 3]: 5}

TypeError: unhashable type: 'list'
```

## None

The Null type, the type for something that doesn't exist is called None.

```
In [25]: type(None)
Out[25]: NoneType
```

# Mutability

We mentioned mutability a lot, but didn't nail down what that means. It's a essentially just whether or not a variable is "changeable". For instance, with a mutable object (like a list) we can change the values in place.

```
In [1]: foo = [1, 2, 3]

In [2]: foo
Out[2]: [1, 2, 3]

In [3]: foo[0] = -5

In [4]: foo
Out[4]: [-5, 2, 3]
```

However an immutable object cannot be changed.

```
In [5]: foo = (1, 2, 3)

In [6]: foo
Out[6]: (1, 2, 3)

In [7]: foo[0] = -5
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
<ipython-input-7-c1a884884171> in <module>()
----> 1 foo[0] = -5

TypeError: 'tuple' object does not support item assignment
```

# Conclusion

Everything object in Python has a type, but above we went over the most basic and necessary ones.

In the next post we'll talk about the types provided by the `typing` module in Python 3 and how they can be used.

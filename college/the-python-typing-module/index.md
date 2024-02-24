---
layout: post
nav-menu: false
show_tile: false
title: "The Python Typing Module"
date: "2017-10-31"
categories: 
  - "python"
---

The `typing` module added in Python 3.5 ([see reference docs here](https://docs.python.org/3/library/typing.html)) adds additional types and meta-types to allow for more control over [python type hints](https://www.python.org/dev/peps/pep-0484/). In this post we'll talk about what this module adds and what neat things you can do with it.

This is the third post in a multi-part series on typing in Python.

1. [Introduction to Python Types](http://dataleek.io/index.php/2017/10/25/an-introduction-to-pythons-types/)
2. [Python Type Hinting](http://dataleek.io/index.php/2017/10/30/python-type-hinting/)

# General Overview

The following types are new in this module and require importing them from the `typing` module (available in python3.5+) before using.

- `Any`
- `Union`
- `Tuple`
- `Callable`
- `List`

This module also adds the functionality to alias types and create new types.

There are also equivalent types for every native type that exist solely for the purpose of type-checking and hinting.

# How to Check Types (the Pythonic Way)

Soooo, technically the following code snippet _works_, but isn't Pythonic.

```
>>> x = 'foobarbizzbazzbang'
>>> type(x) is str
True
```

The better way to check this is to use `isinstance` and `issubclass`

```
>>> isinstance(x, str)
True
>>> issubclass(type(x), str)
True
```

For more information see [this SO post](https://stackoverflow.com/questions/152580/whats-the-canonical-way-to-check-for-type-in-python).

For these examples however, we'll use [enforce](https://github.com/RussBaz/enforce) to check the types. Feel free to run the examples and verify the results.

```
from typing import *
from enforce import runtime_validation
```

# The `Any` Type

Everything is of the `Any` type. Everything.

```
@runtime_validation
def foobar(x: Any, y: Any, z: Any) -> Any:
    return x

foobar(5, 6, 7)
foobar(5, 'asdfasdf', 7)
foobar(2.2, 'sdasdiiijsdjo  ', lambda : None)
```

# The `Union` Type

This type allows for an object to be either one type or the other.

```
@runtime_validation
def uniontype(x: Union[int, str]) -> int:
    return int(x)

uniontype(5)
uniontype('5')
uniontype(5.0)

---------------------------------------------------------------------------
RuntimeTypeError                          Traceback (most recent call last)
<ipython-input-7-59167f3f0321> in <module>()
      5 uniontype(5)
      6 uniontype('5')
----> 7 uniontype(5.0)

RuntimeTypeError: 
  The following runtime type errors were encountered:
       Argument 'x' was not of type typing.Union[int, str]. Actual type was float.
```

# The `Tuple` Type

The Tuple type models exactly what tuples do in python, immutable ordered collections.

```
@runtime_validation
def tupletype(x: Tuple[int, int]) -> Tuple[int, int]:
    return x

tupletype((2, 3))
tupletype((2, '3'))

---------------------------------------------------------------------------
RuntimeTypeError                          Traceback (most recent call last)
<ipython-input-9-ec3da0b44c47> in <module>()
      4 
      5 tupletype((2, 3))
----> 6 tupletype((2, '3'))

RuntimeTypeError: 
  The following runtime type errors were encountered:
       Argument 'x' was not of type typing.Tuple[int, int]. Actual type was typing.Tuple[int, str].
```

# The `Callable` Type

This is for typing functions and other callable items.

```
@runtime_validation
def callabletype(x: Callable[[int, int], str]) -> None:
    return

def dummyfunc(x: int, y: int) -> str:
    return 'foo'

callabletype(dummyfunc)
callabletype(lambda x, y: 'sdf')
---------------------------------------------------------------------------
RuntimeTypeError                          Traceback (most recent call last)

<ipython-input-12-9a2a66861b54> in <module>()
      7 
      8 callabletype(dummyfunc)
----> 9 callabletype(lambda x, y: 'sdf')

RuntimeTypeError: 
  The following runtime type errors were encountered:
       Argument 'x' was not of type typing.Callable[[int, int], str]. Actual type was typing.Callable.
```

# The `List` Type

As with `Tuple`, this is the type for python lists.

```
@runtime_validation
def listtype(x: List[int]) -> List[int]:
    return x

listtype([1, 2, 3])
listtype([1])
listtype([1.0, 8])

---------------------------------------------------------------------------
RuntimeTypeError                          Traceback (most recent call last)
<ipython-input-16-ae038f6b0641> in <module>()
      5 listtype([1, 2, 3])
      6 listtype([1])
----> 7 listtype([1.0, 8])

RuntimeTypeError: 
  The following runtime type errors were encountered:
       Argument 'x' was not of type typing.List[int]. Actual type was typing.List[int, float].
```

# Other Types

There are a ton of other official types introduced in this module with the purpose of typing functions more accurately. They all have fairly intuitive usages, which is why I'm not going through and listing all of them. For a full list check [the reference docs here](https://docs.python.org/3/library/typing.html).

---
title: "Python Type Hinting"
date: "2017-10-30"
categories: 
  - "python"
tags: 
  - "enforce"
  - "mypy"
  - "python"
  - "typing"
---

In Python 3.5 and greater an "optional type hinting syntax" was added. This is part of a [gradual typing](https://en.wikipedia.org/wiki/Gradual_typing) implementation (gradual typing is essentially adding a few types to an untyped codebase, or only partially typing the codebase as you go).

This is the second post in a multi-part series on typing in Python. The first post in the series covers native python types, [check it out here](http://dataleek.io/index.php/2017/10/25/an-introduction-to-pythons-types/).

Let's dig into what this is and how it looks.

# The Goal

The goal of this gradual typing initiative is to add type hinting to codebases that either didn't have it to begin with, or to include while developing a new codebase. Note that it's type _hinting_, not type _checking_. This means that for all intents and purposes this type hinting is merely a glorified comment that can be used by other programs. This isn't to sell it short however, by adding this hinting, you essentially add another form of comment that many developers do not include in their initial docstring.

# The Syntax

The syntax is very straightforward. In Python 3.5 you can optionally include these type hints on function declarations. The general form is for function variables to use the colon to indicate its type, `foo: str`. For function returns you use an arrow to indicate the function's return type. `def foo() -> str:`. An example follows.

```
def some_function(foo: str, bar: int, bazz: float) -> None:
    return None
```

In Python 3.6+ you can also type variables declared in the code itself, and not just in function calls.

```
def some_function() -> str:
    foo: str = 'test string'
    bar: int = 5
    bazz: float = 3.14159
    return foo
```

# Why do we care?

Well, a couple reasons. first of which being that these type hints act as additional comments that convey information about your codebase to others.

The second is that they can be used by other programs (or libraries) to check the validity of your code either statically or in realtime.

## Real Time Type Checking

If you're using these type hints, you might want your code to check that it's being called correctly with the correct type of variables. By default this functionality is not supported. This is where [a realtime type-checker like enforce](https://github.com/RussBaz/enforce) comes in. Simply put, it just checks that your input variables and function output match the function's declaration. It's easy to enable on any function just by simply using a decorator.

```
>>> import enforce
>>>
>>> @enforce.runtime_validation
... def foo(text: str) -> None:
...     print(text)
>>>
>>> foo('Hello World')
Hello World
>>>
>>> foo(5)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/zoe/.local/lib/python3.5/site-packages/enforce/decorators.py", line 106, in universal
    _args, _kwargs = enforcer.validate_inputs(parameters)
  File "/home/zoe/.local/lib/python3.5/site-packages/enforce/enforcers.py", line 69, in validate_inputs
    raise RuntimeTypeError(exception_text)
enforce.exceptions.RuntimeTypeError: 
  The following runtime type errors were encountered:
       Argument 'text' was not of type <class 'str'>. Actual type was <class 'int'>.
>>>
```

(I'm also one of the contributors on the Enforce project)

## MyPy

MyPy is a different style of approaching the typing problem, and has much more popularity than Enforce. Instead of checking validity of your program in realtime, it performs what is known as [static type checking](https://en.wikipedia.org/wiki/Type_system#Static_type_checking), which checks the validity of the types without actually running the code. The usage is very easy, and in many ways this acts as a linter (such as pylint), just with more functionality.

Let's say you have some code that looks like the following. From a typing perspective, this is completely wrong even though it's valid python.

```
def some_func(x: str) -> None:
    return x

some_func(5)
```

If we run mypy against this code, it will let us know that there's an incompatibility between the typing and the expected result.

```
[11:47:53] zoe@phoenix /home/zoe/projects/typing_docs (0) 
> mypy --ignore-missing ./examples.py 
examples.py:18: error: No return value expected
examples.py:20: error: Argument 1 to "some_func" has incompatible type "int"; expected "str"
[11:49:03] zoe@phoenix /home/zoe/projects/typing_docs (1) 
> 
```

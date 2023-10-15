# Spartan Lisp - Simple Lisp

Spartan Lisp (Slisp) is designed to be a quick and easy lisp with focus on ergonomics 
without sacrificing performance. Currently, this repo only includes the python implemention used
for testing and quick prototyping.

## Getting Started
Assuming you have python already installed Spartan Lisp can be started by:

```
python main.py
```

However I highly recommend using `rlwrap` or similar to get autocomplete as well as history

A few things to note about Slisp
1) There are no comments - The interpreter will ignore anything not in a Sexpr so you can freely write whatever you want so long as it doesnt include (
2) Its "Functionally Pure*" -`set` only modifys the active locale so you can not update global variables in function scope (to be reviewed)
3) It is somewhat vectorized - operations such as + % / etc are all natively vectorized meaning (+ [1 2 3] 2) will work
4) There is no Int type - this is more for simplicity for now but at the minute all you get are floats
5) There isnt any write functionality - although you can read a file with `read0` currently your best bet is to use `os` to echo into a file (to be updated)

## Todo
- [ ] Proper Macros
- [ ] More primitives especially list related ones such as drop and @
- [ ] Clean up the python implementation to not be a monolith file
- [ ] More examples
- [ ] Docs
- [ ] Remote IPC support


class Sexpr:
    def __init__(self, first, rest):
        self.first = first
        self.rest = rest

    def __eq__(self, other):
        return self.first == other.first and self.rest == other.rest

    def __repr__(self):
        return f"({self.first} {' '.join([str(x) for x in self.rest])})"

    def __str__(self):
        return f"({self.first} {' '.join([str(x) for x in self.rest])})"


class SLISPString:
    def __init__(s, x):
        s.value = x

    def __str__(s):
        return f'"{s.value}"'


class SLISPNumber:
    def __init__(s, x):
        s.value = x

    def __str__(s):
        return f"{s.value}"

    def __add__(self, other):
        if other is not None:
            return SLISPNumber(self.value + other.value)

    def __mul__(self, other):
        if other is not None:
            return SLISPNumber(self.value * other.value)

    def __sub__(self, other):
        if other is not None:
            return SLISPNumber(self.value - other.value)

    def __lt__(self, other):
        if other is not None:
            return self.value < other.value

    def __bool__(self):
        return self.value > 0

    def __mod__(self, other):
        if other is not None:
            return SLISPNumber(float(int(self.value) % int(other.value)))

    def __eq__(self, other):
        if other is not None:
            return SLISPNumber(1 if self.value == other.value else 0)

    def as_int(self):
        return int(self.value)


class SLISPSymbol:
    def __init__(s, x):
        s.value = x

    def __str__(s):
        return f"'{s.value}"


class SLISPList:
    def __init__(self, vs):
        self.values = vs

    def __str__(self):
        return f"[{' '.join([str(X) for X in self.values])}]"

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, item):
        return self.values[item]


class SLISPFunction:
    def __init__(self, name, params, exprs):
        self.name = name
        self.params = params
        self.exprs = exprs
        self.locale = {}

    def __str__(self):
        return f"(λ {self.name} [{';'.join(self.params)}])"

    def execute(self, rest, locale):
        if len(rest) != len(self.params):
            raise Exception("rank")
        self.locale = {x: evaluate(y, locale) for x, y in zip(self.params, rest)}
        self.locale["self"] = self
        # self.locale["set"] = self.set_value
        # self.locale["fn"] = self.set_value
        r = None
        for E in self.exprs:
            r = evaluate(E, locale=self.locale)
        return r


class StringStream:
    def __init__(self, s):
        self.s = s
        self.i = 0

    def next(self):
        c = self.s[self.i]
        self.i += 1
        return c

    def peek(self):
        return self.s[self.i]

    def step_back(self):
        self.i -= 1

    def done(self):
        return self.i >= len(self.s)


def parse_string(s: StringStream):
    x = ""
    escapped = False
    while not s.done():
        c = s.next()
        if escapped:
            x += c
            escapped = False
            continue
        if c == "\\":
            escapped = True
            continue
        if c == '"':
            return SLISPString(x)
        x += c
    raise Exception("syntax unterminated string")


def parse_symbol(s: StringStream):
    symbol = ""
    while not s.done():
        c = s.peek()
        if not c.isspace() and c not in '"()[]':
            symbol += c
            _ = s.next()
        else:
            return SLISPSymbol(symbol)
    return SLISPSymbol(symbol)


def parse_number(s: StringStream):
    num = ""
    in_decimal = False
    c = s.peek()
    if c == "-":
        num += c
        s.next()
    while not s.done():
        c = s.peek()
        if c == ".":
            if in_decimal:
                raise Exception("syntax unexpected .")
            in_decimal = True
            _ = s.next()
            num += "."
        elif c in "1234567890":
            num += c
            _ = s.next()
        else:
            return SLISPNumber(float(num))
    return SLISPNumber(float(num))


def parse_sexpr(s: StringStream):
    items = []
    item = ""
    while not s.done():
        c = s.next()
        if c == "(":
            items.append(parse_sexpr(s))
            continue
        if c == "[":
            items.append(parse_list(s))
            continue
        if c == ")":
            if len(item):
                items.append(item)
            return Sexpr(items[0], items[1:])
        if c == " ":
            if item:
                items.append(item)
                item = ""
            continue
        if c == '"':
            items.append(parse_string(s))
            continue
        if c in "123456790" or (c == "-" and s.peek() in "1234567890"):
            # print("number", c)
            s.step_back()
            items.append(parse_number(s))
            continue
        s.step_back()
        items.append(parse_symbol(s))
    raise Exception("syntax missing ')'")


def parse_list(s: StringStream):
    items = []
    item = ""
    while not s.done():
        c = s.next()
        if c == "(":
            items.append(parse_sexpr(s))
            continue
        if c == "[":
            items.append(parse_list(s))
        if c == "]":
            if len(item):
                items.append(item)
            return SLISPList(items)
        if c == " ":
            if item:
                items.append(item)
                item = ""
            continue
        if c == '"':
            items.append(parse_string(s))
            continue
        if c in "123456790" or (c == "-" and s.peek() in "1234567890"):
            # print("number", c)
            s.step_back()
            items.append(parse_number(s))
            continue
        s.step_back()
        items.append(parse_symbol(s))
    raise Exception("syntax missing ')'")


def parse_program(s: str):
    exprs = []
    s = StringStream(s)
    while not s.done():
        c = s.next()
        if c == "(":
            exprs.append(parse_sexpr(s))
    return exprs


def isatomic(x):
    return isinstance(x, SLISPNumber)


def islist(x):
    return isinstance(x, Sexpr) or isinstance(x, SLISPList)


global_vars = {"isp-version": SLISPNumber(1), "None": None}


def evaluate(x, locale):
    # print(f"Evaluating {x}")
    locale = locale or {}
    if isinstance(x, Sexpr):
        f = x.first
        if isinstance(x.first, SLISPSymbol):
            if f.value in keywords:
                return keywords[f.value](x.rest, locale)
            if f.value in locale:
                F = locale[f.value]
                if isinstance(F, SLISPFunction):
                    return F.execute(x.rest, locale)
            if f.value in global_vars:
                F = global_vars[f.value]
                if isinstance(F, SLISPFunction):
                    return F.execute(x.rest, locale)
        if isinstance(f, SLISPFunction):
            return f.execute(x.rest, locale)
        return x
    if isinstance(x, SLISPSymbol):
        if x.value in locale:
            return locale[x.value]
        if x.value in global_vars:
            return global_vars[x.value]
        raise Exception(f"Unkown vairable:  {x.value}")
    return x


def dyadic_operation_with_lists(op):
    def wrapper(rest, locale):
        if len(rest) != 2:
            raise Exception("length")
        f = evaluate(rest[0], locale)
        s = evaluate(rest[1], locale)
        if isatomic(f) and isatomic(s):
            return op(f, s)
        if isatomic(f) and islist(s):
            return SLISPList([op(f, S) for S in s])
        if islist(f) and isatomic(s):
            return SLISPList([op(F, s) for F in f])
        if islist(f) and islist(s):
            if len(f) != len(s):
                raise Exception("length")
            return SLISPList([op(F, S) for F, S in zip(f, s)])
        print(f, s)
        raise Exception(f"type got {type(f)} and {type(s)}")

    return wrapper


def slisp_set_value(rest, locale):
    if len(rest) != 2:
        raise Exception("length")
    f = rest[0]
    v = rest[1]
    if isinstance(f, SLISPSymbol):
        V = evaluate(v, locale)
        if locale:
            locale[f.value] = V
            return V
        global_vars[f.value] = V
        return V
    raise Exception("type")


def slisp_unset_value(rest, locale):
    if len(rest) != 1:
        raise Exception("length")
    f = rest[0]
    if isinstance(f, SLISPSymbol):
        del global_vars[f.value]
        return
    raise Exception("type")


def slisp_show(rest, locale):
    for x in rest:
        v = evaluate(x, locale)
        print(v.value)
        return v


def slisp_func_def(rest, locale):
    if len(rest) != 3:
        raise Exception("length")
    name, params, exprs = rest
    if not isinstance(name, SLISPSymbol):
        raise Exception("Function name must be a symbol")
    name = name.value
    if not isinstance(params, SLISPList):
        raise Exception("Function params must be a list of symbols")
    ps = []
    for p in params.values:
        if not isinstance(p, SLISPSymbol):
            raise Exception("Function params must be a list of symbols")
        ps.append(p.value)
    if isinstance(exprs, Sexpr):
        exprs = SLISPList([exprs])
    if not isinstance(exprs, SLISPList):
        raise Exception("Function params must be a list of Sexprs")
    exps = []
    for e in exprs.values:
        if not isinstance(e, Sexpr):
            raise Exception("Function params must be a list of Sexpr")
        exps.append(e)
    func = SLISPFunction(name, ps, exps)
    if locale:
        locale[func.name] = func
    else:
        global_vars[func.name] = func


def slisp_if(rest, locale):
    if len(rest) != 3:
        raise Exception("rank")
    b = evaluate(rest[0], locale)
    if b:
        return evaluate(rest[1], locale)
    return evaluate(rest[2], locale)


def slisp_til(rest, locale):
    if len(rest) != 1:
        raise Exception("rank")
    v = evaluate(rest[0], locale)
    if not isinstance(v, SLISPNumber):
        raise Exception("type")
    return SLISPList([SLISPNumber(x) for x in range(v.as_int())])


def slisp_map(rest, locale):
    if len(rest) != 2:
        raise Exception("rank")
    f = evaluate(rest[0], locale)
    l = evaluate(rest[1], locale)
    if not isinstance(l, SLISPList):
        raise Exception("type")
    if not isinstance(f, SLISPFunction):
        raise Exception("type")
    return SLISPList([evaluate(Sexpr(f, [x]), locale) for x in l])


def slisp_reduce(rest, locale):
    if len(rest) != 2:
        raise Exception("rank")
    f = evaluate(rest[0], locale)
    l = evaluate(rest[1], locale)
    if not isinstance(l, SLISPList):
        raise Exception("type")
    if not isinstance(f, SLISPFunction):
        raise Exception("type")
    if len(f.params) != 2:
        raise Exception("rank")
    if len(l) == 0:
        return None
    if len(l) == 1:
        return l[0]
    v = evaluate(Sexpr(f, [l[0], l[1]]), locale)
    for x in l[2:]:
        v = evaluate(Sexpr(f, [v, x]), locale)
    return v


def slisp_scan(rest, locale):
    if len(rest) != 3:
        raise Exception("rank")
    f = evaluate(rest[0], locale)
    s = evaluate(rest[1], locale)
    l = evaluate(rest[2], locale)
    if not isinstance(l, SLISPList):
        raise Exception("type")
    if not isinstance(f, SLISPFunction):
        raise Exception("type")
    if len(f.params) != 2:
        raise Exception("rank")
    if len(l) == 0:
        return None
    if len(l) == 1:
        return l
    v = [evaluate(Sexpr(f, [s, l[0]]), locale)]
    for x in l[1:]:
        v.append(evaluate(Sexpr(f, [v[-1], x]), locale))
    return SLISPList(v)


def slisp_each_prior(rest, locale):
    if len(rest) == 2:
        rest = [rest[0], None, rest[1]]
    if len(rest) != 3:
        raise Exception("rank")
    f = evaluate(rest[0], locale)
    s = evaluate(rest[1], locale)
    l = evaluate(rest[2], locale)
    if not isinstance(l, SLISPList):
        raise Exception("type")
    if not isinstance(f, SLISPFunction):
        raise Exception("type")
    if len(f.params) != 2:
        raise Exception("rank")

    if len(l) == 0:
        return None
    v = [evaluate(Sexpr(f, [s, l[0]]), locale)]
    for i in range(1, len(l)):
        v.append(evaluate(Sexpr(f, [l[i - 1], l[i]]), locale))
    return SLISPList(v)


def slisp_import(rest, locale):
    for x in rest:
        v = evaluate(x, locale)
        if isinstance(v, SLISPString):
            return run_file(v.value)
        raise Exception("import =type")


def slisp_read_zero(rest, locale):
    if len(rest) != 1:
        raise Exception("rank")
    v = evaluate(rest[0], locale)
    if isinstance(v, SLISPString):
        with open(v.value, "r") as f:
            return SLISPList([SLISPString(x.strip()) for x in f.readlines()])
    raise Exception("type")


def slisp_only(rest, locale):
    if len(rest) != 1:
        raise Exception("rank")
    v = evaluate(rest[0], locale)
    return v


def slisp_cast_num(rest, locale):
    if len(rest) != 1:
        raise Exception("rank")
    v = evaluate(rest[0], locale)
    if isinstance(v, SLISPString):
        if v.value == "":
            return None
        return SLISPNumber(float(v.value))
    return v


def slisp_enlist(rest, locale):
    if len(rest) != 1:
        raise Exception("rank")
    v = evaluate(rest[0], locale)
    if isinstance(v, SLISPList):
        return v
    return SLISPList([v])


def slisp_str(rest, locale):
    if len(rest) != 1:
        raise Exception("rank")
    v = evaluate(rest[0], locale)
    if isinstance(v, SLISPString):
        return v
    return SLISPString(str(v))


def slisp_os(rest, locale):
    if len(rest) != 1:
        raise Exception("rank")
    v = evaluate(rest[0], locale)
    if isinstance(v, SLISPString):
        import subprocess

        return SLISPList(
            SLISPString(x)
            for x in subprocess.run(v.value, shell=True, stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .split("\n")
        )
    return SLISPString(str(v))


def slisp_split(rest, locale):
    if len(rest) != 2:
        raise Exception("rank")
    f = evaluate(rest[0], locale)
    l = evaluate(rest[1], locale)
    if not isinstance(l, SLISPList):
        raise Exception("type")
    if len(l) == 0:
        return SLISPList([])
    r = []
    c = []
    for rec in l:
        R = evaluate(rec, locale)
        if R == f and c:
            r.append(SLISPList(c))
            c = []
            continue
        c.append(R)
    if c:
        r.append(SLISPList(c))
    return SLISPList(r)


def slisp_sort(desc=False):
    def wrapper(rest, locale):
        if len(rest) != 1:
            raise Exception("rank")
        v = evaluate(rest[0], locale)
        if isinstance(v, SLISPList):
            return SLISPList(sorted(v.values, reverse=desc))

    return wrapper


def slisp_take(rest, locale):
    if len(rest) != 2:
        raise Exception("rank")
    n = evaluate(rest[0], locale)
    ls = evaluate(rest[1], locale)
    if not isinstance(n, SLISPNumber):
        raise Exception("type")
    return SLISPList(ls.values[: int(n.value)])


def slisp_join(rest, locale):
    r = []
    for x in rest:
        v = evaluate(x, locale)
        if isinstance(v, SLISPString):
            r.extend(v.value)
        if isinstance(v, SLISPList):
            r.extend(v.values)
    if min([isinstance(x, str) for x in r]):
        r = "".join(r)
        return SLISPString(r)
    return SLISPList(r)


def slisp_call(rest, locale):
    if len(rest) != 2:
        raise Exception("rank")
    n = evaluate(rest[0], locale) if not isinstance(rest[0], SLISPSymbol) else rest[0]
    ls = evaluate(rest[1], locale)
    return evaluate(Sexpr(n, ls), locale)


keywords = {
    "+": dyadic_operation_with_lists(lambda x, y: x + y),
    "-": dyadic_operation_with_lists(lambda x, y: x - y),
    "*": dyadic_operation_with_lists(lambda x, y: x * y),
    "%": dyadic_operation_with_lists(lambda x, y: x / y),
    "|": dyadic_operation_with_lists(lambda x, y: x if x > y else y),
    "&": dyadic_operation_with_lists(lambda x, y: x if x < y else y),
    "=": dyadic_operation_with_lists(lambda x, y: x == y),
    "mod": dyadic_operation_with_lists(lambda x, y: y % x),
    "fill": dyadic_operation_with_lists(lambda x, y: x if y is None else y),
    "set": slisp_set_value,
    "del": slisp_unset_value,
    "show": slisp_show,
    "fn": slisp_func_def,
    "if": slisp_if,
    "til": slisp_til,
    "map": slisp_map,
    "reduce": slisp_reduce,
    "scan": slisp_scan,
    "import": slisp_import,
    "read0": slisp_read_zero,
    "only": slisp_only,
    "prior": slisp_each_prior,
    "castnum": slisp_cast_num,
    "split": slisp_split,
    "desc": slisp_sort(desc=True),
    "asc": slisp_sort(desc=False),
    "take": slisp_take,
    "join": slisp_join,
    "call": slisp_call,
    "enlist": slisp_enlist,
    "str": slisp_str,
    "os": slisp_os,
}


def repl():
    while True:
        try:
            x = input("> ").strip()
            if x == "\\\\":
                return
            e = parse_program(x)
            r = None
            for E in e:
                r = evaluate(E, {})
            print(r)
        except Exception as e:
            print(e)


imported = []


def run_file(p):
    if p in imported:
        return None
    with open(p, "r") as f:
        imported.append(p)
        prog = parse_program(f.read().replace("\n", " "))
        r = None
        for E in prog:
            r = evaluate(E, {})
        return r


if __name__ == "__main__":
    print("ISP Version 0.0.1")
    run_file("stdlib.slisp")
    repl()
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, TextIO, Tuple

from icu_rbnf import spellout

EPS = "<eps>"
SPACE = "<space>"

STATUS_EMPTY = 0
STATUS_WORD = 1
STATUS_SPACE = 2

TEMPLATE_CHARS: Set[str] = {
    "[",
    "]",
    "(",
    ")",
    "{",
    "}",
    "|",
}

# -----------------------------
# Finite State Transducer (FST)
# -----------------------------


@dataclass
class FstArc:
    to_state: int
    in_label: str = EPS
    out_label: str = EPS
    log_prob: Optional[float] = None


@dataclass
class Fst:
    arcs: Dict[int, List[FstArc]] = field(default_factory=lambda: defaultdict(list))
    states: Set[int] = field(default_factory=lambda: {0})
    final_states: Set[int] = field(default_factory=set)
    words: Set[str] = field(default_factory=set)
    output_words: Set[str] = field(default_factory=set)
    start: int = 0
    current_state: int = 0

    def next_state(self) -> int:
        self.states.add(self.current_state)
        self.current_state += 1
        return self.current_state

    def next_edge(
        self,
        from_state: int,
        in_label: Optional[str] = None,
        out_label: Optional[str] = None,
        log_prob: Optional[float] = None,
    ) -> int:
        to_state = self.next_state()
        self.add_edge(from_state, to_state, in_label, out_label, log_prob)
        return to_state

    def add_edge(  # pylint: disable=too-many-positional-arguments
        self,
        from_state: int,
        to_state: int,
        in_label: Optional[str] = None,
        out_label: Optional[str] = None,
        log_prob: Optional[float] = None,
    ) -> None:
        if in_label is None:
            in_label = EPS

        if out_label is None:
            out_label = in_label

        if (" " in in_label) or (" " in out_label):
            raise ValueError(
                f"Cannot have white space in labels: from={in_label}, to={out_label}"
            )

        if (not in_label) or (not out_label):
            raise ValueError(f"Labels cannot be empty: from={in_label}, to={out_label}")

        if in_label != EPS:
            self.words.add(in_label)

        if out_label != EPS:
            self.output_words.add(out_label)

        self.states.add(from_state)
        self.states.add(to_state)
        self.arcs[from_state].append(FstArc(to_state, in_label, out_label, log_prob))

    def accept(self, state: int) -> None:
        self.states.add(state)
        self.final_states.add(state)

    def write(self, fst_file: TextIO, symbols_file: Optional[TextIO] = None) -> None:
        symbols = {EPS: 0}

        for state, arcs in self.arcs.items():
            for arc in arcs:
                if arc.in_label not in symbols:
                    symbols[arc.in_label] = len(symbols)

                if arc.out_label not in symbols:
                    symbols[arc.out_label] = len(symbols)

                if arc.log_prob is None:
                    print(
                        state, arc.to_state, arc.in_label, arc.out_label, file=fst_file
                    )
                else:
                    print(
                        state,
                        arc.to_state,
                        arc.in_label,
                        arc.out_label,
                        arc.log_prob,
                        file=fst_file,
                    )

        for state in self.final_states:
            print(state, file=fst_file)

        if symbols_file is not None:
            for symbol, symbol_id in symbols.items():
                print(symbol, symbol_id, file=symbols_file)


# ----------------------------
# AST nodes
# ----------------------------


@dataclass
class Node:
    pass


@dataclass
class SequenceNode(Node):
    parts: List[Node]
    separators: List[bool]  # True if there was whitespace before the corresponding part


@dataclass
class LiteralNode(Node):
    text: str  # single literal chunk, no whitespace


@dataclass
class OptionalNode(Node):
    child: Node


@dataclass
class AlternativesNode(Node):
    options: List[Node]


@dataclass
class ListRefNode(Node):
    name: str


@dataclass
class NumberRangeNode(Node):
    """Represents a single number, range, or comma-separated list of both.

    items: List of tuples - either (number,) for single numbers or (start, end, step) for ranges.
           Step defaults to 1 if not specified in a range.
    """

    items: List[Tuple[int, ...]]  # (number,) or (start, end, step)


# ----------------------------
# Parser
# ----------------------------


class TemplateParser:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0

    def parse(self) -> Node:
        node = self._parse_sequence(stop_chars=set())
        trailing_ws = self._consume_ws()
        if trailing_ws:
            # trailing whitespace is ignored
            pass
        if self.pos != len(self.text):
            raise ValueError(f"Unexpected trailing text at position {self.pos}")
        return node

    def _peek(self) -> Optional[str]:
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def _take(self) -> str:
        if self.pos >= len(self.text):
            raise ValueError("Unexpected end of input")
        ch = self.text[self.pos]
        self.pos += 1
        return ch

    def _expect(self, expected: str) -> None:
        actual = self._take()
        if actual != expected:
            raise ValueError(
                f"Expected '{expected}' at position {self.pos - 1}, got '{actual}'"
            )

    def _consume_ws(self) -> bool:
        had_ws = False
        # while self._peek() is not None and self._peek().isspace():
        while True:
            peek = self._peek()
            if (peek is None) or (not peek.isspace()):
                break
            had_ws = True
            self.pos += 1
        return had_ws

    def _parse_sequence(self, stop_chars: Set[str]) -> Node:
        parts: List[Node] = []
        separators: List[bool] = []

        first = True
        while True:
            had_ws = self._consume_ws()
            ch = self._peek()

            if ch is None or ch in stop_chars:
                break

            if not first:
                separators.append(had_ws)

            if ch == "[":
                parts.append(self._parse_optional())
            elif ch == "(":
                parts.append(self._parse_alternatives())
            elif ch == "{":
                parts.append(self._parse_list_ref())
            else:
                parts.append(self._parse_literal())

            first = False

        if not parts:
            return SequenceNode([], [])

        if len(parts) == 1:
            return parts[0]

        return SequenceNode(parts, separators)

    def _parse_optional(self) -> Node:
        self._expect("[")
        child = self._parse_sequence(stop_chars={"]"})
        self._expect("]")
        return OptionalNode(child)

    def _parse_alternatives(self) -> Node:
        self._expect("(")
        options: List[Node] = []

        while True:
            option = self._parse_sequence(stop_chars={"|", ")"})
            options.append(option)

            ch = self._peek()
            if ch == "|":
                self._take()
                continue
            if ch == ")":
                break
            raise ValueError(f"Expected '|' or ')' at position {self.pos}")

        self._expect(")")
        return AlternativesNode(options)

    def _parse_list_ref(self) -> Node:
        self._expect("{")
        start_pos = self.pos

        while True:
            ch = self._peek()
            if ch is None:
                raise ValueError("Unterminated '{'")
            if ch == "}":
                break
            self.pos += 1

        content = self.text[start_pos : self.pos].strip()
        self._expect("}")

        if not content:
            raise ValueError("Empty list reference {} is not allowed")

        # Parse comma-separated list of numbers and ranges
        # Supports: 42, 1..10, 1..10/2, 1,2,3..10, 1..10/2,20..50/5, etc.
        items: List[Tuple[int, ...]] = []
        parts = [p.strip() for p in content.split(",")]

        for part in parts:
            if not part:
                raise ValueError(f"Empty part in number list: {content}")

            # Check for range with optional step: start..end or start..end/step
            m_range = re.fullmatch(r"(-?\d+)\s*\.\.\s*(-?\d+)(/\s*(-?\d+))?", part)
            if m_range:
                start = int(m_range.group(1))
                end = int(m_range.group(2))
                step_str = m_range.group(4)
                step = int(step_str) if step_str else 1
                if step == 0:
                    raise ValueError(f"Step cannot be zero: {part}")
                items.append((start, end, step))
                continue

            # Check for single number
            m_num = re.fullmatch(r"(-?\d+)", part)
            if m_num:
                items.append((int(m_num.group(1)),))
                continue

            raise ValueError(f"Invalid number/range format: {part!r}")

        if not items:
            raise ValueError(f"Invalid number list format: {content!r}")

        return NumberRangeNode(items)

    def _parse_literal(self) -> Node:
        start = self.pos
        while True:
            ch = self._peek()
            if ch is None or ch.isspace() or ch in "[](){}|":
                break
            self.pos += 1

        text = self.text[start : self.pos]
        if not text:
            raise ValueError(f"Expected literal at position {self.pos}")

        return LiteralNode(text)


# ----------------------------
# Analysis helpers
# ----------------------------


def _node_can_be_empty(node: Node) -> bool:
    if isinstance(node, SequenceNode):
        return all(_node_can_be_empty(part) for part in node.parts)

    if isinstance(node, LiteralNode):
        return False

    if isinstance(node, OptionalNode):
        return True

    if isinstance(node, AlternativesNode):
        return any(_node_can_be_empty(option) for option in node.options)

    if isinstance(node, (ListRefNode, NumberRangeNode)):
        return False

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


def _node_can_start_with_word(node: Node) -> bool:
    if isinstance(node, SequenceNode):
        for part in node.parts:
            if _node_can_start_with_word(part):
                return True
            if not _node_can_be_empty(part):
                return False
        return False

    if isinstance(node, LiteralNode):
        return True

    if isinstance(node, OptionalNode):
        return _node_can_start_with_word(node.child)

    if isinstance(node, AlternativesNode):
        return any(_node_can_start_with_word(option) for option in node.options)

    if isinstance(node, (ListRefNode, NumberRangeNode)):
        return True

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


# ----------------------------
# FST helpers
# ----------------------------


def _add_word(fst: Fst, state: int, word: str) -> int:
    if not word:
        raise ValueError("Word cannot be empty")
    if any(ch.isspace() for ch in word):
        raise ValueError(f"Word cannot contain whitespace: {word!r}")

    for char in word:
        state = fst.next_edge(state, in_label=char, out_label=char)

    return state


def _add_space(fst: Fst, state: int) -> int:
    return fst.next_edge(state, in_label=SPACE, out_label=SPACE)


def _expand_list_value(value: str) -> List[str]:
    words = value.strip().split()
    if not words:
        raise ValueError(f"List value must not be empty: {value!r}")
    return words


def _expand_ref(
    node: Node, list_values: Dict[str, Sequence[str]], locale: str
) -> List[List[str]]:
    if isinstance(node, ListRefNode):
        if node.name not in list_values:
            raise KeyError(f"Missing values for list '{node.name}'")
        return [_expand_list_value(v) for v in list_values[node.name]]

    if isinstance(node, NumberRangeNode):
        range_words = []
        for item in node.items:
            if len(item) == 1:
                # Single number
                num = item[0]
                num_words = spellout(num, locale)
                num_words = num_words.replace("-", " ")
                range_words.append(num_words.split())
            else:
                # Range with step: (start, end, step)
                start, end, step = item
                # Determine direction based on start vs end
                if start <= end:
                    step = abs(step)
                else:
                    step = -abs(step)

                for num in range(start, end + step, step):
                    num_words = spellout(num, locale)
                    num_words = num_words.replace("-", " ")
                    range_words.append(num_words.split())

        return range_words

    raise TypeError(f"Unsupported reference node: {type(node).__name__}")


# ----------------------------
# Compiler
# ----------------------------

Config = Tuple[int, int]  # (state, status)


def _compile_node(
    fst: Fst,
    node: Node,
    configs: Set[Config],
    list_values: Dict[str, Sequence[str]],
    locale: str,
) -> Set[Config]:
    out: Set[Config]
    if isinstance(node, SequenceNode):
        current = set(configs)

        for i, part in enumerate(node.parts):
            if i > 0 and node.separators[i - 1]:
                current = _maybe_insert_space_before_part(fst, current, part)

            current = _compile_node(fst, part, current, list_values, locale)

        return current

    if isinstance(node, LiteralNode):
        out = set()
        for state, _status in configs:
            new_state = _add_word(fst, state, node.text)
            out.add((new_state, STATUS_WORD))
        return out

    if isinstance(node, OptionalNode):
        taken = _compile_node(fst, node.child, set(configs), list_values, locale)
        return set(configs) | taken

    if isinstance(node, AlternativesNode):
        out = set()
        for option in node.options:
            out |= _compile_node(fst, option, set(configs), list_values, locale)
        return out

    if isinstance(node, (ListRefNode, NumberRangeNode)):
        expansions = _expand_ref(node, list_values, locale)
        out = set()

        for state, status in configs:
            for words in expansions:
                cur_state = state
                cur_status = status
                for j, word in enumerate(words):
                    if j > 0:
                        cur_state = _add_space(fst, cur_state)
                        cur_status = STATUS_SPACE
                    cur_state = _add_word(fst, cur_state, word)
                    cur_status = STATUS_WORD
                out.add((cur_state, cur_status))

        return out

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


def _maybe_insert_space_before_part(
    fst: Fst,
    configs: Set[Config],
    part: Node,
) -> Set[Config]:
    """
    Insert a single explicit SPACE token only when needed:
    - there was whitespace in the template between previous and next part
    - the current path ends with a word
    - the next part can begin with a word

    This is what makes:
        off [the] light
    work correctly while:
        light[s]
    does not get a space before [s].
    """
    if not _node_can_start_with_word(part):
        return configs

    out: Set[Config] = set()
    for state, status in configs:
        if status == STATUS_WORD:
            spaced_state = _add_space(fst, state)
            out.add((spaced_state, STATUS_SPACE))
        else:
            out.add((state, status))
    return out


# ----------------------------
# Public API
# ----------------------------


def templates_to_fst(
    templates: List[str],
    locale: str,
    list_values: Optional[Dict[str, Sequence[str]]] = None,
) -> Fst:
    """
    Parse a template string and compile it into an FST.

    Supported syntax:
      [ ... ]         optional
      (a|b|c)         alternatives
      {name}          named external list
      {1..100}        numeric range (default step 1)
      {0..100/10}     numeric range with step
      {0,1,2,3..10}   comma-separated numbers and ranges

    Spacing:
      Source whitespace is preserved as a single explicit SPACE token
      between adjacent emitted words, including across omitted optionals.

    Examples:
      "turn off [the] light[s] now"
      "set color to (red|green|blue)"
      "turn on {name}"
      "set brightness to {1..100}"
      "set value to {0..100/10}"
      "choose from {0,1,2,3..10,15..100/10}"
    """
    if list_values is None:
        list_values = {}

    fst = Fst()

    for template in templates:
        parser = TemplateParser(template)
        ast = parser.parse()

        template_state = fst.next_edge(fst.start, EPS, EPS)

        end_configs = _compile_node(
            fst, ast, {(template_state, STATUS_EMPTY)}, list_values, locale
        )

        for state, _status in end_configs:
            fst.accept(state)

    return fst

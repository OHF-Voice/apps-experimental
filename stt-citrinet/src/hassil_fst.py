import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Set, TextIO, Tuple

from icu_rbnf import spellout


class TokenizerLike(Protocol):
    def text_to_ids(self, text: str) -> List[int]: ...  # noqa: E704


EPS = "<eps>"

STATUS_EMPTY = 0
STATUS_WORD = 1


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
        self._consume_ws()
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

        items: List[Tuple[int, ...]] = []
        parts = [p.strip() for p in content.split(",")]

        for part in parts:
            if not part:
                raise ValueError(f"Empty part in number list: {content}")

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

            m_num = re.fullmatch(r"(-?\d+)", part)
            if m_num:
                items.append((int(m_num.group(1)),))
                continue

            raise ValueError(f"Invalid number/range format: {part!r}")

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
# Token helpers
# ----------------------------


def _token_label(token_id: int) -> str:
    return f"t_{token_id}"


def _add_token_ids(fst: Fst, state: int, token_ids: Sequence[int]) -> int:
    if not token_ids:
        raise ValueError("Tokenizer returned no ids")

    for token_id in token_ids:
        label = _token_label(int(token_id))
        state = fst.next_edge(state, in_label=label, out_label=label)

    return state


def _tokenize_word(tokenizer: TokenizerLike, word: str) -> List[int]:
    if not word:
        raise ValueError("Word cannot be empty")
    if any(ch.isspace() for ch in word):
        raise ValueError(f"Expected a single word, got: {word!r}")

    token_ids = tokenizer.text_to_ids(word)
    if not token_ids:
        raise ValueError(f"Tokenizer returned no ids for word: {word!r}")

    return token_ids


def _tokenize_words(tokenizer: TokenizerLike, words: Sequence[str]) -> List[int]:
    if not words:
        raise ValueError("Expected at least one word")

    for word in words:
        if not word:
            raise ValueError("Word cannot be empty")
        if any(ch.isspace() for ch in word):
            raise ValueError(f"Expected pre-split words, got: {word!r}")

    # Important: tokenize the complete phrase so word boundaries are seen
    # by the tokenizer normally, but only after we've assembled whole words.
    text = " ".join(words)
    token_ids = tokenizer.text_to_ids(text)
    if not token_ids:
        raise ValueError(f"Tokenizer returned no ids for text: {text!r}")

    return token_ids


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
                num = item[0]
                num_words = spellout(num, locale).replace("-", " ")
                range_words.append(num_words.split())
            else:
                start, end, step = item
                if start <= end:
                    step = abs(step)
                else:
                    step = -abs(step)

                for num in range(start, end + step, step):
                    num_words = spellout(num, locale).replace("-", " ")
                    range_words.append(num_words.split())

        return range_words

    raise TypeError(f"Unsupported reference node: {type(node).__name__}")


# ----------------------------
# Text expansion helpers
# ----------------------------

Config = Tuple[int, int]  # (state, status)


def _compile_text_words(
    fst: Fst,
    configs: Set[Config],
    words: Sequence[str],
    tokenizer: TokenizerLike,
    prepend_space_if_needed: bool,
) -> Set[Config]:
    if not words:
        return set(configs)

    out: Set[Config] = set()

    for state, status in configs:
        actual_words = list(words)

        # Preserve template whitespace semantics:
        # if the prior emitted thing ended in a word and template syntax
        # says a separator is needed, realize that as a normal space in the
        # text fed to the tokenizer.
        if prepend_space_if_needed and status == STATUS_WORD:
            token_ids = tokenizer.text_to_ids(" " + " ".join(actual_words))
            if not token_ids:
                raise ValueError(
                    f"Tokenizer returned no ids for text: {' '.join(actual_words)!r}"
                )
        else:
            token_ids = _tokenize_words(tokenizer, actual_words)

        new_state = _add_token_ids(fst, state, token_ids)
        out.add((new_state, STATUS_WORD))

    return out


# ----------------------------
# Compiler
# ----------------------------


def _compile_node(  # pylint: disable=too-many-positional-arguments
    fst: Fst,
    node: Node,
    configs: Set[Config],
    list_values: Dict[str, Sequence[str]],
    locale: str,
    tokenizer: TokenizerLike,
    needs_leading_space: bool = False,
) -> Set[Config]:
    out: Set[Config]

    if isinstance(node, SequenceNode):
        current = set(configs)

        i = 0
        while i < len(node.parts):
            part = node.parts[i]

            needs_space = (i > 0) and node.separators[i - 1]

            # --- Detect suffix optional: minute[s] ---
            if (
                isinstance(part, LiteralNode)
                and i + 1 < len(node.parts)
                and isinstance(node.parts[i + 1], OptionalNode)
                and isinstance(
                    node.parts[i + 1].child, LiteralNode  # type: ignore[attr-defined]
                )
                and not (
                    i < len(node.separators) and node.separators[i]
                )  # no space before optional
            ):
                base = part.text
                suffix = node.parts[i + 1].child.text  # type: ignore[attr-defined]

                variants = [
                    [base],
                    [base + suffix],
                ]

                new_configs = set()
                for words in variants:
                    new_configs |= _compile_text_words(
                        fst,
                        current,
                        words,
                        tokenizer,
                        prepend_space_if_needed=needs_space,
                    )

                current = new_configs
                i += 2
                continue

            # --- Normal path ---
            current = _compile_node(
                fst,
                part,
                current,
                list_values,
                locale,
                tokenizer,
                needs_leading_space=needs_space,
            )

            i += 1

        return current

    if isinstance(node, LiteralNode):
        return _compile_text_words(
            fst,
            configs,
            [node.text],
            tokenizer,
            prepend_space_if_needed=needs_leading_space,
        )

    if isinstance(node, OptionalNode):
        taken = _compile_node(
            fst,
            node.child,
            set(configs),
            list_values,
            locale,
            tokenizer,
            needs_leading_space=needs_leading_space,
        )
        return set(configs) | taken

    if isinstance(node, AlternativesNode):
        out = set()
        for option in node.options:
            out |= _compile_node(
                fst,
                option,
                set(configs),
                list_values,
                locale,
                tokenizer,
                needs_leading_space=needs_leading_space,
            )
        return out

    if isinstance(node, (ListRefNode, NumberRangeNode)):
        expansions = _expand_ref(node, list_values, locale)
        out = set()
        for words in expansions:
            out |= _compile_text_words(
                fst,
                configs,
                words,
                tokenizer,
                prepend_space_if_needed=needs_leading_space,
            )
        return out

    raise TypeError(f"Unsupported node type: {type(node).__name__}")


# ----------------------------
# Public API
# ----------------------------


def templates_to_fst(
    templates: List[str],
    tokenizer: TokenizerLike,
    locale: str,
    list_values: Optional[Dict[str, Sequence[str]]] = None,
) -> Fst:
    """
    Parse templates and compile them into an FST whose labels are tokenizer ids
    encoded as strings like "t_123".

    Important behavior:
    - literals are tokenized as complete words, never character-by-character
    - explicit SPACE arcs are no longer emitted
    - when template whitespace is required between emitted words, that whitespace
      is included only in the text passed to tokenizer.text_to_ids(...)
    - list values and spoken-out numbers are split into words first, then
      tokenized as full phrases so the tokenizer sees normal word boundaries
    """
    if list_values is None:
        list_values = {}

    fst = Fst()

    for template in templates:
        parser = TemplateParser(template)
        ast = parser.parse()

        template_state = fst.next_edge(fst.start, EPS, EPS)

        end_configs = _compile_node(
            fst,
            ast,
            {(template_state, STATUS_EMPTY)},
            list_values,
            locale,
            tokenizer,
        )

        for state, _status in end_configs:
            fst.accept(state)

    return fst

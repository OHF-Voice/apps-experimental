"""GBNF grammar constraining Gemma tool-call output to the loaded tools.

The grammar mirrors what ``tools.parse_tool_calls`` accepts::

    <|tool_call>call:turn_off{area:<|"|>Basement<|"|>,domain:<|"|>light<|"|>}<tool_call|>

Constrained decoding makes malformed calls, unknown tool names, and off-enum
slot values (e.g. name="Natalie's Bedroom" when no such entity exists)
impossible at the sampling stage. Enum slots become literal alternations;
numeric slots become number rules; free-string slots stay unconstrained. A
free-text alternative (any output not starting a tool call) is kept so the
model can still decline ("I don't understand") exactly as it does without the
grammar.

Grammars affect sampling only: they do not touch the prompt, so the persisted
tools-prefix cache stays valid and enabling/disabling the grammar does not
reshuffle unconstrained decoding.
"""

from typing import Any, Dict, List

# Gemma wraps string values in <|"|> pseudo-quote tokens.
_SHARED_RULES = """\
q ::= "<|\\"|>"
num ::= "-"? [0-9]+ ("." [0-9]+)?
fstr ::= [^<]+
ws ::= [ \\t\\n]*
toolcall ::= "<|tool_call>call:" call "<tool_call|>"
freetext ::= [^<] [^<]*
root ::= (ws toolcall)+ ws | freetext
"""


def _lit(text: str) -> str:
    """A GBNF literal for exact text."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _value_rule(schema: Dict[str, Any]) -> str:
    """The GBNF expression for one slot's value."""
    enum = schema.get("enum")
    if enum:
        alts = " | ".join(_lit(str(value)) for value in enum)
        return f"q ({alts}) q"
    if schema.get("type") in ("integer", "number"):
        return "num"
    return "q fstr q"


def build_gbnf(tools: List[Dict[str, Any]]) -> str:
    """Build the full GBNF grammar for a list of OpenAI-style tools."""
    lines = [_SHARED_RULES]
    call_rules = []
    for index, tool in enumerate(tools):
        function = tool.get("function") or {}
        tool_name = function.get("name") or ""
        props = (function.get("parameters") or {}).get("properties") or {}

        rule = f"c{index}"
        call_rules.append(rule)
        arg_alts = []
        for arg_index, (slot, schema) in enumerate(sorted(props.items())):
            value_rule = f"{rule}-v{arg_index}"
            lines.append(f"{value_rule} ::= {_value_rule(schema or {})}")
            arg_alts.append(f"{_lit(slot + ':')} {value_rule}")

        if arg_alts:
            arg_rule = f"{rule}-arg"
            lines.append(f"{arg_rule} ::= {' | '.join(arg_alts)}")
            args_expr = f'({arg_rule} ("," {arg_rule})*)?'
        else:
            args_expr = ""
        lines.append(
            f'{rule} ::= {_lit(tool_name + "{")} {args_expr} "}}"'.replace("  ", " ")
        )

    lines.append(f"call ::= {' | '.join(call_rules)}")
    return "\n".join(lines) + "\n"

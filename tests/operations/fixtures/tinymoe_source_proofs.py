"""Independent source proof helpers, extracted without semantic changes.

Source: TinyMoE product adde6ca52b556df8585edfb6c7ade3a241964620,
research/latch_tinymoe/elementwise_probe.py::close_brace and
research/latch_tinymoe/pointer_source_proof.py. No compiler-pass imports.
"""
import re

def close_brace(source, opening):
    """Independent brace scan ignoring comments and quoted literals."""
    token = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    depth = 0
    for match in token.finditer(source, opening):
        if match[0] == "{": depth += 1
        elif match[0] == "}": depth -= 1
        if depth == 0:
            return match.start()
    raise AssertionError("unclosed generated scope")


DECLARATION = re.compile(r"^(?:float|double|__nv_bfloat16|int64_t|bool)\* (v_\w+) = [^\n]+;\n", re.M)


def original_declarations(prelude):
    matches = list(DECLARATION.finditer(prelude))
    result = {m[1]: m[0] for m in matches}
    assert len(result) == len(matches), "duplicate original declaration"
    return result


def relocated_names(prefix, original_prelude, marker, phase):
    """Require exact original types/RHS/order and the ordinary opaque bases."""
    original = original_declarations(original_prelude) if isinstance(original_prelude, str) else original_prelude
    actual = list(DECLARATION.finditer(prefix))
    names = [m[1] for m in actual]
    assert names and len(names) == len(set(names)), "missing or duplicate moved declaration"
    for match in actual:
        assert match[1] in original and match[0] == original[match[1]], "moved declaration type/RHS changed"
    selected = set(names)
    text = "".join(line for name, line in original.items() if name in selected)
    bases = []
    if "latch_pa[" in text: bases.append("float** latch_pa = latch_opq(args);")
    if "latch_pr +" in text: bases.append("float* latch_pr = latch_opq(arena);")
    if "latch_po[" in text: bases.append("float** latch_po = latch_opq(out);")
    expected = f"// {marker}: {phase}\n" + " ".join(bases) + "\n" + text + f"// {marker}: end\n"
    assert prefix == expected, "moved opaque base, declaration order or unparsed prefix changed"
    return selected


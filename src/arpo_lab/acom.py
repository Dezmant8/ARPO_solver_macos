"""Auditable C89 scaffolding, separate from the measured task implementation."""

import re
from pathlib import Path

CODING_STYLE = """# Common implementation requirements

Use ANSI C89 and the exact public ABI supplied by the project.
Use descriptive names and consistent indentation. Keep functions focused.
Write one operation per line; avoid compressed blocks and unexplained constants.
Use short comments for ownership, invariants and non-obvious decisions, not every line.
Do not change published IID values, method order, signatures or calling conventions.
Obtain unrelated interfaces with QueryInterface, not pointer casts.
Every acquired interface reference needs a matching Release or an explicit ownership transfer.
On failure, clear interface output pointers, return an error and release temporary resources.
Keep the supported interface set static and return one canonical IEcoUnknown.
Reject unsupported aggregation. Keep task algorithms out of the shared framework.
"""


def identifiers(project, stem):
    text = (project / "SharedFiles" / f"Id{stem}.h").read_text(encoding="utf-8-sig")
    match = re.search(r"GetIEcoComponentFactoryPtr_([A-Fa-f0-9]{32})", text)
    if not match or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", stem):
        raise ValueError("Cannot identify the generated ACOM component safely")
    return {"@STEM@": stem, "@GUID@": match[1]}


def render(template, values):
    text = (Path(__file__).with_name("templates") / template).read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(key, value)
    if re.search(r"@[A-Z]+@", text):
        raise ValueError("Unresolved ACOM template placeholder")
    return text


def prepare_scaffold(project, stem):
    """Replace only wizard implementation files; leave vendor SDK and public IID intact."""
    values = identifiers(project, stem)
    public = (project / "SharedFiles" / f"I{stem}.h").read_text(encoding="utf-8-sig")
    if not re.search(r"\*MyFunction\)\([^;]+char_t\*\* CopyName\);", public):
        raise ValueError("Wizard ABI changed; review the common scaffold before using it")
    for template, target in [
        ("component.h.in", f"HeaderFiles/C{stem}.h"),
        ("factory.h.in", f"HeaderFiles/C{stem}Factory.h"),
        ("component.c.in", f"SourceFiles/C{stem}.c"),
        ("factory.c.in", f"SourceFiles/C{stem}Factory.c"),
        ("console.c.in", f"UnitTestFiles/SourceFiles/{stem}.c"),
    ]:
        (project / target).write_text(render(template, values), encoding="utf-8")
    (project / "CODING_STYLE.md").write_text(CODING_STYLE, encoding="utf-8")


def write_contract_check(project, stem, target):
    """The trusted client exercises only object/interface behavior, no task algorithm."""
    target.write_text(render("acom-check.c.in", identifiers(project, stem)), encoding="utf-8")

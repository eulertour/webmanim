import operator as op
from functools import reduce

def get_modified_expression(tex_string, alignment=""):
    result = alignment + " " + tex_string
    result = result.strip()
    result = modify_special_strings(result)
    return result

def modify_special_strings(tex):
    tex = remove_stray_braces(tex)
    should_add_filler = reduce(op.or_, [
        # Fraction line needs something to be over
        tex == "\\over",
        tex == "\\overline",
        # Makesure sqrt has overbar
        tex == "\\sqrt",
        # Need to add blank subscript or superscript
        tex.endswith("_"),
        tex.endswith("^"),
        tex.endswith("dot"),
    ])
    if should_add_filler:
        filler = "{\\quad}"
        tex += filler

    if tex == "\\substack":
        tex = "\\quad"

    if tex == "":
        tex = "\\quad"

    # Handle imbalanced \left and \right
    num_lefts, num_rights = [
        len([
            s for s in tex.split(substr)[1:]
            if s and s[0] in "(){}[]|.\\"
        ])
        for substr in ("\\left", "\\right")
    ]
    if num_lefts != num_rights:
        tex = tex.replace("\\left", "\\big")
        tex = tex.replace("\\right", "\\big")

    for context in ["array"]:
        begin_in = ("\\begin{%s}" % context) in tex
        end_in = ("\\end{%s}" % context) in tex
        if begin_in ^ end_in:
            # Just turn this into a blank string,
            # which means caller should leave a
            # stray \\begin{...} with other symbols
            tex = ""
    return tex

def remove_stray_braces(tex):
    """
    Makes TexMobject resiliant to unmatched { at start
    """
    num_lefts, num_rights = [
        tex.count(char)
        for char in "{}"
    ]
    while num_rights > num_lefts:
        tex = "{" + tex
        num_lefts += 1
    while num_lefts > num_rights:
        tex = tex + "}"
        num_rights += 1
    return tex


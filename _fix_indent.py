#!/usr/bin/env python3
"""Comprehensive indentation fix for constructive_sequence_builder.py"""
with open("src/aps_cp_sat/model/constructive_sequence_builder.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

fixed_lines = []
for i, l in enumerate(lines):
    ln = i + 1
    stripped = l.lstrip()
    spaces = len(l) - len(stripped)

    # Determine what this line SHOULD be based on context
    # Structure:
    # - Lines ~55-59: ConstructiveChain class methods (4 spaces for def, 8 for body)
    # - Lines ~73-101: ConstructiveBuildResult class methods (4 for def, 8 for body)
    # - Lines ~115-156: TemplateEdgeGraph.__init__ (4 for def, 8 for body)
    # - Lines ~157+: TemplateEdgeGraph class methods (8 for def, 12 for body)
    # - Lines ~310+: Standalone functions (4 for def, 8 for body)

    # Rules:
    # 1. If it's a `def` line inside a class: 8 spaces (TemplateEdgeGraph methods)
    # 2. If it's a `def` line inside a standalone function: 4 spaces
    # 3. If it's inside TemplateEdgeGraph.__init__ (lines 118-156): 8 spaces for body
    # 4. If it's a standalone function body: 8 spaces
    # 5. If it's a class method body of ConstructiveChain/ConstructiveBuildResult: 8 spaces

    if ln < 117:
        # Lines 1-116: mostly fine from earlier fixes, but check for issues
        # Line 57: body of if inside __post_init__ needs 12 spaces
        if ln == 57:
            if spaces != 12 and stripped:
                l = "            " + stripped
        # Lines 88-101: ConstructiveBuildResult class methods
        elif ln == 88:
            if spaces == 8 and stripped.startswith("def "):
                l = "    " + stripped
        elif ln == 89:
            if spaces != 8 and stripped:
                l = "        " + stripped
        elif ln == 90:
            if spaces != 8 and stripped:
                l = "        " + stripped
        elif ln == 91:
            if spaces != 8 and stripped.startswith("for "):
                l = "        " + stripped
        elif ln == 92:
            if spaces != 12 and stripped:
                l = "            " + stripped
        elif ln == 93:
            if spaces != 8 and stripped.startswith("return "):
                l = "        " + stripped
        elif ln == 95:
            if spaces == 8 and stripped.startswith("def "):
                l = "    " + stripped
        elif ln == 96:
            if spaces != 8 and stripped:
                l = "        " + stripped
        elif ln == 97:
            if spaces != 8 and stripped.startswith("return "):
                l = "        " + stripped
        elif ln == 99:
            if spaces == 8 and stripped.startswith("def "):
                l = "    " + stripped
        elif ln == 100:
            if spaces != 8 and stripped:
                l = "        " + stripped
        elif ln == 101:
            if spaces != 8 and stripped.startswith("return "):
                l = "        " + stripped

    elif 117 <= ln <= 156:
        # Inside TemplateEdgeGraph.__init__
        if ln == 117:
            # def __init__ line needs 4 spaces
            if spaces == 8 and stripped.startswith("def "):
                l = "    " + stripped
        else:
            # __init__ body needs 8 spaces
            if stripped and not stripped.startswith("#"):
                if spaces == 4:
                    l = "        " + stripped

    else:
        # Lines 157+: TemplateEdgeGraph class methods OR standalone functions
        # Class methods: def at 8, body at 12
        # Standalone functions: def at 4, body at 8
        if stripped and not stripped.startswith("#") and not stripped.startswith('"""'):
            if stripped.startswith("def ") or stripped.startswith("@"):
                # This is a function/method def
                # TemplateEdgeGraph methods: 8 spaces (were at 4)
                # Standalone functions: 4 spaces (were at 8)
                if spaces == 4:
                    l = "        " + stripped
                elif spaces == 8:
                    l = "    " + stripped
            elif stripped.startswith("return ") or stripped.startswith("for ") or \
                 stripped.startswith("if ") or stripped.startswith("while ") or \
                 stripped.startswith("try:") or stripped.startswith("with "):
                # Inner body: TemplateEdgeGraph methods = 12 (were at 8 or 4)
                # Standalone function body = 8 (were at 4 or 8)
                if spaces == 4:
                    l = "            " + stripped
                elif spaces == 8:
                    l = "        " + stripped
            elif stripped.startswith("else:") or stripped.startswith("elif "):
                if spaces == 4:
                    l = "            " + stripped
                elif spaces == 8:
                    l = "        " + stripped
            elif stripped.startswith("break") or stripped.startswith("continue"):
                if spaces == 4:
                    l = "            " + stripped
                elif spaces == 8:
                    l = "        " + stripped
            elif stripped.startswith("import ") or stripped.startswith("from "):
                pass  # These are fine at 0
            elif stripped.startswith("_") and "=" in stripped:
                # Assignment inside function
                if spaces == 4:
                    l = "        " + stripped
            else:
                # Generic body line
                if spaces == 4:
                    l = "        " + stripped

    fixed_lines.append(l)

with open("src/aps_cp_sat/model/constructive_sequence_builder.py", "w", encoding="utf-8") as f:
    f.writelines(fixed_lines)
print("Done")

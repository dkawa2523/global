"""Accept GitHub's protected math syntax in the site and PDF Markdown reader."""

from __future__ import annotations

import re
from typing import Any

from markdown import Markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from typing_extensions import override


class ProtectedMath(Preprocessor):
    """Adapt math fences and dollar-backtick spans without changing code samples."""

    @override
    def run(self, lines: list[str]) -> list[str]:
        output = []
        fence = ""
        math = False
        for line in lines:
            if fence:
                if re.fullmatch(
                    r" {0,3}" + re.escape(fence + fence[0]) + r"*\s*", line
                ):
                    output.append("$$" if math else line)
                    fence = ""
                else:
                    output.append(line)
            elif opening := re.fullmatch(r" {0,3}(`{3,}|~{3,})(.*)", line):
                fence = opening[1]
                math = opening[2].strip() == "math"
                output.append("$$" if math else line)
            else:
                output.append(re.sub(r"\$`([^`\n]+)`\$", r"$\1$", line))
        return output


class GitHubMathExtension(Extension):
    """Normalize GitHub math before the standard fenced-code and math readers."""

    @override
    def extendMarkdown(self, md: Markdown) -> None:
        md.preprocessors.register(ProtectedMath(md), "github_math", 27)


def makeExtension(**kwargs: Any) -> GitHubMathExtension:
    return GitHubMathExtension(**kwargs)

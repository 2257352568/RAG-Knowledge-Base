import mistune
from core.parsing.models import Document


def parse_markdown(filepath: str) -> Document:
    """Parse a Markdown file into a Document using mistune AST tokenization."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    token_list, _state = mistune.Markdown().parse(raw)
    return tokens_to_document(token_list, source_path=filepath, fallback_text=raw)


def tokens_to_document(token_list: list[dict], source_path: str = "",
                       fallback_text: str = "") -> Document:
    """Build a Document from mistune AST tokens.

    This is the shared parsing engine used by both vanilla Markdown
    and Obsidian-enhanced parsing.
    """
    title = ""
    sections: list[dict] = []
    current_section: dict | None = None

    for token in token_list:
        ttype = token["type"]

        if ttype == "blank_line":
            continue

        if ttype == "block_code":
            lang = token.get("attrs", {}).get("info", "")
            code = token["raw"]
            text = f"```{lang}\n{code}```"
            if current_section is None:
                current_section = {"title": "", "text": "", "page": None}
            current_section["text"] += text + "\n"
            continue

        if ttype == "heading":
            level = token["attrs"]["level"]
            heading_text = _extract_text(token)
            if level == 1 and not title:
                title = heading_text
            if current_section:
                current_section["text"] = current_section["text"].strip()
                if current_section["text"]:
                    sections.append(current_section)
            current_section = {"title": heading_text, "text": heading_text + "\n", "page": None}
            continue

        if ttype in ("paragraph", "list", "block_quote", "block_html"):
            text = _extract_text(token)
            if current_section is None:
                current_section = {"title": "", "text": "", "page": None}
            current_section["text"] += text + "\n"
            continue

    if current_section:
        current_section["text"] = current_section["text"].strip()
        if current_section["text"]:
            sections.append(current_section)

    full_text = "\n\n".join(s["text"] for s in sections if s["text"]).strip()

    return Document(
        text=full_text or fallback_text.strip(),
        title=title,
        source_path=source_path,
        pages=0,
        sections=sections,
    )


def extract_inline_text(token: dict) -> str:
    """Recursively extract plain text from a mistune inline token tree.

    Public — used by obsidian_parser for wiki link replacement.
    """
    if "children" in token:
        parts = []
        for child in token["children"]:
            ct = child["type"]
            if ct == "text":
                parts.append(child.get("raw", ""))
            elif ct == "codespan":
                parts.append("`" + child.get("raw", "") + "`")
            elif ct == "softbreak":
                parts.append(" ")
            elif ct == "linebreak":
                parts.append("\n")
            elif ct in ("emphasis", "strong"):
                parts.append(extract_inline_text(child))
            elif ct == "link":
                parts.append(extract_inline_text(child))
            elif ct == "image":
                alt = extract_inline_text(child)
                src = child.get("attrs", {}).get("url", "")
                parts.append(f"[图片: {alt}]" if alt else f"[图片: {src}]")
            else:
                parts.append(child.get("raw", ""))
        return "".join(parts)
    return token.get("raw", "")


# Backward-compatible alias
_extract_text = extract_inline_text

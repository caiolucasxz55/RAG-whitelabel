"""Gerador mínimo de PDF com camada de texto (Helvetica, WinAnsi), sem dependências.

Usado para criar os PDFs de exemplo e fixtures de teste. Não é parte do pacote `kit`.
"""

from __future__ import annotations

from pathlib import Path


def _escape(line: str) -> bytes:
    raw = line.encode("cp1252", errors="replace")
    return raw.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _content_stream(lines: list[str], font_size: int = 11, leading: int = 15) -> bytes:
    out = [b"BT", f"/F1 {font_size} Tf".encode(), f"{leading} TL".encode(), b"50 800 Td"]
    for line in lines:
        out.append(b"(" + _escape(line) + b") Tj T*")
    out.append(b"ET")
    return b"\n".join(out)


def write_pdf(path: Path, pages: list[list[str]]) -> Path:
    """Escreve um PDF com uma página por item de `pages` (cada página = lista de linhas)."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog = add(b"")  # preenchido depois
    pages_obj = add(b"")
    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    kids = []
    for lines in pages:
        stream = _content_stream(lines)
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        kids.append(
            add(
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 595 842] "
                b"/Resources << /Font << /F1 %d 0 R >> >> /Contents %d 0 R >>"
                % (pages_obj, font, content)
            )
        )
    objects[catalog - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_obj
    objects[pages_obj - 1] = b"<< /Type /Pages /Kids [%s] /Count %d >>" % (
        b" ".join(b"%d 0 R" % k for k in kids),
        len(kids),
    )

    data = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(data))
        data += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(data)
    data += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        data += b"%010d 00000 n \n" % offset
    data += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        catalog,
        xref,
    )
    path.write_bytes(bytes(data))
    return path

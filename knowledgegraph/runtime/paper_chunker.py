"""Lossless PDF-page chunks measured with the actual BGE tokenizer."""
from functools import lru_cache
import json
from pathlib import Path
import re

from cognee.modules.chunking.Chunker import Chunker
from cognee.modules.chunking.chunk_id import chunk_content_hash, content_chunk_id
from cognee.modules.chunking.models.DocumentChunk import DocumentChunk


@lru_cache(maxsize=1)
def tokenizer():
    from tokenizers import Tokenizer
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config/settings.json").read_text())
    result = Tokenizer.from_file(str(Path(config["embedding"]["snapshot"]) / "tokenizer.json"))
    result.no_truncation()
    result.no_padding()
    return result


def split_exact(text, maximum):
    """Return contiguous source slices; do not decode tokens or normalize text."""
    remaining = text
    while remaining:
        if len(tokenizer().encode(remaining, add_special_tokens=False).ids) <= maximum:
            yield remaining
            return
        low, high = 1, len(remaining)
        while low < high:
            middle = (low + high + 1) // 2
            count = len(tokenizer().encode(remaining[:middle], add_special_tokens=False).ids)
            if count <= maximum:
                low = middle
            else:
                high = middle - 1
        end = low
        # Prefer a paragraph or word boundary close to the safe endpoint.
        boundary = remaining.rfind("\n\n", max(0, end // 2), end)
        if boundary >= 0:
            end = boundary + 2
        else:
            boundaries = list(re.finditer(r"\s+", remaining[max(0, end // 2):end]))
            if boundaries:
                end = end // 2 + boundaries[-1].end()
        # Token counts can change at a partial-word boundary; verify the final
        # slice explicitly instead of assuming monotonic subword tokenization.
        while end > 0 and len(tokenizer().encode(remaining[:end], add_special_tokens=False).ids) > maximum:
            end -= 1
        if end == 0:
            raise ValueError("No source slice fits the configured token budget")
        yield remaining[:end]
        remaining = remaining[end:]


class PaperPageChunker(Chunker):
    chunker_id = "paper_pdf_page_exact_bge_v1"

    async def read(self):
        occurrences = {}
        async for source in self.get_text():
            markers = list(re.finditer(r"(?m)^## PDF page (\d+) of (\d+)\n", source))
            starts = [0] + [marker.start() for marker in markers]
            starts = sorted(set(starts)) + [len(source)]
            for start, end in zip(starts, starts[1:]):
                section = source[start:end]
                marker = re.match(r"## PDF page (\d+) of (\d+)\n", section)
                page = int(marker.group(1)) if marker else None
                for text in split_exact(section, self.max_chunk_size):
                    tokens = len(tokenizer().encode(text, add_special_tokens=False).ids)
                    content_hash = chunk_content_hash(text)
                    occurrence = occurrences.get(content_hash, 0)
                    occurrences[content_hash] = occurrence + 1
                    yield DocumentChunk(
                        id=content_chunk_id(str(self.document.id), content_hash, occurrence),
                        chunker_id=self.chunker_id, text=text, chunk_size=tokens,
                        max_chunk_tokens=self.max_chunk_size, content_hash=content_hash,
                        is_part_of=self.document, contains=[], chunk_index=self.chunk_index,
                        cut_type="pdf_page", importance_weight=self.document.importance_weight,
                        document_id=str(self.document.id), document_name=self.document.name,
                        metadata={"index_fields": ["text"], "pdf_page": page},
                    )
                    self.chunk_index += 1

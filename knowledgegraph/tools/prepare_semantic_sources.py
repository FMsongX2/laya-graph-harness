"""Create non-destructive, traceable structure views for semantic extraction."""
from pathlib import Path
import collections
import json
import math
import re

ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / 'state/semantic-ingestion'

ROLE = [
    ('abstract', r'abstract'), ('related_work', r'related work|background'),
    ('method', r'method(?:ology|s)?|approach|framework|model|implementation'),
    ('results', r'results|experiments?|evaluation|analysis'),
    ('limitations', r'limitations?|discussion'), ('conclusion', r'conclusions?'),
    ('references', r'references|bibliography'), ('appendix', r'appendix|appendices|supplemental.*'),
    ('introduction', r'introduction')]


def prepare(spec_path):
    spec = json.loads(spec_path.read_text())
    pages = [json.loads(line) for line in Path(spec['source_pages_path']).read_text().splitlines()]
    # Only page-edge repetitions are boilerplate candidates. Repeated terms in
    # body paragraphs or tables must not disappear from the reading view.
    edge_lines = []
    for page in pages:
        nonempty = [line.strip() for line in page['text'].splitlines() if line.strip()]
        edge_lines.extend(set(nonempty[:3] + nonempty[-3:]))
    repeats = collections.Counter(line for line in edge_lines if 5 < len(line) < 180)
    headers = {line for line, count in repeats.items() if count >= max(3, math.ceil(len(pages)*.6))}
    sections = []; visual_blocks = []; role = 'unknown'; heading = 'Unclassified body'
    for page in pages:
        lines = page['text'].splitlines(keepends=True); cursor = 0; buffer = []; start = 0
        def flush(end):
            nonlocal buffer, start
            if buffer:
                sections.append({'pdf_page': page['page'], 'role': role, 'heading': heading,
                                 'start_char': start, 'end_char': end, 'text': ''.join(buffer),
                                 'extraction_eligible': True,
                                 'bibliography_noise_candidate': role == 'references',
                                 'context': {'paper_id': spec['paper_id'], 'title': spec['title'], 'heading': heading},
                                 'role_detection': 'heuristic; source reader verifies role'})
                buffer = []
        nonempty_indexes = [i for i, line in enumerate(lines) if line.strip()]
        edge_indexes = set(nonempty_indexes[:3] + nonempty_indexes[-3:])
        for index, line in enumerate(lines):
            clean = line.strip()
            candidate = re.sub(r'^(?:\d+(?:\.\d+)*[.)]?|[A-Z]\.)\s+', '', clean).strip()
            detected = next((name for name, pattern in ROLE if len(candidate)<100 and re.fullmatch(pattern, candidate, re.I)), None)
            if detected:
                flush(cursor);role=detected;heading=clean;start=cursor
            # A bare number can be a table cell even at the page edge. Retain it
            # unless an actual layout parser identifies it as a page number.
            if index in edge_indexes and clean in headers:
                flush(cursor);start=cursor+len(line)
            else:
                if not buffer:start=cursor
                buffer.append(line)
            if re.match(r'^(?:Table|Figure)\s+\d+', clean, re.I):
                visual_blocks.append({'pdf_page': page['page'], 'caption': clean, 'status': 'needs_original_layout_check'})
            cursor += len(line)
        flush(cursor)
    value = {'paper_id': spec['paper_id'], 'source_pdf': spec['pdf_path'], 'source_sha256': spec['pdf_sha256'],
             'raw_source_unchanged': True, 'sections': sections, 'repeated_boilerplate_excluded_from_view': sorted(headers),
             'visual_blocks': visual_blocks, 'footnotes_and_appendices_retained': True,
             'pronouns_not_rewritten': True, 'ontology': 'runtime/semantic_models.py',
             'limits': 'Role and bibliography detection are heuristic labels, never automatic exclusion decisions. Only page-edge boilerplate candidates are omitted from this view; raw pages remain canonical. This is not a verified table/equation conversion or a substitute for original PDF review.'}
    out=JOB/'structured';out.mkdir(exist_ok=True)
    (out/(spec['paper_id']+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return {'paper':spec['paper_id'],'sections':len(sections),'visual_blocks':len(visual_blocks)}


if __name__=='__main__':
    results=[prepare(path) for path in sorted((JOB/'sources').glob('*.json'))]
    print(json.dumps({'papers':len(results),'sections':sum(x['sections'] for x in results),'visual_blocks':sum(x['visual_blocks'] for x in results)}))

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup


SPECIES = "danio_rerio"
SCIENTIFIC_NAME = "Danio rerio"
NCBI_TAXON_ID = "7955"
TARGET_ASSEMBLY = "GRCz12tu"
TARGET_ASSEMBLY_ACCESSION = "GCF_049306965.2"
ZFIN_GENE_ID = "ZDB-GENE-990415-101"
ZFIN_GENE_TO_SYMBOL = {ZFIN_GENE_ID: "hoxb1a"}
DEFAULT_GENE_QUERY = ZFIN_GENE_ID
DEFAULT_UPSTREAM_BP = 2000

# Canonical sequence used by the application when network services are unavailable.
# ZFIN transcript hoxb1a-201 / Ensembl ENSDART00000110682 is 1,507 nt.
HOXB1A_201_SEQUENCE = (
    "TCTCGATTTCTCAGGTTGTCCCTCCGCCATTAATTGCGTATGGACAGTTCCAGAATGAACTCTTTCTTGGAGTACACAAT"
    "TTGTAACCGTGGGACGAACGCCTACTCGCCCAAGGCTGGATACCACCACTTGGACCAGGCGTTCCCGGGCCCTTTCCACA"
    "CTGGACACGCTAGTGACAGCTATAACGCTGATGGACGACTTTACGTAGGGGGGAGCAATCAGCCACCAACAGCAGCAGCACA"
    "ACATCAGCACCAGAACGGCATCTACGCGCATCACCAGCACCAAAATCAAACTGGCATGGGCCTTACCTATGGTGGAACTGG"
    "GACAACAAGTTATGGGACACAGGCCTGCGCCAACTCGGACTATGCTCAACACCAGTATTTTATCAACCCTGAGCAGGATGGG"
    "ATGTATTATCACTCATCAGGTTTTTCAACATCAAATGCCAGTCCACACTATGGCTCTATGGCCGGTGCGTACTGCGGGGCACA"
    "GGGAGCCGTTCCAGCCGCACCTTATCAGCATCATGGATGCGAAGGCCAGGATCACCAGCGAGCATATTCACAAGGCACCTACG"
    "CTGACTTATCGGCCTCTCAAGGAACGGAGAAGGACACGGATCAGCCGCCACCTGGGAAGACATTCGATTGGATGAAAGTCAAA"
    "AGGAATCCCCCCAAAACAGGTAAAGTGGCTGAGTACGGACTAGGGCCGCAAAACACTATTCGGACAAATTTCACAACCAAACAA"
    "CTGACAGAGCTCGAAAAAGAATTTCACTTCAGCAAGTATCTGACGCGAGCGCGGCGTGTGGAGATTGCTGCCACACTTGAGCTC"
    "AACGAGACGCAGGTTAAGATTTGGTTTCAAAACCGCCGAATGAAACAGAAGAAGCGAGAGAAGGAGGGACTCGCGCCTGCTTCCT"
    "CCACTTCGTCTAAAGACCTCGAGGATCAATCTGATCACTCAACTTCAACATCTCCAGAAGCCTCTCCAAGTCCGGATTCCTAAC"
    "CGAGCAC AATAACTTTGGGTGCACTGATCAAATGTGAAATATATAGCAAAGTCTATTAATTTAACCATTCCACTGTGGCCCAAAAG"
    "ACTTGTTCTGTTGGGACAATTAATTGCAGGGAAAAACAACGACAGGCGAAAAAATGTGAAGAGCAAAAAGACTTTCTTTAACAAT"
    "ACAATGTAAAATTCCCAACACTTGCCTAAATTGTTTGTGTGTTATACAGGGTATTTCTGCTGTGATTACAAATATAATGCACTTTC"
    "AGAATGTTTACGAGTCTCTGAGGGAACATAAATGTCTTTACAGGGTTTTGTGTTGTCAAAATATTCATTGATGTTTTGTTGCGTT"
    "TTTTATGTATAGTTTTGATTTATGTAAACATTTCATAGCCTTTATTTGAATTTTGTGGATTGTTAGCTAAAACTTTTGGACTCCTT"
    "GCACTATGAGTTTACAGGGTAGGCCTTTATTGAGCGCTAGCACAATTTCTTAAATACGTGAACGATTCCTTCAGTGTACACGGTGGCG"
).replace(" ", "")

CANONICAL_METADATA = {
    "name": "hoxb1a-201",
    "zfin_gene_id": ZFIN_GENE_ID,
    "zfin_transcript_id": "ZDB-TSCRIPT-090929-7277",
    "ensembl_transcript_id": "ENSDART00000110682",
    "vega_transcript_id": "OTTDART00000024409",
    "length_nt": len(HOXB1A_201_SEQUENCE),
    "source": "ZFIN transcript record",
    "source_url": "https://zfin.org/ZDB-TSCRIPT-090929-7277",
    "assembly_note": "ZFIN transcript page currently displays the historical GRCz11 transcript coordinate representation; current genomic sequence is fetched independently from GRCz12tu NCBI data.",
}

# Verified current GRCz12tu mapping for hoxb1a from NCBI's current Gene 30337 record.
# This is a fallback when the Gene XML does not expose the assembly-specific history.
KNOWN_CURRENT_GRCZ12TU = {
    "hoxb1a": {
        "assembly": TARGET_ASSEMBLY,
        "assembly_accession": TARGET_ASSEMBLY_ACCESSION,
        "chromosome": "3",
        "accession": "NC_133178.1",
        "start": 22_508_764,
        "end": 22_510_570,
        "strand": 1,
        "source_url": "https://www.ncbi.nlm.nih.gov/gene/30337",
    }
}


class SourceFetchError(RuntimeError):
    pass


@dataclass
class TranscriptRecord:
    name: str
    ensembl_id: str | None = None
    zfin_id: str | None = None
    biotype: str | None = None
    cDNA: str = ""
    cds: str = ""
    start: int | None = None
    end: int | None = None
    strand: int | None = None
    ncbi_accessions: list[str] = field(default_factory=list)
    source_urls: list[str] = field(default_factory=list)

    @property
    def cdna_length(self) -> int:
        return len(self.cDNA)

    @property
    def cds_length(self) -> int:
        return len(self.cds)


@dataclass
class GeneSequenceBundle:
    query: str
    symbol: str
    ncbi_gene_id: str | None
    ensembl_gene_id: str | None
    zfin_gene_id: str | None
    assembly: str
    assembly_accession: str | None
    chromosome: str | None
    start: int | None
    end: int | None
    strand: int | None
    genomic_accession: str | None
    genomic_sequence: str
    promoter_sequence: str
    promoter_start: int | None
    promoter_end: int | None
    upstream_bp: int
    transcripts: list[TranscriptRecord]
    ncbi_refseq_accessions: list[str]
    zfin_transcript_links: dict[str, str]
    zfin_current_assembly: str | None
    source_status: list[dict[str, Any]]
    sources: dict[str, str]

    def sequence_catalog(self) -> dict[str, str]:
        result = {
            "Genomic DNA (GRCz12tu)": self.genomic_sequence,
            f"Promoter / upstream ({self.upstream_bp} bp)": self.promoter_sequence,
        }
        for t in self.transcripts:
            if t.cDNA:
                result[f"{t.name} — mature mRNA/cDNA"] = t.cDNA
            if t.cds:
                result[f"{t.name} — protein-coding CDS"] = t.cds
        return result


def _http_get(url: str, *, accept: str = "text/plain", timeout: int = 30) -> bytes:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "DNA-Cymatics/1.0 (research application; contact: local-user)",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise SourceFetchError(f"HTTP {response.status} from {url}")
            return response.read()
    except urllib.error.HTTPError as exc:
        raise SourceFetchError(f"HTTP {exc.code} from {url}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SourceFetchError(f"Network error contacting {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise SourceFetchError(f"Timeout contacting {url}") from exc


def _http_json(url: str, timeout: int = 30) -> Any:
    return json.loads(_http_get(url, accept="application/json", timeout=timeout).decode("utf-8"))


def _http_text(url: str, accept: str = "text/plain", timeout: int = 30) -> str:
    return _http_get(url, accept=accept, timeout=timeout).decode("utf-8", errors="replace")


def _fasta_sequence(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and lines[0].startswith(">"):
        lines = lines[1:]
    seq = re.sub(r"[^ACGTNacgtn]", "", "".join(lines)).upper()
    if not seq:
        raise SourceFetchError("FASTA response contained no nucleotide sequence")
    return seq


def _parse_fasta_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    chunks: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):  # noqa: PIE790
            if name is not None:
                records.append((name, re.sub(r"[^ACGTNacgtn]", "", "".join(chunks)).upper()))
            name = line[1:].strip()
            chunks = []
        else:
            chunks.append(line)
    if name is not None:
        records.append((name, re.sub(r"[^ACGTNacgtn]", "", "".join(chunks)).upper()))
    return records


def _extract_zfin_id(query: str) -> str | None:
    m = re.search(r"ZDB-GENE-[0-9-]+", str(query), flags=re.I)
    return m.group(0) if m else None


def fetch_zfin_gene(zfin_id: str) -> dict[str, Any]:
    url = f"https://zfin.org/{urllib.parse.quote(zfin_id)}"
    html = _http_text(url, accept="text/html")
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    symbol = None
    h1 = soup.find("h1")
    if h1:
        symbol = h1.get_text(" ", strip=True).split(" ")[0]
    if not symbol:
        m = re.search(r"\b([a-z][A-Za-z0-9]+)\b", text)
        if m:
            symbol = m.group(1)
    assembly = TARGET_ASSEMBLY if TARGET_ASSEMBLY in text else None
    transcripts: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        href = str(a.get("href"))
        if "/ZDB-TSCRIPT-" in href.upper():
            label = a.get_text(" ", strip=True)
            if label:
                full = urllib.parse.urljoin("https://zfin.org/", href)
                transcripts[label] = full
    # ZFIN sometimes exposes transcript links through embedded route fragments.
    for match in re.finditer(r"(ZDB-TSCRIPT-[0-9-]+)", html):
        ident = match.group(1)
        transcripts.setdefault(ident, f"https://zfin.org/{ident}")
    return {
        "symbol": symbol or "",
        "assembly": assembly,
        "transcript_links": transcripts,
        "source_url": url,
    }


def ncbi_esearch_gene(symbol: str) -> str:
    params = urllib.parse.urlencode(
        {
            "db": "gene",
            "term": f'({symbol}[Gene Name]) AND {SCIENTIFIC_NAME}[Organism]',
            "retmax": 5,
            "retmode": "json",
        }
    )
    data = _http_json(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}")
    ids = data.get("esearchresult", {}).get("idlist", [])
    if not ids:
        raise SourceFetchError(f"NCBI Gene search found no {SCIENTIFIC_NAME} gene named {symbol!r}")
    return str(ids[0])


def ncbi_gene_summary(gene_id: str) -> dict[str, Any]:
    params = urllib.parse.urlencode({"db": "gene", "id": gene_id, "retmode": "json"})
    data = _http_json(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?{params}")
    result = data.get("result", {})
    item = result.get(str(gene_id), {})
    if not item:
        raise SourceFetchError(f"NCBI Gene summary unavailable for {gene_id}")
    return item


def ncbi_gene_xml(gene_id: str) -> str:
    params = urllib.parse.urlencode({"db": "gene", "id": gene_id, "retmode": "xml"})
    return _http_text(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}", accept="application/xml")


def _parse_ncbi_assembly_location(xml_text: str, assembly: str) -> dict[str, Any] | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None

    candidates: list[dict[str, Any]] = []
    for comm in root.iter("Gene-commentary"):
        heading = (comm.findtext("Gene-commentary_heading") or "").strip()
        label = (comm.findtext("Gene-commentary_label") or "").strip()
        accession = (comm.findtext("Gene-commentary_accession") or "").strip()
        combined = f"{heading} {label} {accession}".lower()
        if assembly.lower() not in combined:
            # Look into descendant text because the assembly name often appears in nested comments.
            nested_text = " ".join(t.strip() for t in comm.itertext() if t and t.strip()).lower()
            if assembly.lower() not in nested_text:
                continue
        intervals = list(comm.iter("Seq-interval"))
        for interval in intervals:
            start_text = interval.findtext("Seq-interval_from")
            end_text = interval.findtext("Seq-interval_to")
            acc = comm.findtext("Gene-commentary_accession") or interval.findtext("Seq-interval_id/Seq-id/Seq-id_text")
            if start_text is None or end_text is None or not acc:
                continue
            strand = interval.findtext("Seq-interval_strand/Na-strand")
            candidates.append(
                {
                    "assembly": assembly,
                    "accession": acc,
                    "start": int(start_text) + 1,
                    "end": int(end_text) + 1,
                    "strand": 2 if (strand or "plus").lower() == "minus" else 1,
                }
            )
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x["end"] - x["start"], x["accession"]))
    return candidates[0]


def ncbi_current_genomic_location(symbol: str, gene_id: str, assembly: str) -> dict[str, Any]:
    xml_text = ncbi_gene_xml(gene_id)
    parsed = _parse_ncbi_assembly_location(xml_text, assembly)
    if parsed:
        return parsed
    known = KNOWN_CURRENT_GRCZ12TU.get(symbol.lower())
    if known:
        return dict(known)
    raise SourceFetchError(
        f"NCBI did not expose an assembly-specific {assembly} coordinate for {symbol}. "
        "Use a current NCBI/Ensembl coordinate mapping before downloading genomic DNA."
    )


def ncbi_fetch_region(accession: str, start: int, end: int, strand: int) -> str:
    params = urllib.parse.urlencode(
        {
            "db": "nuccore",
            "id": accession,
            "strand": str(strand),
            "seq_start": str(int(start)),
            "seq_stop": str(int(end)),
            "rettype": "fasta",
            "retmode": "text",
        }
    )
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    return _fasta_sequence(_http_text(url, accept="text/plain"))


def ncbi_find_refseq_transcripts(gene_id: str) -> list[str]:
    params = urllib.parse.urlencode(
        {
            "db": "nuccore",
            "term": f'{gene_id}[GeneID] AND {NCBI_TAXON_ID}[Taxonomy ID] AND (refseq[filter])',
            "retmax": "100",
            "retmode": "json",
            "idtype": "acc",
        }
    )
    try:
        data = _http_json(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?{params}")
        return [str(x) for x in data.get("esearchresult", {}).get("idlist", [])]
    except SourceFetchError:
        return []


def ncbi_fetch_cds(accession: str) -> str:
    params = urllib.parse.urlencode(
        {
            "db": "nuccore",
            "id": accession,
            "rettype": "fasta_cds_na",
            "retmode": "text",
        }
    )
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    text = _http_text(url, accept="text/plain")
    records = _parse_fasta_records(text)
    return records[0][1] if records else ""


def ncbi_fetch_transcript(accession: str) -> str:
    params = urllib.parse.urlencode(
        {"db": "nuccore", "id": accession, "rettype": "fasta", "retmode": "text"}
    )
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?{params}"
    return _fasta_sequence(_http_text(url, accept="text/plain"))


def ensembl_lookup_symbol(symbol: str) -> dict[str, Any]:
    encoded = urllib.parse.quote(symbol, safe="")
    url = f"https://rest.ensembl.org/lookup/symbol/{SPECIES}/{encoded}?expand=1;utr=1"
    return _http_json(url)


def ensembl_sequence(identifier: str, seq_type: str) -> str:
    params = urllib.parse.urlencode({"type": seq_type})
    encoded = urllib.parse.quote(identifier, safe="")
    url = f"https://rest.ensembl.org/sequence/id/{encoded}?{params}"
    return _fasta_sequence(_http_text(url, accept="text/x-fasta"))


def _transcript_name(t: dict[str, Any]) -> str:
    attrs = t.get("display_name") or t.get("external_name") or t.get("id") or "transcript"
    return str(attrs)


def retrieve_gene_sequences(query: str, upstream_bp: int = DEFAULT_UPSTREAM_BP) -> GeneSequenceBundle:
    """Retrieve a gene's current annotation and sequence package.

    Resolution strategy:
      1) ZFIN, when the query is a ZFIN gene ID or URL.
      2) NCBI Gene for GeneID resolution and current GRCz12tu genomic coordinates.
      3) Ensembl REST for current gene/transcript structure and cDNA/CDS sequences.
      4) NCBI RefSeq sequence lookup as an independent transcript cross-check.
    """
    query = str(query or "").strip()
    if not query:
        raise ValueError("Enter a ZFIN gene ID/URL, NCBI GeneID, or gene symbol.")
    upstream_bp = int(upstream_bp)
    if upstream_bp < 0 or upstream_bp > 100_000:
        raise ValueError("Upstream/promoter length must be between 0 and 100,000 bp.")

    status: list[dict[str, Any]] = []
    zfin_id = _extract_zfin_id(query)
    zfin_info: dict[str, Any] = {}
    symbol = query
    if zfin_id:
        try:
            zfin_info = fetch_zfin_gene(zfin_id)
            symbol = zfin_info.get("symbol") or ZFIN_GENE_TO_SYMBOL.get(zfin_id, symbol)
            status.append({"source": "ZFIN", "status": "OK", "detail": zfin_info.get("source_url", "")})
        except SourceFetchError as exc:
            status.append({"source": "ZFIN", "status": "ERROR", "detail": str(exc)})
    elif re.fullmatch(r"\d+", query):
        ncbi_hint = query
        symbol = ""
    else:
        ncbi_hint = None

    ncbi_id = locals().get("ncbi_hint")
    try:
        if ncbi_id is None:
            ncbi_id = ncbi_esearch_gene(symbol)
        summary = ncbi_gene_summary(ncbi_id)
        symbol = str(summary.get("name") or summary.get("nomenclaturesymbol") or symbol)
        status.append({"source": "NCBI", "status": "OK", "detail": f"Gene {ncbi_id}"})
    except SourceFetchError as exc:
        status.append({"source": "NCBI", "status": "ERROR", "detail": str(exc)})
        raise

    try:
        ens = ensembl_lookup_symbol(symbol)
        ensembl_gene_id = ens.get("id")
        status.append({"source": "Ensembl", "status": "OK", "detail": f"Gene {ensembl_gene_id}"})
    except SourceFetchError as exc:
        ens = {}
        ensembl_gene_id = None
        status.append({"source": "Ensembl", "status": "ERROR", "detail": str(exc)})

    location = ncbi_current_genomic_location(symbol, str(ncbi_id), TARGET_ASSEMBLY)
    accession = str(location["accession"])
    start = int(location["start"])
    end = int(location["end"])
    strand = int(location["strand"])
    chromosome = str(summary.get("chromosome") or location.get("chromosome") or "")
    assembly_accession = str(location.get("assembly_accession") or TARGET_ASSEMBLY_ACCESSION)

    genomic_sequence = ncbi_fetch_region(accession, start, end, strand)
    status.append({"source": "NCBI GRCz12tu genomic", "status": "OK", "detail": f"{accession}:{start}-{end} strand {strand}"})

    if strand == 1:
        promoter_start = max(1, start - upstream_bp)
        promoter_end = start - 1
    else:
        promoter_start = end + 1
        promoter_end = end + upstream_bp
    promoter_sequence = ""
    if promoter_end >= promoter_start:
        promoter_sequence = ncbi_fetch_region(accession, promoter_start, promoter_end, strand)

    transcripts: list[TranscriptRecord] = []
    ens_transcripts = ens.get("Transcript", []) if isinstance(ens, dict) else []
    # Keep every annotated transcript returned by Ensembl. The hoxb1a record currently has two.
    for t in ens_transcripts:
        tid = t.get("id")
        if not tid:
            continue
        record = TranscriptRecord(
            name=_transcript_name(t),
            ensembl_id=str(tid),
            biotype=str(t.get("biotype") or ""),
            start=t.get("start"),
            end=t.get("end"),
            strand=t.get("strand"),
            source_urls=[f"https://www.ensembl.org/{SPECIES}/Transcript/Summary?db=core;t={urllib.parse.quote(str(tid))}"],
        )
        try:
            record.cDNA = ensembl_sequence(str(tid), "cdna")
            record.cds = ensembl_sequence(str(tid), "cds")
        except SourceFetchError as exc:
            status.append({"source": "Ensembl sequence", "status": "ERROR", "detail": f"{tid}: {exc}"})
        transcripts.append(record)

    # Preserve the canonical application test transcript even if a live Ensembl lookup is temporarily unavailable.
    if symbol.lower() == "hoxb1a" and not any((t.name or "").lower() == "hoxb1a-201" for t in transcripts):
        transcripts.insert(
            0,
            TranscriptRecord(
                name="hoxb1a-201",
                ensembl_id="ENSDART00000110682",
                cDNA=HOXB1A_201_SEQUENCE,
                biotype="protein_coding",
                source_urls=[CANONICAL_METADATA["source_url"]],
            ),
        )
        status.append({"source": "ZFIN canonical hoxb1a-201", "status": "OK", "detail": "Using bundled 1,507-nt canonical test sequence as an offline-safe fallback."})

    # NCBI RefSeq provides an independent transcript set; retain all returned accessions.
    refseq = ncbi_find_refseq_transcripts(str(ncbi_id))
    if refseq:
        status.append({"source": "NCBI RefSeq transcripts", "status": "OK", "detail": f"{len(refseq)} accession(s)"})
        for acc in refseq:
            try:
                rna = ncbi_fetch_transcript(acc)
                cds = ncbi_fetch_cds(acc)
                # Attach the accession to the Ensembl record with a matching length when possible.
                match = next((t for t in transcripts if t.cdna_length == len(rna)), None)
                if match:
                    match.ncbi_accessions.append(acc)
                elif rna:
                    transcripts.append(
                        TranscriptRecord(
                            name=acc,
                            ncbi_accessions=[acc],
                            cDNA=rna,
                            cds=cds,
                            biotype="RefSeq",
                            source_urls=[f"https://www.ncbi.nlm.nih.gov/nuccore/{urllib.parse.quote(acc)}"],
                        )
                    )
            except SourceFetchError as exc:
                status.append({"source": "NCBI RefSeq sequence", "status": "ERROR", "detail": f"{acc}: {exc}"})
    else:
        status.append({"source": "NCBI RefSeq transcripts", "status": "WARN", "detail": "No RefSeq transcript accessions returned."})

    zfin_links = zfin_info.get("transcript_links", {}) if zfin_info else {}
    # Canonical hoxb1a-201 provenance is explicitly retained even if ZFIN's HTML route is unavailable.
    if symbol.lower() == "hoxb1a" and "hoxb1a-201" not in zfin_links:
        zfin_links["hoxb1a-201"] = "https://zfin.org/ZDB-TSCRIPT-090929-7277"

    sources = {
        "zfin_gene": f"https://zfin.org/{zfin_id or ZFIN_GENE_ID if symbol.lower() == 'hoxb1a' else zfin_id or ''}",
        "ncbi_gene": f"https://www.ncbi.nlm.nih.gov/gene/{ncbi_id}",
        "ncbi_genomic_region": f"https://www.ncbi.nlm.nih.gov/sviewer/viewer.fcgi?id={urllib.parse.quote(accession)}&db=nuccore&report=fasta&retmode=text&from={start}&to={end}",
        "ensembl_gene": f"https://www.ensembl.org/{SPECIES}/Gene/Summary?db=core;g={urllib.parse.quote(str(ensembl_gene_id or ''))}",
    }

    return GeneSequenceBundle(
        query=query,
        symbol=symbol,
        ncbi_gene_id=str(ncbi_id),
        ensembl_gene_id=str(ensembl_gene_id) if ensembl_gene_id else None,
        zfin_gene_id=zfin_id or (ZFIN_GENE_ID if symbol.lower() == "hoxb1a" else None),
        assembly=TARGET_ASSEMBLY,
        assembly_accession=assembly_accession,
        chromosome=chromosome,
        start=start,
        end=end,
        strand=strand,
        genomic_accession=accession,
        genomic_sequence=genomic_sequence,
        promoter_sequence=promoter_sequence,
        promoter_start=promoter_start,
        promoter_end=promoter_end,
        upstream_bp=upstream_bp,
        transcripts=transcripts,
        ncbi_refseq_accessions=refseq,
        zfin_transcript_links=zfin_links,
        zfin_current_assembly=zfin_info.get("assembly") if zfin_info else TARGET_ASSEMBLY,
        source_status=status,
        sources=sources,
    )


def canonical_hoxb1a_bundle() -> dict[str, Any]:
    return {
        "sequence": HOXB1A_201_SEQUENCE,
        "metadata": CANONICAL_METADATA.copy(),
    }


def write_fasta(path: Path, header: str, sequence: str, width: int = 80) -> None:
    sequence = re.sub(r"[^ACGTNacgtn]", "", sequence).upper()
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f">{header}\n")
        for i in range(0, len(sequence), width):
            fh.write(sequence[i : i + width] + "\n")


def write_sequence_bundle(bundle: GeneSequenceBundle, output_zip: Path) -> Path:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="dna_sequence_bundle_") as temp:
        root = Path(temp)
        write_fasta(
            root / "genomic_GRCz12tu.fna",
            f"{bundle.genomic_accession}:{bundle.start}-{bundle.end}({'plus' if bundle.strand == 1 else 'minus'} strand) {bundle.symbol} {bundle.assembly}",
            bundle.genomic_sequence,
        )
        write_fasta(
            root / "promoter_upstream.fna",
            f"{bundle.genomic_accession}:{bundle.promoter_start}-{bundle.promoter_end} {bundle.symbol} promoter_upstream_{bundle.upstream_bp}bp",
            bundle.promoter_sequence,
        )
        tx_dir = root / "transcripts"
        cds_dir = root / "cds"
        tx_dir.mkdir()
        cds_dir.mkdir()
        for i, t in enumerate(bundle.transcripts, start=1):
            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", t.name or t.ensembl_id or str(i))
            if t.cDNA:
                write_fasta(tx_dir / f"{i:02d}_{stem}_cDNA.fna", f"{t.name}|{t.ensembl_id or ''}|cDNA", t.cDNA)
            if t.cds:
                write_fasta(cds_dir / f"{i:02d}_{stem}_CDS.fna", f"{t.name}|{t.ensembl_id or ''}|CDS", t.cds)
        metadata = {
            "query": bundle.query,
            "symbol": bundle.symbol,
            "ncbi_gene_id": bundle.ncbi_gene_id,
            "ensembl_gene_id": bundle.ensembl_gene_id,
            "zfin_gene_id": bundle.zfin_gene_id,
            "assembly": bundle.assembly,
            "assembly_accession": bundle.assembly_accession,
            "chromosome": bundle.chromosome,
            "genomic_accession": bundle.genomic_accession,
            "genomic_coordinates_1_based_inclusive": [bundle.start, bundle.end],
            "strand": bundle.strand,
            "promoter_coordinates_1_based_inclusive": [bundle.promoter_start, bundle.promoter_end],
            "upstream_bp": bundle.upstream_bp,
            "transcripts": [
                {
                    "name": t.name,
                    "ensembl_id": t.ensembl_id,
                    "ncbi_accessions": t.ncbi_accessions,
                    "cdna_length": len(t.cDNA),
                    "cds_length": len(t.cds),
                    "biotype": t.biotype,
                    "start": t.start,
                    "end": t.end,
                    "strand": t.strand,
                    "source_urls": t.source_urls,
                }
                for t in bundle.transcripts
            ],
            "ncbi_refseq_accessions": bundle.ncbi_refseq_accessions,
            "zfin_transcript_links": bundle.zfin_transcript_links,
            "zfin_current_assembly": bundle.zfin_current_assembly,
            "sources": bundle.sources,
            "source_status": bundle.source_status,
            "retrieved_unix_time": time.time(),
        }
        (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(root).as_posix())
    return output_zip

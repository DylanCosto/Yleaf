"""Exercise both extraction paths with real indexed BAM/CRAM and samtools.

Run with pytest; pysam is needed to construct the small alignment fixtures.
"""
from argparse import Namespace
import importlib
import json
import shutil
import random

import pandas as pd
import pytest

pysam = pytest.importorskip("pysam")
yleaf = importlib.import_module("yleaf.Yleaf")
pytestmark = pytest.mark.skipif(shutil.which("samtools") is None, reason="samtools required")


def make_alignment(tmp_path, y_contig, file_type):
    ref = tmp_path / "reference.fa"
    rng = random.Random(7)
    reference_seq = list("".join(rng.choices("ACGT", k=4000)))
    for pos in [100, 500, 900, 1300]:
        reference_seq[pos - 1] = "A"
    reference_seq = "".join(reference_seq)
    ref.write_text(f">chr1\n{reference_seq}\n>{y_contig}\n{reference_seq}\n")
    pysam.faidx(str(ref))
    header = {"HD": {"VN": "1.6", "SO": "coordinate"},
              "SQ": [{"SN": "chr1", "LN": 4000}, {"SN": y_contig, "LN": 4000}]}
    bam = tmp_path / "sample.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=header) as out:
        # A non-Y marker at the same coordinate must never leak into the calls.
        for chrom, pos, bases in [(0, 100, "G" * 12), (1, 100, "T" * 12),
                                  (1, 500, "C" * 2), (1, 900, "AAAGGG"),
                                  (1, 1300, "G" * 6)]:
            for i, base in enumerate(bases):
                read = pysam.AlignedSegment(out.header)
                read.query_name = f"read-{chrom}-{pos}-{i}"
                read.reference_id = chrom
                read.reference_start = pos - 21
                read.query_sequence = reference_seq[pos - 21:pos - 1] + base + reference_seq[pos:pos + 29]
                read.query_qualities = pysam.qualitystring_to_array("I" * 50)
                read.mapping_quality = 60
                read.flag = 16 if i % 2 else 0
                read.cigarstring = "50M"
                out.write(read)
            # These must be filtered by the unchanged mapping/base quality thresholds.
            for i in range(2):
                read.query_name = f"filtered-{chrom}-{pos}-{i}"
                read.mapping_quality = 0 if i == 0 else 60
                read.query_qualities = pysam.qualitystring_to_array("!" * 50)
                out.write(read)
    pysam.index(str(bam))
    if file_type == "bam":
        return bam, ref
    cram = tmp_path / "sample.cram"
    with pysam.AlignmentFile(str(bam), "rb") as source:
        with pysam.AlignmentFile(str(cram), "wc", header=source.header,
                                 reference_filename=str(ref)) as out:
            for read in source:
                out.write(read)
    pysam.index(str(cram))
    return cram, ref


@pytest.mark.parametrize("y_contig", ["Y", "chrY"])
@pytest.mark.parametrize("file_type", ["bam", "cram"])
def test_indexed_extraction_matches_sequential(tmp_path, monkeypatch, y_contig, file_type):
    alignment, reference = make_alignment(tmp_path, y_contig, file_type)
    marker = tmp_path / "markers.txt"
    marker.write_text("chry\tm1\tA00\t100\tA->T\tA\tT\n"
                      "chry\tm2\tA00\t500\tA->C\tA\tC\n"
                      "chry\tm3\tA00\t900\tA->G\tA\tG\n"
                      "chry\tm4\tA00\t1300\tA->T\tA\tT\n")
    monkeypatch.setattr(yleaf, "get_position_file", lambda *args: marker)
    args = Namespace(threads=1, cram_reference=reference, reference_genome="hg38",
                     ancient_DNA=False, use_old=False, quality_thresh=20,
                     reads_treshold=3, base_majority=90)
    execute, command = yleaf.execute_mpileup, yleaf.call_command
    calls = []
    def run_command(cmd, **kwargs):
        calls.append(cmd)
        return command(cmd, **kwargs)
    monkeypatch.setattr(yleaf, "call_command", run_command)
    def run(root):
        folder = root / "sample"
        folder.mkdir(parents=True)
        info = yleaf.samtools(folder, alignment, file_type == "bam", args)
        (folder / "sample.info").write_text("\n".join(info))
        return {p.name: p.read_bytes() for p in folder.iterdir() if p.is_file()}
    indexed = run(tmp_path / "indexed")
    pileup_calls = [cmd for cmd in calls if cmd.startswith("samtools mpileup")]
    assert len(pileup_calls) == 1
    assert f'-r "{y_contig}"' in pileup_calls[0]
    table = pd.read_csv(tmp_path / "indexed/sample/sample.out", sep="\t")
    assert table["marker_name"].tolist() == ["m1"]
    assert table["called_base"].tolist() == ["T"]
    assert table["reads"].tolist() == [12]
    assert table["state"].tolist() == ["D"]
    def sequential(bed, bam, pileup, quality, ref, region=None):
        return execute(bed, bam, pileup, quality, ref)
    monkeypatch.setattr(yleaf, "execute_mpileup", sequential)
    assert run(tmp_path / "sequential") == indexed

"""The frozen v1 validation artefacts must never change.

Hashes are of the content committed at tag checkpoint/pre-remediation-56897c8 (LF line endings);
text files are compared after normalising CRLF, so the check works on any git checkout setting.
"""
import hashlib
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = {
    "docs/validation/frozen_decisions.json": "22d40434cdab831991af551c305569c27667961742b5590420f2f12d8513ca6f",
    "docs/validation/stage1_dev_val.md": "eec8664431815de3d1adff4f1d468a40004efa3eb0900df3ea1b647765d7267b",
    "docs/validation/stage2_test.md": "a8e4d216748d66949ae117b64dd7992c148a5bf208f2feaa43d3c4daf733a9d3",
    "docs/validation/ml_mean_reversion.pkl": "6d4b704441b08c7588f8bf12af75a9e004717863d619ea7408b89e7c2429be56",
    "docs/VALIDATION_PROTOCOL.md": "726b2c63bc27fcf13f3c9068d07a97771ebe95ab335d41f29723b54cdd5c0812",
    "docs/REAL_MARKET_EVALUATION.md": "d1a792e9e27343caa180038d23de9cac7dcefb3211b7e9a4d47d5b66225b4c19",
}


def sha(path):
    data = open(os.path.join(ROOT, path), "rb").read()
    # git on Windows may check out text files with CRLF; compare the LF form for text files
    if not path.endswith(".pkl"):
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("path,expected", FROZEN.items())
def test_frozen_v1_artefact_unchanged(path, expected):
    assert sha(path) == expected, f"{path} changed; frozen v1 results must never be modified"

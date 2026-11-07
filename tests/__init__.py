# v12: tests must not write into the project's outputs/ (the oracle cache would otherwise collect entries from test missions)
import os as _os
import tempfile as _tempfile

_os.environ.setdefault("CA_CLP_ORACLE_CACHE", _tempfile.mkdtemp(prefix="ca_clp_test_oracle_cache_"))

PY := .venv/bin/python

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .
	# macOS/iCloud can set the hidden flag on .pth files; python>=3.11 then skips them
	chflags nohidden .venv/lib/python*/site-packages/*.pth 2>/dev/null || true

ingest:
	$(PY) -m daytrader.cli ingest

profile:
	$(PY) -m daytrader.cli profile

test:
	$(PY) -m pytest

.PHONY: setup ingest profile test

.PHONY: tests
tests:
	pytest -vvv -s -l tests

caches:
	find polars_ti -type d -name "__pycache__"
	find tests -type d -name "__pycache__"
	find __pycache__ -type d -name "__pycache__"

clean:
	find polars_ti -type d -name "__pycache__" -exec rm -r {} +
	find tests -type d -name "__pycache__" -exec rm -r {} +
	find __pycache__ -type d -name "__pycache__" -exec rm -r {} +

init:
	uv pip sync requirements.lock

test_metrics:
	pytest -vv -s -l -W ignore::DeprecationWarning --cache-clear tests/test_metrics.py

test_numba:
	pytest -vv -s -l -W ignore::DeprecationWarning --cache-clear tests/test_numba.py

test_studies:
	pytest -vv -s -l -W ignore::DeprecationWarning --cache-clear tests/test_studies.py

test_ti:
	pytest -vv -s -l -W ignore::DeprecationWarning --cache-clear tests/test_indicator_*.py

test_utils:
	pytest -vv -s -l -W ignore::DeprecationWarning --cache-clear tests/test_utils.py

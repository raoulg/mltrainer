format:
	black deeptoolkit
	isort deeptoolkit

lint:
	flake8 deeptoolkit
	mypy deeptoolkit
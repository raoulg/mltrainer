format:
	black mltrainer
	isort mltrainer

lint:
	flake8 mltrainer
	mypy mltrainer
.PHONY: start enter clean

start:
	python3 -m venv .venv
	echo "now run"
	echo "source .venv/bin/activate"
	echo "to activate the virtual environment"

clean:
	rm -r .venv
.PHONY: build shell clean

# Build the simulator image. Takes 15-25 min the first time.
build:
	docker build -t rdma-sim .

# Drop into an interactive shell inside the container with this folder bind-mounted.
shell:
	docker run -it --rm -v "$(PWD)":/work rdma-sim bash

# Wipe simulation outputs but keep the directory structure.
clean:
	rm -f results/*.tr results/*.pcap results/*.csv results/*.txt
	rm -f plots/*.png plots/*.pdf plots/*.svg
	find . -type d -name __pycache__ -exec rm -rf {} +

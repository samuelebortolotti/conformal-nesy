# Define variables
SEEDS := 1011 1213 1415 1617 1819 2021 2223 2425 2627 2829
DIGITS := 3 4 5
CS := 0.0 1.0

.PHONY: run_mnist run_chx

run_mnist:
	@for seed in $(SEEDS); do \
		# Run mnistadd (digit=2) \
		echo "Running seed=$$seed, epochs=5, digit=2 (mnistadd)"; \
		python -m conformal --seed $$seed CONF_MNIST train \
			--concept-sup 0.0 \
			--epochs 4 \
			mnistadd \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_MNIST test \
			--concept-sup 0.0 \
			--epochs 4 \
			mnistadd \
			lenet dpl; \
		\
		# Run mnistaddn for multiple digits \
		for digit in $(DIGITS); do \
			echo "Running seed=$$seed, epochs=5, digit=$$digit (mnistaddn)"; \
			python -m conformal --seed $$seed CONF_MNIST train \
				--concept-sup 0.0 \
				--epochs 4 \
				mnistaddn --n-digits $$digit \
				lenet dpl; \
			python -m conformal --seed $$seed CONF_MNIST test \
				--concept-sup 0.0 \
				--epochs 4 \
				mnistaddn --n-digits $$digit \
				lenet dpl; \
		done; \
	done; \
	echo "Running analyze (mnistaddn)"; \
	python -m conformal CONF_MNIST analyze \
		--concept-sup 0.0 \
		--epochs 4 \
		mnistadd \
		lenet dpl; \
	for digit in $(DIGITS); do \
		echo "Running analyze epochs=5, digit=$$digit (mnistaddn)"; \
		python -m conformal CONF_MNIST analyze \
			--concept-sup 0.0 \
			--epochs 4 \
			mnistaddn --n-digits $$digit \
			lenet dpl; \
	done;


run_mnist_analysis_only:
	echo "Running analyze (mnistaddn)"; \
	python -m conformal CONF_MNIST analyze \
		--concept-sup 0.0 \
		--epochs 5 \
		mnistadd \
		lenet dpl; \
	for digit in $(DIGITS); do \
		echo "Running analyze epochs=5, digit=$$digit (mnistaddn)"; \
		python -m conformal CONF_MNIST analyze \
			--concept-sup 0.0 \
			--epochs 5 \
			mnistaddn --n-digits $$digit \
			lenet dpl; \
	done;


run_chx:
	@for seed in $(SEEDS); do \
		for cs in $(CS); do \
			echo "Running seed=$$seed, epochs=20, cs=$$cs"; \
			python -m conformal --seed $$seed CONF_CHX train \
				--concept-sup $$cs \
				--epochs 20 \
				chx --chx-multi-class \
				resnet18 dpl; \
			python -m conformal --seed $$seed CONF_CHX test \
				--concept-sup $$cs \
				--epochs 20 \
				chx --chx-multi-class \
				resnet18 dpl; \
		done; \
	done; \
	\
	for cs in $(CS); do \
		echo "Running analyze epochs=20, cs=$$cs"; \
		python -m conformal CONF_CHX analyze \
			--concept-sup $$cs \
			--epochs 20 \
			chx --chx-multi-class \
			resnet18 dpl; \
	done;

run_chx_analysis_only:
	for cs in $(CS); do \
		echo "Running analyze epochs=20, cs=$$cs"; \
		python -m conformal CONF_CHX analyze \
			--concept-sup $$cs \
			--epochs 20 \
			chx --chx-multi-class \
			resnet18 dpl; \
	done;
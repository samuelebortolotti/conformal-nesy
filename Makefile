# Define variables
SEEDS := 1011 1213 1415 1617 1819 2021 2223 2425 2627 2829
DIGITS := 3 4 5
CS := 0.0 1.0

.PHONY: run_mnist run_chx

# MNIST

run_mnist:
	@for seed in $(SEEDS); do \
		# Run mnistadd (digit=2) \
		echo "Running seed=$$seed, epochs=20, digit=2 (mnistadd)"; \
		python -m conformal --seed $$seed CONF_MNIST_20_EP train \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistadd \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_MNIST_20_EP train \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistadd \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		python -m conformal --seed $$seed CONF_MNIST_20_EP test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistadd \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_MNIST_20_EP test \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistadd \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		\
		# Run mnistaddn for multiple digits \
		for digit in $(DIGITS); do \
			echo "Running seed=$$seed, epochs=20, digit=$$digit (mnistaddn)"; \
			python -m conformal --seed $$seed CONF_MNIST_20_EP train \
				--learning-rate 0.1 \
				--momentum 0.1 \
				--batch-size 32 \
				--opt sgd \
				--concept-sup 0.0 \
				--epochs 20 \
				mnistaddn --n-digits $$digit \
				lenet dpl; \
			python -m conformal --seed $$seed CONF_MNIST_20_EP test \
				--learning-rate 0.1 \
				--momentum 0.1 \
				--batch-size 32 \
				--opt sgd \
				--concept-sup 0.0 \
				--epochs 20 \
				mnistaddn --n-digits $$digit \
				lenet dpl; \
			python -m conformal --seed $$seed CONF_MNIST_20_EP test \
				--learning-rate 0.1 \
				--momentum 1e-05 \
				--batch-size 64 \
				--opt sgd \
				--concept-sup 0.0 \
				--epochs 20 \
				mnistaddn --n-digits $$digit \
				lenet ltn \
				--and_op prod \
				--or_op prod \
				--imp_op prod \
				--p 8; \
		done; \
	done; \
	echo "Running analyze (mnistaddn)"; \
	python -m conformal CONF_MNIST_20_EP analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 20 \
		mnistadd \
		lenet dpl; \
	python -m conformal CONF_MNIST_20_EP analyze \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 20 \
		mnistadd \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	for digit in $(DIGITS); do \
		echo "Running analyze epochs=20, digit=$$digit (mnistaddn)"; \
		python -m conformal CONF_MNIST_20_EP analyze \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistaddn --n-digits $$digit \
			lenet dpl; \
		python -m conformal CONF_MNIST_20_EP analyze \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistaddn --n-digits $$digit \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
	done;

run_mnist_test:
	@for seed in $(SEEDS); do \
		# Run mnistadd (digit=2) \
		echo "Running seed=$$seed, epochs=20, digit=2 (mnistadd)"; \
		python -m conformal --seed $$seed CONF_MNIST_20_EP test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistadd \
			lenet dpl; \
		\
		python -m conformal --seed $$seed CONF_MNIST_20_EP test \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistadd \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		\
		# Run mnistaddn for multiple digits \
		for digit in $(DIGITS); do \
			echo "Running seed=$$seed, epochs=20, digit=$$digit (mnistaddn)"; \
			python -m conformal --seed $$seed CONF_MNIST_20_EP test \
				--learning-rate 0.1 \
				--momentum 1e-05 \
				--batch-size 64 \
				--opt sgd \
				--concept-sup 0.0 \
				--epochs 20 \
				mnistaddn --n-digits $$digit \
				lenet dpl; \
			python -m conformal --seed $$seed CONF_MNIST_20_EP test \
				--learning-rate 0.1 \
				--momentum 1e-05 \
				--batch-size 64 \
				--opt sgd \
				--concept-sup 0.0 \
				--epochs 20 \
				mnistaddn --n-digits $$digit \
				lenet ltn \
				--and_op prod \
				--or_op prod \
				--imp_op prod \
				--p 8; \
		done; \
	done; \
	echo "Running analyze (mnistaddn)"; \
	python -m conformal CONF_MNIST_20_EP analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 20 \
		mnistadd \
		lenet dpl; \
	python -m conformal CONF_MNIST_20_EP analyze \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 20 \
		mnistadd \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	for digit in $(DIGITS); do \
		echo "Running analyze epochs=20, digit=$$digit (mnistaddn)"; \
		python -m conformal CONF_MNIST_20_EP analyze \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistaddn --n-digits $$digit \
			lenet dpl; \
		python -m conformal CONF_MNIST_20_EP analyze \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistaddn --n-digits $$digit \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
	done;


run_mnist_analysis_only:
	echo "Running analyze (mnistaddn)"; \
	python -m conformal CONF_MNIST_20_EP analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 20 \
		mnistadd \
		lenet dpl; \
	python -m conformal CONF_MNIST_20_EP analyze \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 20 \
		mnistadd \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	for digit in $(DIGITS); do \
		echo "Running analyze epochs=20, digit=$$digit (mnistaddn)"; \
		python -m conformal CONF_MNIST_20_EP analyze \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistaddn --n-digits $$digit \
			lenet dpl; \
		python -m conformal CONF_MNIST_20_EP analyze \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistaddn --n-digits $$digit \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
	done;


run_mnist_deltas_only:
	echo "Running deltas (mnistaddn)"; \
	python -m conformal CONF_MNIST_20_EP deltas \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 20 \
		mnistadd \
		lenet dpl; \
	python -m conformal CONF_MNIST_20_EP deltas \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 20 \
		mnistadd \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	for digit in $(DIGITS); do \
		echo "Running deltas epochs=20, digit=$$digit (mnistaddn)"; \
		python -m conformal CONF_MNIST_20_EP deltas \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistaddn --n-digits $$digit \
			lenet dpl; \
		python -m conformal CONF_MNIST_20_EP deltas \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 20 \
			mnistaddn --n-digits $$digit \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
	done;


# CHX

run_chx:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=200"; \
		python -m conformal --seed $$seed CONF_CHX train \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained dpl; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained dpl; \
		python -m conformal --seed $$seed CONF_CHX train \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained dpl; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained dpl; \
		python -m conformal --seed $$seed CONF_CHX train \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op luk \
			--or_op luk \
			--imp_op luk \
			--p 7; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op luk \
			--or_op luk \
			--imp_op luk \
			--p 7; \
		python -m conformal --seed $$seed CONF_CHX train \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
	done; \
	\
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained ltn \
		--and_op luk \
		--or_op luk \
		--imp_op luk \
		--p 7; \
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3;

run_chx_test:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=20"; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained dpl; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained dpl; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op luk \
			--or_op luk \
			--imp_op luk \
			--p 7; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
	done;

run_chx_analysis_only:
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained ltn \
		--and_op luk \
		--or_op luk \
		--imp_op luk \
		--p 7; \
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3;

run_chx_deltas_only:
	python -m conformal CONF_CHX deltas \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CHX deltas \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CHX deltas \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained ltn \
		--and_op luk \
		--or_op luk \
		--imp_op luk \
		--p 7; \
	python -m conformal CONF_CHX deltas \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3;


# RIVAL AND CIFAR

run_rival:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=200"; \
		python -m conformal --seed $$seed CONF_RIVAL train \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 dpl; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 dpl; \
		python -m conformal --seed $$seed CONF_RIVAL train \
			--learning-rate 0.001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt sgd \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 dpl; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 0.001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt sgd \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 dpl; \
		python -m conformal --seed $$seed CONF_RIVAL train \
			--learning-rate 1e-5 \
			--momentum 0.0001 \
			--batch-size 32 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 6; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 1e-5 \
			--momentum 0.0001 \
			--batch-size 32 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 6; \
		python -m conformal --seed $$seed CONF_RIVAL train \
			--learning-rate 1e-5 \
			--momentum 0.0001 \
			--batch-size 32 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 6; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 1e-5 \
			--momentum 0.0001 \
			--batch-size 32 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 6; \
	done; \
	\
	echo "Running analyze epochs=200"; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		rival \
		resnet18 dpl; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 0.001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt sgd \
		--concept-sup 1.0 \
		--epochs 200 \
		rival \
		resnet18 dpl; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6;


run_rival_test:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=200"; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 dpl; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 0.001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt sgd \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 dpl; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 1e-5 \
			--momentum 0.0001 \
			--batch-size 32 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 6; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 1e-5 \
			--momentum 0.0001 \
			--batch-size 32 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 6; \
	done; \


run_rival_analysis_only:
	echo "Running analyze epochs=200"; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		rival \
		resnet18 dpl; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 0.001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt sgd \
		--concept-sup 1.0 \
		--epochs 200 \
		rival \
		resnet18 dpl; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6;

run_rival_deltas_only:
	echo "Running deltas epochs=200"; \
	python -m conformal CONF_RIVAL deltas \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		rival \
		resnet18 dpl; \
	python -m conformal CONF_RIVAL deltas \
		--learning-rate 0.001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt sgd \
		--concept-sup 1.0 \
		--epochs 200 \
		rival \
		resnet18 dpl; \
	python -m conformal CONF_RIVAL deltas \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6; \
	python -m conformal CONF_RIVAL deltas \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6;


run_cifar:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=200"; \
		python -m conformal --seed $$seed CONF_CIFAR train \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained dpl; \
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained dpl; \
		\
		python -m conformal --seed $$seed CONF_CIFAR train \
			--learning-rate 0.001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt sgd \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained dpl; \
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt sgd \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained dpl; \
		\
		python -m conformal --seed $$seed CONF_CIFAR train \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		\
		python -m conformal --seed $$seed CONF_CIFAR train \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
	done; \
	\
	python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt sgd \
		--concept-sup 1.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3; \
	python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3;

run_cifar_test:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=200"; \
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.0001 \
			--momentum 0.01 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained dpl; \
		\
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt sgd \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained dpl; \
		\
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		\
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
	done; \

run_cifar_analysis_only:
		python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt sgd \
		--concept-sup 1.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3; \
	python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3;


run_cifar_deltas_only:
		python -m conformal CONF_CIFAR deltas \
		--learning-rate 0.0001 \
		--momentum 0.01 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CIFAR deltas \
		--learning-rate 0.001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt sgd \
		--concept-sup 1.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained dpl; \
	python -m conformal CONF_CIFAR deltas \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3; \
	python -m conformal CONF_CIFAR deltas \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3;

# CEBAB

run_cebab:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=500"; \
		python -m conformal --seed $$seed CONF_CEBAB train \
			--learning-rate 0.01 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 500 \
			cebab \
			bert dpl; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.01 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 500 \
			cebab \
			bert dpl; \
		python -m conformal --seed $$seed CONF_CEBAB train \
			--learning-rate 0.01 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 500 \
			cebab \
			bert dpl; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.01 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 500 \
			cebab \
			bert dpl; \
		python -m conformal --seed $$seed CONF_CEBAB train \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		python -m conformal --seed $$seed CONF_CEBAB train \
			--learning-rate 0.01 \
			--momentum 0.001 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 8; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.01 \
			--momentum 0.001 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 8; \
	done; \
	\
	echo "Running analyze epochs=500"; \
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.01 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 500 \
		cebab \
		bert dpl; \
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.01 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 500 \
		cebab \
		bert dpl; \
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 500 \
		cebab \
		bert ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3; \
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 500 \
		cebab \
		bert ltn \
		--and_op godel \
		--or_op godel \
		--imp_op godel \
		--p 8;

run_cebab_test:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=500"; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.01 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 500 \
			cebab \
			bert dpl; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.01 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 500 \
			cebab \
			bert dpl; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.01 \
			--momentum 0.001 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 8; \
	done;


run_cebab_analysis_only:
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.01 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 500 \
		cebab \
		bert dpl; \
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.01 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 500 \
		cebab \
		bert dpl; \
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 500 \
		cebab \
		bert ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3; \
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 500 \
		cebab \
		bert ltn \
		--and_op godel \
		--or_op godel \
		--imp_op godel \
		--p 8;


run_cebab_deltas_only:
	python -m conformal CONF_CEBAB deltas \
		--learning-rate 0.01 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 500 \
		cebab \
		bert dpl; \
	python -m conformal CONF_CEBAB deltas \
		--learning-rate 0.01 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 500 \
		cebab \
		bert dpl; \
	python -m conformal CONF_CEBAB deltas \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 500 \
		cebab \
		bert ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 3; \
	python -m conformal CONF_CEBAB deltas \
		--learning-rate 0.0001 \
		--momentum 0.99 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 500 \
		cebab \
		bert ltn \
		--and_op godel \
		--or_op godel \
		--imp_op godel \
		--p 8;


# RSs

run_rss:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=20"; \
		python -m conformal --seed $$seed CONF_RSs train \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs train \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		python -m conformal --seed $$seed CONF_RSs train \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs train \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		python -m conformal --seed $$seed CONF_RSs train \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs train \
			--learning-rate 0.01 \
			--momentum 0.001 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 8; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.01 \
			--momentum 0.001 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 8; \
	done; \
	\
	echo "Running analyze epochs=20"; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnisthalf \
		lenet dpl; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnisthalf \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistsump \
		lenet dpl; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistsump \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistevenodd \
		lenet dpl; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.01 \
		--momentum 0.001 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistevenodd \
		lenet ltn \
		--and_op godel \
		--or_op godel \
		--imp_op godel \
		--p 8;

run_rss_test:
	@for seed in $(SEEDS); do \
		echo "Running seed=$$seed, epochs=20"; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.01 \
			--momentum 0.001 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 8; \
	done; \


run_rss_analysis_only:
	echo "Running analyze epochs=20"; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnisthalf \
		lenet dpl; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnisthalf \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistsump \
		lenet dpl; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistsump \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistevenodd \
		lenet dpl; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.01 \
		--momentum 0.001 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistevenodd \
		lenet ltn \
		--and_op godel \
		--or_op godel \
		--imp_op godel \
		--p 8;


run_rss_deltas_only:
	echo "Running deltas epochs=20"; \
	python -m conformal CONF_RSs deltas \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnisthalf \
		lenet dpl; \
	python -m conformal CONF_RSs deltas \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnisthalf \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	python -m conformal CONF_RSs deltas \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistsump \
		lenet dpl; \
	python -m conformal CONF_RSs deltas \
		--learning-rate 0.1 \
		--momentum 1e-05 \
		--batch-size 64 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistsump \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 8; \
	python -m conformal CONF_RSs deltas \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistevenodd \
		lenet dpl; \
	python -m conformal CONF_RSs deltas \
		--learning-rate 0.01 \
		--momentum 0.001 \
		--batch-size 128 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistevenodd \
		lenet ltn \
		--and_op godel \
		--or_op godel \
		--imp_op godel \
		--p 8;


run_ltn_optimize:
	python -m conformal CONF_A optuna \
		--n-trials 100 \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 100 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6; \
	\
	python -m conformal CONF_A optuna \
		--n-trials 100 \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 100 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6; \
	\
	python -m conformal CONF_A optuna \
		--n-trials 200 \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 300 \
		cebab \
		bert ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6; \
	\
	python -m conformal CONF_A optuna \
		--n-trials 200 \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 300 \
		chx --chx-multi-class\
		resnet --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6; \
	\
	python -m conformal CONF_A optuna \
		--n-trials 200 \
		--learning-rate 1e-5 \
		--momentum 0.0001 \
		--batch-size 32 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 50 \
		mnisteovenodd \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op prod \
		--p 6; \

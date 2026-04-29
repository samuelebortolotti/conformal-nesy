# Define variables
SEEDS := 1011 1213 1415 1617 1819 2021 2223 2425 2627 2829
DIGITS := 3 4 5
CS := 0.0 1.0

.PHONY: run_mnist run_chx run_rival run_cifar run_cebab run_rss run_time

# MNISTADD

run_mnist:
	@for seed in $(SEEDS); do \
		echo "Running mnistadd DPL seed=$$seed"; \
		python -m conformal --seed $$seed CONF_MNISTADD train \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistadd \
			lenet dpl; \
		python -m conformal --seed $$seed CONF_MNISTADD test \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistadd \
			lenet dpl; \
		echo "Running mnistadd LTN cs=0.0 seed=$$seed"; \
		python -m conformal --seed $$seed CONF_MNISTADD train \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistadd \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_MNISTADD test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistadd \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		echo "Running mnistadd LTN cs=1.0 seed=$$seed"; \
		python -m conformal --seed $$seed CONF_MNISTADD train \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			mnistadd \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_MNISTADD test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			mnistadd \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		for digit in $(DIGITS); do \
			echo "Running mnistaddn DPL seed=$$seed digit=$$digit"; \
			python -m conformal --seed $$seed CONF_MNISTADD train \
				--learning-rate 0.1 \
				--momentum 0.1 \
				--batch-size 32 \
				--opt sgd \
				--concept-sup 0.0 \
				--epochs 200 \
				mnistaddn --n-digits $$digit \
				lenet dpl; \
			python -m conformal --seed $$seed CONF_MNISTADD test \
				--learning-rate 0.1 \
				--momentum 0.1 \
				--batch-size 32 \
				--opt sgd \
				--concept-sup 0.0 \
				--epochs 200 \
				mnistaddn --n-digits $$digit \
				lenet dpl; \
			echo "Running mnistaddn LTN cs=1.0 seed=$$seed digit=$$digit"; \
			python -m conformal --seed $$seed CONF_MNISTADD train \
				--learning-rate 0.0001 \
				--momentum 0.9 \
				--batch-size 256 \
				--opt adam \
				--concept-sup 1.0 \
				--epochs 200 \
				mnistaddn --n-digits $$digit \
				lenet ltn \
				--and_op prod \
				--or_op prod \
				--imp_op goguen \
				--p 4; \
			python -m conformal --seed $$seed CONF_MNISTADD test \
				--learning-rate 0.0001 \
				--momentum 0.9 \
				--batch-size 256 \
				--opt adam \
				--concept-sup 1.0 \
				--epochs 200 \
				mnistaddn --n-digits $$digit \
				lenet ltn \
				--and_op prod \
				--or_op prod \
				--imp_op goguen \
				--p 4; \
		done; \
	done; \
	python -m conformal CONF_MNISTADD analyze \
		--learning-rate 0.1 \
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistadd \
		lenet dpl; \
	python -m conformal CONF_MNISTADD analyze \
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistadd \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4; \
	python -m conformal CONF_MNISTADD analyze \
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		mnistadd \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4; \
	for digit in $(DIGITS); do \
		python -m conformal CONF_MNISTADD analyze \
			--learning-rate 0.1 \
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistaddn --n-digits $$digit \
			lenet dpl; \
		python -m conformal CONF_MNISTADD analyze \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			mnistaddn --n-digits $$digit \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
	done

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
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CHX train \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CHX test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
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
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4; \
	python -m conformal CONF_CHX analyze \
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		chx --chx-multi-class \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4;

# RIVAL-10

run_rival:
	@for seed in $(SEEDS); do \
		echo "Running RIVAL seed=$$seed, epochs=200"; \
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
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RIVAL train \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RIVAL test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
	done; \
	\
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
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4; \
	python -m conformal CONF_RIVAL analyze \
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		rival \
		resnet18 ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4

# CIFAR-10

run_cifar:
	@for seed in $(SEEDS); do \
		echo "Running CIFAR seed=$$seed, epochs=200"; \
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
		python -m conformal --seed $$seed CONF_CIFAR train \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CIFAR train \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CIFAR test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
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
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4; \
	python -m conformal CONF_CIFAR analyze \
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 200 \
		cifar \
		resnet18 --pretrained ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4

# CEBaB

run_cebab:
	@for seed in $(SEEDS); do \
		echo "Running CEBaB seed=$$seed, epochs=500"; \
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
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CEBAB train \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CEBAB test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 500 \
			cebab \
			bert ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
	done; \
	\
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
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 500 \
		cebab \
		bert ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4; \
	python -m conformal CONF_CEBAB analyze \
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 1.0 \
		--epochs 500 \
		cebab \
		bert ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4

# RSS (mnisthalf, mnistsump, mnistevenodd)

run_rss:
	@for seed in $(SEEDS); do \
		echo "Running RSS seed=$$seed, epochs=200"; \
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
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet dpl; \
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
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RSs train \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RSs train \
			--learning-rate 0.01 \
			--momentum 0.99 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 3; \
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
			--momentum 0.1 \
			--batch-size 32 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet dpl; \
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
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RSs test \
			--learning-rate 0.01 \
			--momentum 0.99 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 3; \
	done; \
	\
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
		--momentum 0.1 \
		--batch-size 32 \
		--opt sgd \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistsump \
		lenet dpl; \
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
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		mnisthalf \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.0001 \
		--momentum 0.9 \
		--batch-size 256 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistsump \
		lenet ltn \
		--and_op prod \
		--or_op prod \
		--imp_op goguen \
		--p 4; \
	python -m conformal CONF_RSs analyze \
		--learning-rate 0.01 \
		--momentum 0.99 \
		--batch-size 64 \
		--opt adam \
		--concept-sup 0.0 \
		--epochs 200 \
		mnistevenodd \
		lenet ltn \
		--and_op godel \
		--or_op godel \
		--imp_op godel \
		--p 3

# Timing experiments

run_time:
	@for seed in $(SEEDS); do \
		echo "=== Timing seed=$$seed ==="; \
		python -m conformal --seed $$seed CONF_MNISTADD timing \
			--learning-rate 0.1 \
			--momentum 1e-05 \
			--batch-size 64 \
			--opt sgd \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistadd \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 8; \
		python -m conformal --seed $$seed CONF_RSs timing \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnisthalf \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RSs timing \
			--learning-rate 0.01 \
			--momentum 0.99 \
			--batch-size 64 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistevenodd \
			lenet ltn \
			--and_op godel \
			--or_op godel \
			--imp_op godel \
			--p 3; \
		python -m conformal --seed $$seed CONF_RSs timing \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			mnistsump \
			lenet ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_CHX timing \
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
		python -m conformal --seed $$seed CONF_CHX timing \
			--learning-rate 0.0001 \
			--momentum 0.99 \
			--batch-size 128 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			chx --chx-multi-class \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op prod \
			--p 3; \
		python -m conformal --seed $$seed CONF_RIVAL timing \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal --seed $$seed CONF_RIVAL timing \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			rival \
			resnet18 ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal CONF_CIFAR timing \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 0.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
		python -m conformal CONF_CIFAR timing \
			--learning-rate 0.0001 \
			--momentum 0.9 \
			--batch-size 256 \
			--opt adam \
			--concept-sup 1.0 \
			--epochs 200 \
			cifar \
			resnet18 --pretrained ltn \
			--and_op prod \
			--or_op prod \
			--imp_op goguen \
			--p 4; \
	done

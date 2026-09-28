.PHONY: download_mnist download_cifar download_rival download_cebab download_chx download_all

download_mnist:
	python -c "from torchvision.datasets import MNIST; MNIST('./data', download=True)"

download_cifar:
	python -c "from torchvision.datasets import CIFAR10; CIFAR10('./data', download=True)"

download_rival:
	@echo "RIVAL-10 must be downloaded manually from Kaggle (search 'rival10')."
	@echo "After downloading, extract so that data/RIVAL10/meta/ and data/RIVAL10/<split>/ exist."

download_cebab:
	python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='CEBaB/CEBaB', repo_type='dataset', local_dir='data/cebab')"

download_chx:
	@echo "ChestX-ray (NIH) requires manual placement of CSV annotation files."
	@echo "Download from https://nihcc.app.box.com/v/ChestXray-NIHCC and place:"
	@echo "  data/four_findings_expert_labels_individual_readers.csv"
	@echo "  data/four_findings_expert_labels_test_labels.csv"
	@echo "  data/four_findings_expert_labels_validation_labels.csv"
	@echo "Images are downloaded automatically by the loader on first use."

download_all: download_mnist download_cifar download_cebab
	@echo "Auto-downloadable datasets fetched. See 'make download_rival' and 'make download_chx' for manual steps."

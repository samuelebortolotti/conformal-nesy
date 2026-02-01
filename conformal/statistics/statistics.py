import json


class Statistics:
    def __init__(self):
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        self.train_c_f1_scores = []
        self.val_c_f1_scores = []
        self.best_f1 = 0
        self.best_model = None

    def log(
        self, epoch, train_loss, train_f1, val_f1, train_c_f1, val_c_f1, val_loss, model
    ):
        self.epochs.append(epoch)

        self.train_losses.append(train_loss)
        self.train_f1_scores.append(train_f1)

        self.val_losses.append(val_loss)
        self.val_f1_scores.append(val_f1)
        self.val_c_f1_scores.append(val_c_f1)
        self.train_c_f1_scores.append(train_c_f1)

        if val_f1 > self.best_f1 or self.best_model is None:
            self.best_f1 = val_f1
            self.best_model = model.state_dict()

    def get_statistics(self):
        return [
            [
                "Epoch",
                "Train Loss",
                "Train F1",
                "Val Loss",
                "Val F1",
                "Train C F1",
                "Val C F1",
            ],
            [
                json.dumps(self.epochs),
                json.dumps(self.train_losses),
                json.dumps(self.train_f1_scores),
                json.dumps(self.val_losses),
                json.dumps(self.val_f1_scores),
                json.dumps(self.train_c_f1_scores),
                json.dumps(self.val_c_f1_scores),
            ],
        ]


class Results:
    def __init__(
        self,
        test_loss,
        test_f1,
        yece,
        cece,
        h_c,
        h_c_per_value,
        all_labels,
        all_preds,
        all_g,
        all_c,
        all_label_prob,
        all_concept_prob,
    ):
        self.test_loss = test_loss
        self.test_f1 = test_f1
        self.yece = yece
        self.cece = cece
        self.h_c = h_c
        self.h_c_per_value = h_c_per_value
        self.all_labels = all_labels
        self.all_preds = all_preds
        self.all_g = all_g
        self.all_c = all_c
        self.all_label_prob = all_label_prob
        self.all_concept_prob = all_concept_prob

    def get_statistics(self):
        return [
            [
                "Test Loss",
                "Test F1",
                "Y ECE",
                "C ECE",
                "H(C|X)",
                "H(ci|X)",
                "Y Labels",
                "Y Predictions",
                "G",
                "C",
                "Y Label Probabilities",
                "C Concept Probabilities",
            ],
            [
                self.test_loss,
                self.test_f1,
                self.yece,
                self.cece,
                self.h_c,
                self.h_c_per_value,
                json.dumps(self.all_labels),
                json.dumps(self.all_preds),
                json.dumps(self.all_g),
                json.dumps(self.all_c),
                json.dumps(self.all_label_prob),
                json.dumps(self.all_concept_prob),
            ],
        ]

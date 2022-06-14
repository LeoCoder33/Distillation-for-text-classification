import tokenization

from Processor.DataProcessor import DataProcessor, InputExample
import pandas as pd
import os


class AGNewsProcessor(DataProcessor):
    """Processor for the AG data set."""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[1] + "-" + line[2])
            label = tokenization.convert_to_unicode(str(line[0]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples

    def get_train_examples(self, data_dir):
        train_data = pd.read_csv(os.path.join(data_dir, "train.csv"), header=None).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        dev_data = pd.read_csv(os.path.join(data_dir, "test.csv"), header=None).values
        return self._create_examples(dev_data, "dev")

    def get_labels(self):
        return ["1", "2", "3", "4"]


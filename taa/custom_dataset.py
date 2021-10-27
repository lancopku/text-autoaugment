from torch.utils.data import Dataset
from theconf import Config as C
from transformers.data.processors.utils import InputFeatures


class general_dataset(Dataset):
    def __init__(self, examples, tokenizer, text_transform=None):
        super(Dataset, self).__init__()
        max_seq_length = C.get()['max_seq_length']
        self.texts = [d.text_a for d in examples]
        self.texts = [" ".join(t.split()[:max_seq_length]) if len(t.split()) > max_seq_length else t
                      for t in self.texts]  # clip texts
        self.labels = [d.label if d.label >= 0 else d.label + 1 for d in examples]
        self.aug_n_dist = 0

        if text_transform and text_transform.transforms:
            texts_aug, labels_aug, n_dist = text_transform(self.texts, self.labels)
            assert len(texts_aug) == len(labels_aug)
            self.texts = texts_aug
            self.labels = labels_aug
            self.aug_n_dist = n_dist

        # convert words to tokens and then to ids
        self.features = []
        tmp_features = tokenizer(self.texts, max_length=max_seq_length, padding='max_length', truncation=True)
        for i in range(len(self.texts)):
            self.features.append(InputFeatures(
                input_ids=tmp_features['input_ids'][i], attention_mask=tmp_features['attention_mask'][i],
                token_type_ids=tmp_features['token_type_ids'][i], label=self.labels[i]
            ))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        return self.features[item]

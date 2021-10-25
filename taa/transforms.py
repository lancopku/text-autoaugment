from __future__ import division
import sys
import collections


if sys.version_info < (3, 3):
    Sequence = collections.Sequence
    Iterable = collections.Iterable
else:
    Sequence = collections.abc.Sequence
    Iterable = collections.abc.Iterable


__all__ = ["Compose"]


class Compose(object):
    """Composes several transforms together.

    Args:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> text_transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, texts, labels):
        for t in self.transforms:
            texts, labels, n_dist_value = t(texts, labels)
        return texts, labels, n_dist_value

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


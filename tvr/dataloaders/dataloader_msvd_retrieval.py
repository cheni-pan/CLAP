from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import pickle
from collections import OrderedDict
from os.path import exists, join, splitext

from .dataloader_retrieval import RetrievalDataset


class MSVDDataset(RetrievalDataset):
    """MSVD dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=32,
                 max_frames=12, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(MSVDDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                          max_frames, video_framerate, image_resolution, mode, config=config)

    @staticmethod
    def _caption_to_text(caption):
        if isinstance(caption, str):
            return caption
        if isinstance(caption, (list, tuple)):
            return " ".join(str(token) for token in caption)
        return str(caption)

    def _scan_video_paths(self):
        video_paths = {}
        for root, _, video_files in os.walk(self.video_path):
            for video_file in video_files:
                video_id = splitext(video_file)[0]
                video_paths[video_id] = join(root, video_file)
        return video_paths

    def _get_anns(self, subset='train'):
        """
        video_dict: dict: video_id -> video_path
        sentences_dict: list: [(video_id, caption)] , caption (list: [text:, start, end])
        """
        split_file = {
            'train': 'train_list.txt',
            'val': 'val_list.txt',
            'test': 'test_list.txt',
            'train_test': 'train_list.txt',
        }[subset]
        split_path = join(self.anno_path, split_file)
        caption_path = join(self.anno_path, 'raw-captions.pkl')

        if not exists(split_path):
            raise FileNotFoundError(split_path)
        if not exists(caption_path):
            raise FileNotFoundError(caption_path)

        with open(split_path, 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines() if itm.strip()]

        with open(caption_path, 'rb') as fp:
            captions = pickle.load(fp)

        all_video_paths = self._scan_video_paths()

        video_dict = OrderedDict()
        sentences_dict = OrderedDict()
        cut_off_points = []
        missing_captions = []
        missing_videos = []

        for video_id in video_ids:
            if video_id not in captions:
                missing_captions.append(video_id)
                continue
            if video_id not in all_video_paths:
                missing_videos.append(video_id)
                continue

            video_dict[video_id] = all_video_paths[video_id]
            caption_items = captions[video_id]
            if subset == 'train_test':
                caption_items = caption_items[:1]
            for caption in caption_items:
                caption_text = self._caption_to_text(caption)
                sentences_dict[len(sentences_dict)] = (video_id, (caption_text, None, None))
            cut_off_points.append(len(sentences_dict))

        if missing_captions:
            raise KeyError("Missing MSVD captions for {} videos, e.g. {}".format(
                len(missing_captions), missing_captions[:5]))
        if missing_videos:
            raise FileNotFoundError("Missing MSVD video files for {} videos under {}, e.g. {}".format(
                len(missing_videos), self.video_path, missing_videos[:5]))

        self.cut_off_points = cut_off_points
        self.sentence_num = len(sentences_dict)
        self.video_num = len(video_dict)
        self.multi_sentence_per_video = subset in ['val', 'test']

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print('[{}] Unique sentence is {} , all num is {}'.format(subset, len(unique_sentence), len(sentences_dict)))

        return video_dict, sentences_dict

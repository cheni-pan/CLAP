from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import math
import os
from collections import OrderedDict
from os.path import join

from .dataloader_didemo_retrieval import _looks_like_failed_download
from .dataloader_retrieval import RetrievalDataset


class ActivityNetDataset(RetrievalDataset):
    """ActivityNet video-paragraph retrieval dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=64,
                 max_frames=64, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(ActivityNetDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                                 max_frames, video_framerate, image_resolution, mode, config=config)

    def _load_json(self, path):
        print("Loading json: {}".format(path))
        with open(path, "r") as fp:
            return json.load(fp)

    def _load_caption_source(self, json_data, valid_video_ids):
        caption_dict = OrderedDict()
        for video_id, item in json_data.items():
            if video_id not in valid_video_ids:
                continue
            paragraph = " ".join(item["sentences"])
            if video_id not in caption_dict:
                caption_dict[video_id] = {"text": [], "start": [], "end": []}
            caption_dict[video_id]["text"].append(paragraph)
            caption_dict[video_id]["start"].append(0)
            caption_dict[video_id]["end"].append(int(math.ceil(float(item["duration"]))))
        return caption_dict

    def _build_video_dict(self, valid_video_ids):
        video_dict = OrderedDict()
        for root, _, files in os.walk(self.video_path):
            for video_file in files:
                video_id = ".".join(video_file.split(".")[:-1])
                if video_id not in valid_video_ids:
                    continue
                file_path = join(root, video_file)
                if _looks_like_failed_download(file_path):
                    continue
                video_dict[video_id] = file_path
        return video_dict

    def _get_anns(self, subset="train"):
        """
        video_dict: video_id -> video_path
        sentences_dict: index -> (video_id, (paragraph_text, start, end))
        """
        assert subset in ["train", "val"]

        if subset == "train":
            train_ids = self._load_json(join(self.anno_path, "train_ids.json"))
            train_json = self._load_json(join(self.anno_path, "train.json"))
            video_ids = train_ids
            caption_sources = [self._load_caption_source(train_json, set(video_ids))]
            self.multi_sentence_per_video = False
        else:
            val_ids = self._load_json(join(self.anno_path, "val_ids.json"))
            val1_json = self._load_json(join(self.anno_path, "val_1.json"))
            video_ids = val_ids
            caption_sources = [self._load_caption_source(val1_json, set(video_ids))]
            self.multi_sentence_per_video = False

        video_dict = self._build_video_dict(set(video_ids))
        available_video_ids = set(video_dict.keys())
        sentences_dict = OrderedDict()
        cut_off_points = []

        for video_id in video_ids:
            if video_id not in available_video_ids:
                continue

            has_caption = False
            for caption_source in caption_sources:
                if video_id not in caption_source:
                    continue
                caption = caption_source[video_id]
                sentences_dict[len(sentences_dict)] = (
                    video_id,
                    (caption["text"][0], caption["start"][0], caption["end"][0]),
                )
                has_caption = True

            if has_caption:
                cut_off_points.append(len(sentences_dict))

        self.cut_off_points = cut_off_points
        self.sentence_num = len(sentences_dict)
        self.video_num = len(cut_off_points) if self.multi_sentence_per_video else len(video_dict)

        if self.multi_sentence_per_video:
            print("For {}, sentence number: {}".format(subset, self.sentence_num))
            print("For {}, video number: {}".format(subset, self.video_num))

        return video_dict, sentences_dict

    def __getitem__(self, idx):
        if self.mode == "all":
            video_id, caption = self.sentences_dict[idx]
            text_ids, text_mask, s, e = self._get_text(caption)
            video, video_mask = self._get_rawvideo_with_decoder(video_id, s, e)
            return text_ids, text_mask, video, video_mask, idx, hash(video_id)
        return super(ActivityNetDataset, self).__getitem__(idx)

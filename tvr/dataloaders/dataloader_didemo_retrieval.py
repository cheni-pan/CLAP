from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import os
from collections import OrderedDict
from os.path import join

import numpy as np

from .dataloader_retrieval import RetrievalDataset


MIN_VIDEO_BYTES = 10 * 1024


def _looks_like_failed_download(file_path):
    try:
        if os.path.getsize(file_path) < MIN_VIDEO_BYTES:
            return True
        with open(file_path, "rb") as fp:
            head = fp.read(256).lstrip().lower()
        return head.startswith(b"<!doctype html") or head.startswith(b"<html")
    except OSError:
        return True


class DiDeMoDataset(RetrievalDataset):
    """DiDeMo video-paragraph retrieval dataset."""

    def __init__(self, subset, anno_path, video_path, tokenizer, max_words=64,
                 max_frames=64, video_framerate=1, image_resolution=224, mode='all', config=None):
        super(DiDeMoDataset, self).__init__(subset, anno_path, video_path, tokenizer, max_words,
                                            max_frames, video_framerate, image_resolution, mode, config=config)

    def _get_anns(self, subset='train'):
        """
        video_dict: video_id -> video_path
        sentences_dict: index -> (video_id, (paragraph_text, start, end))
        """
        assert subset in ["train", "val", "test"]

        split_path = join(self.anno_path, "{}_list.txt".format(subset))
        anno_path = join(self.anno_path, "{}_data.json".format(subset))

        with open(split_path, "r") as fp:
            video_ids = [line.strip() for line in fp if line.strip()]
        video_id_set = set(video_ids)

        caption_dict = OrderedDict()
        with open(anno_path, "r") as fp:
            data = json.load(fp)

        for item in data:
            video_id = item["video"]
            if video_id not in video_id_set:
                continue

            description = item["description"]
            times = item["times"]
            start_time = np.mean([t[0] for t in times]) * 5
            end_time = (np.mean([t[1] for t in times]) + 1) * 5

            if video_id not in caption_dict:
                caption_dict[video_id] = {"text": [], "start": [], "end": []}
            caption_dict[video_id]["text"].append(description)
            caption_dict[video_id]["start"].append(start_time)
            caption_dict[video_id]["end"].append(end_time)

        video_dict = OrderedDict()
        for root, _, files in os.walk(self.video_path):
            for video_file in files:
                if video_file not in video_id_set:
                    continue
                file_path = join(root, video_file)
                if _looks_like_failed_download(file_path):
                    continue
                video_dict[video_file] = file_path

        sentences_dict = OrderedDict()
        for video_id in video_ids:
            if video_id not in caption_dict or video_id not in video_dict:
                continue
            # CLIP4Clip evaluates DiDeMo as video-paragraph retrieval.
            paragraph = " ".join(caption_dict[video_id]["text"])
            sentences_dict[len(sentences_dict)] = (video_id, (paragraph, 0, 31))

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print('[{}] Unique sentence is {} , all num is {}'.format(subset, len(unique_sentence), len(sentences_dict)))

        return video_dict, sentences_dict

    def __getitem__(self, idx):
        if self.mode == 'all':
            video_id, caption = self.sentences_dict[idx]
            text_ids, text_mask, s, e = self._get_text(caption)
            video_path = self.video_dict[video_id]
            try:
                if video_path.lower().endswith(".wmv"):
                    video, video_mask = self._get_rawvideo(video_id, s, e)
                else:
                    video, video_mask = self._get_rawvideo_dec(video_id, s, e)
            except Exception as excep:
                print("decord video path: {} error. video id: {}, Error: {}".format(
                    video_path, video_id, excep))
                video, video_mask = self._get_rawvideo(video_id, s, e)
            return text_ids, text_mask, video, video_mask, idx, hash(video_id)
        return super(DiDeMoDataset, self).__getitem__(idx)

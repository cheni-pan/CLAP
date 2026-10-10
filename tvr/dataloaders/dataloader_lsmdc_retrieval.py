from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import os
from collections import OrderedDict
from os.path import exists, join

from .dataloader_retrieval import RetrievalDataset


class LSMDCClip4ClipDataset(RetrievalDataset):
    """LSMDC loader using the CLIP4Clip/DGL-style video_path.json mapping."""

    ANNO_FILES = {
        "train": "LSMDC16_annos_training.csv",
        "train_test": "LSMDC16_annos_val.csv",
        "val": "LSMDC16_annos_val.csv",
        "test": "LSMDC16_challenge_1000_publictect.csv",
    }

    def __init__(
            self,
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words=32,
            max_frames=12,
            video_framerate=1,
            image_resolution=224,
            mode="all",
            config=None,
    ):
        super(LSMDCClip4ClipDataset, self).__init__(
            subset,
            anno_path,
            video_path,
            tokenizer,
            max_words,
            max_frames,
            video_framerate,
            image_resolution,
            mode,
            config=config,
        )

    def _load_video_path_json(self):
        json_path = os.environ.get("VIDEO_PATH_JSON") or join(self.anno_path, "video_path.json")
        if not exists(json_path):
            raise FileNotFoundError(
                "{} is required for CLIP4Clip-style LSMDC loading. "
                "Generate it as clip_id -> relative video path under the video directory.".format(json_path)
            )
        with open(json_path, "r", encoding="utf-8") as fp:
            return json.load(fp)

    def _get_anns(self, subset="train"):
        """
        video_dict: OrderedDict video_id -> absolute video path
        sentences_dict: OrderedDict index -> (video_id, (text, start, end))
        """
        if subset not in self.ANNO_FILES:
            raise ValueError("Unsupported LSMDC subset: {}".format(subset))

        anno_file = join(self.anno_path, self.ANNO_FILES[subset])
        if not exists(anno_file):
            raise FileNotFoundError(anno_file)

        video_id_list = []
        caption_items = []
        with open(anno_file, "r", encoding="utf-8") as fp:
            for line_no, line in enumerate(fp, start=1):
                line = line.strip()
                if not line:
                    continue
                line_split = line.split("\t")
                if len(line_split) != 6:
                    raise ValueError(
                        "Invalid LSMDC annotation line {} in {}: expected 6 tab-separated fields".format(
                            line_no,
                            anno_file,
                        )
                    )
                clip_id, _, _, _, _, sentence = line_split
                caption_items.append((clip_id, (sentence, None, None)))
                if clip_id not in video_id_list:
                    video_id_list.append(clip_id)

        all_video_dict = self._load_video_path_json()

        video_dict = OrderedDict()
        missing_in_json = []
        missing_files = []
        for clip_id in video_id_list:
            rel_path = all_video_dict.get(clip_id)
            if rel_path is None:
                missing_in_json.append(clip_id)
                continue
            file_path = join(self.video_path, rel_path)
            if not exists(file_path):
                missing_files.append(clip_id)
                continue
            video_dict[clip_id] = file_path

        sentences_dict = OrderedDict()
        for clip_id, sentence in caption_items:
            if clip_id not in video_dict:
                continue
            sentences_dict[len(sentences_dict)] = (clip_id, sentence)

        if missing_in_json:
            print("[{}] Missing LSMDC ids in video_path.json: {} skipped, examples: {}".format(
                subset,
                len(missing_in_json),
                missing_in_json[:5],
            ))
        if missing_files:
            print("[{}] Missing LSMDC video files referenced by video_path.json: {} skipped, examples: {}".format(
                subset,
                len(missing_files),
                missing_files[:5],
            ))

        if not video_dict or not sentences_dict:
            raise FileNotFoundError(
                "No LSMDC samples matched {} with video_path.json under {}".format(
                    anno_file,
                    self.video_path,
                )
            )

        unique_sentence = set([v[1][0] for v in sentences_dict.values()])
        print("[{}] Unique sentence is {} , all num is {}".format(
            subset,
            len(unique_sentence),
            len(sentences_dict),
        ))

        return video_dict, sentences_dict


LsmdcDataset = LSMDCClip4ClipDataset

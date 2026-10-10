import torch
from torch.utils.data import DataLoader

def dataloader_msrvtt_train(args, tokenizer):
    from .dataloader_msrvtt_retrieval import MSRVTTDataset

    msrvtt_dataset = MSRVTTDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, subset="test"):
    from .dataloader_msrvtt_retrieval import MSRVTTDataset

    msrvtt_testset = MSRVTTDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )

    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_testset)
    except:
        test_sampler = None  # cpu
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


def dataloader_msrvtt_train_test(args, tokenizer):
    from .dataloader_msrvtt_retrieval import MSRVTTDataset

    msrvtt_dataset = MSRVTTDataset(
        subset='train_test',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    except:
        train_sampler = None  # cpu
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msvd_train(args, tokenizer):
    from .dataloader_msvd_retrieval import MSVDDataset

    msvd_dataset = MSVDDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    except:
        train_sampler = None
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msvd_dataset), train_sampler


def dataloader_msvd_test(args, tokenizer, subset="test"):
    from .dataloader_msvd_retrieval import MSVDDataset

    msvd_dataset = MSVDDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    except:
        test_sampler = None
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader, len(msvd_dataset)


def dataloader_msvd_train_test(args, tokenizer):
    from .dataloader_msvd_retrieval import MSVDDataset

    msvd_dataset = MSVDDataset(
        subset='train_test',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)
    except:
        train_sampler = None
    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msvd_dataset), train_sampler


def dataloader_lsmdc_train(args, tokenizer):
    from .dataloader_lsmdc_retrieval import LsmdcDataset

    lsmdc_dataset = LsmdcDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    except:
        train_sampler = None
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(lsmdc_dataset), train_sampler


def dataloader_lsmdc_test(args, tokenizer, subset="test"):
    from .dataloader_lsmdc_retrieval import LsmdcDataset

    lsmdc_dataset = LsmdcDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    except:
        test_sampler = None
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader, len(lsmdc_dataset)


def dataloader_lsmdc_train_test(args, tokenizer):
    from .dataloader_lsmdc_retrieval import LsmdcDataset

    lsmdc_dataset = LsmdcDataset(
        subset='train_test',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(lsmdc_dataset)
    except:
        train_sampler = None
    dataloader = DataLoader(
        lsmdc_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(lsmdc_dataset), train_sampler


def dataloader_didemo_train(args, tokenizer):
    from .dataloader_didemo_retrieval import DiDeMoDataset

    didemo_dataset = DiDeMoDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    except:
        train_sampler = None
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(didemo_dataset), train_sampler


def dataloader_didemo_test(args, tokenizer, subset="test"):
    from .dataloader_didemo_retrieval import DiDeMoDataset

    didemo_dataset = DiDeMoDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(didemo_dataset)
    except:
        test_sampler = None
    dataloader = DataLoader(
        didemo_dataset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader, len(didemo_dataset)


def dataloader_activity_train(args, tokenizer):
    from .dataloader_activitynet_retrieval import ActivityNetDataset

    activity_dataset = ActivityNetDataset(
        subset='train',
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    except:
        train_sampler = None
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(activity_dataset), train_sampler


def dataloader_activity_test(args, tokenizer, subset="val"):
    from .dataloader_activitynet_retrieval import ActivityNetDataset

    activity_dataset = ActivityNetDataset(
        subset=subset,
        anno_path=args.anno_path,
        video_path=args.video_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        video_framerate=args.video_framerate,
        config=args
    )
    try:
        test_sampler = torch.utils.data.distributed.DistributedSampler(activity_dataset)
    except:
        test_sampler = None
    dataloader = DataLoader(
        activity_dataset,
        batch_size=args.batch_size_val // args.world_size,
        num_workers=args.workers,
        shuffle=False,
        sampler=test_sampler,
        drop_last=False,
    )
    return dataloader, len(activity_dataset)


DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train": dataloader_msrvtt_train,
                             "val": dataloader_msrvtt_test,
                             "test": None,
                             "train_test": dataloader_msrvtt_train_test}
DATALOADER_DICT["msvd"] = {"train": dataloader_msvd_train,
                           "val": dataloader_msvd_test,
                           "test": dataloader_msvd_test,
                           "train_test": dataloader_msvd_train_test}
DATALOADER_DICT["lsmdc"] = {"train": dataloader_lsmdc_train,
                            "val": dataloader_lsmdc_test,
                            "test": dataloader_lsmdc_test,
                            "train_test": dataloader_lsmdc_train_test}
DATALOADER_DICT["didemo"] = {"train": dataloader_didemo_train,
                             "val": dataloader_didemo_test,
                             "test": dataloader_didemo_test,
                             "train_test": None}
DATALOADER_DICT["activity"] = {"train": dataloader_activity_train,
                               "val": dataloader_activity_test,
                               "test": None,
                               "train_test": None}

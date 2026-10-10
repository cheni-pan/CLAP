from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import sys
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
import datetime
from os.path import join, exists

def startup_trace(message):
    if os.environ.get("TEMPME_STARTUP_TRACE", "1") == "1":
        rank = os.environ.get("LOCAL_RANK", "?")
        print("[startup rank {}] {}".format(rank, message), file=sys.stderr, flush=True)

import torch
startup_trace("imported torch")
import torch.nn.functional as F
from tvr.models.tokenization_clip import SimpleTokenizer as ClipTokenizer
startup_trace("imported tokenizer")
from tvr.dataloaders.data_dataloaders import DATALOADER_DICT
startup_trace("imported dataloaders")
from tvr.models.modeling import VTRModel, AllGather
startup_trace("imported model")
from tvr.models.optimization_adamw import AdamW, get_cosine_schedule_with_warmup
startup_trace("imported optimizer")
from tvr.utils.metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from tvr.utils.comm import is_main_process, synchronize
from tvr.utils.logger import setup_logger
from tvr.utils.metric_logger import MetricLogger

from scipy.special import softmax
startup_trace("finished imports")

allgather = AllGather.apply

global logger

def get_args(description='Temporal Token Merging for Efficient Text-Video Retrieval'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_train", type=int, default=0, help="Whether to run training.")
    parser.add_argument("--do_eval", type=int, default=0, help="Whether to run evaluation.")

    parser.add_argument("--datatype", default="msrvtt", type=str, help="Point the dataset to finetune.")
    parser.add_argument('--anno_path', type=str, default='data/MSR-VTT/anns', help='annotation path')
    parser.add_argument('--video_path', type=str, default='data/MSR-VTT/videos', help='video path')
    parser.add_argument('--pretrained_path', type=str, default="your_path", help='pretrained model path')
    parser.add_argument('--eval_split', type=str, default='test', choices=['test', 'val'], help='Split used when --do_eval is enabled')

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--clip_lr', type=float, default=6e-4, help='learning rate')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
    parser.add_argument('--weight_decay', type=float, default=0.2, help='weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=128, help='batch size eval')

    parser.add_argument('--max_words', type=int, default=32, help='max text token number')
    parser.add_argument('--max_frames', type=int, default=12, help='max key frames')
    parser.add_argument('--video_framerate', type=int, default=1, help='framerate to sample video frame')

    parser.add_argument("--device", default='cuda', type=str, help="cpu/cuda")
    parser.add_argument("--world_size", default=1, type=int, help="distribted training")
    parser.add_argument(
        "--local-rank",
        "--local_rank",
        dest="local_rank",
        default=int(os.environ.get("LOCAL_RANK", 0)),
        type=int,
        help="distribted training",
    )
    parser.add_argument("--distributed", default=0, type=int, help="multi machine DDP")
    parser.add_argument(
        "--ddp_mode",
        default=os.environ.get("DDP_MODE", "manual"),
        choices=["manual", "ddp"],
        help="manual synchronizes trainable gradients without DDP's full-parameter init broadcast",
    )
    parser.add_argument(
        "--dist_backend",
        default=os.environ.get("DIST_BACKEND", "gloo"),
        choices=["gloo", "nccl"],
        help="distributed communication backend",
    )
    parser.add_argument(
        "--detect_anomaly",
        default=int(os.environ.get("DETECT_ANOMALY", "0")),
        type=int,
        help="Enable torch autograd anomaly detection. It is memory-heavy and should stay off for training.",
    )
    parser.add_argument(
        "--use_amp",
        default=int(os.environ.get("USE_AMP", "1")),
        type=int,
        help="Use CUDA autocast to reduce activation memory.",
    )
    parser.add_argument(
        "--amp_dtype",
        default=os.environ.get("AMP_DTYPE", "bf16"),
        choices=["bf16", "fp16"],
        help="Autocast dtype when --use_amp is enabled.",
    )

    parser.add_argument('--n_display', type=int, default=50, help='Information display frequence')
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--base_encoder", default="ViT-B/32", type=str, help="Choose a CLIP version")

    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    
    parser.add_argument('--lora_dim', type=int, default=8)

    parser.add_argument('--tome_r', type=int, default=2)
    parser.add_argument('--tome_tracesource', type=bool, default=False)
    parser.add_argument('--tome_propattn', type=bool, default=True)

    parser.add_argument('--use_claa', type=int, default=1, help='Enable CLAA cross-layer attention aggregation')
    parser.add_argument('--use_sap', type=int, default=1, help='Enable SAP semantic attention prior refinement')
    parser.add_argument('--resclip_window', type=int, default=4, help='Number of consecutive visual blocks reserved for CLAA/SAP adaptation')
    parser.add_argument('--claa_tau', type=float, default=2.0, help='Scaling factor for CLAA fusion')
    parser.add_argument('--claa_lambda', type=float, default=0.6, help='Blend ratio for CLAA fusion')
    parser.add_argument('--sap_tau', type=float, default=2.1, help='Scaling factor for SAP fusion')
    parser.add_argument('--sap_lambda', type=float, default=0.6, help='Blend ratio for SAP fusion')
    parser.add_argument('--sap_temp_thd', type=float, default=0.0, help='Threshold for filtering weak token affinities when building SAP')
    parser.add_argument('--use_pamr', type=int, default=0, help='Enable PAMR refinement inside the adapted SAP path')
    parser.add_argument('--pamr_steps', type=int, default=10, help='Number of PAMR refinement iterations')
    parser.add_argument('--pamr_stride', type=str, default='8-16', help='Dash-separated PAMR dilation list, e.g. 8-16')
    parser.add_argument('--use_clip_refiner', type=int, default=0, help='Enable CLIPRefiner-style dense spatial refinement')
    parser.add_argument('--clip_refiner_depth', type=int, default=2, help='Number of lightweight refiner blocks')
    parser.add_argument('--clip_refiner_hidden_ratio', type=float, default=0.25, help='Hidden-width ratio inside the refiner MLP')
    parser.add_argument('--clip_refiner_residual_scale', type=float, default=0.25, help='Residual scale for refined dense patch tokens')
    parser.add_argument('--clip_refiner_temperature', type=float, default=0.07, help='Temperature for refiner spatial affinity prior')
    parser.add_argument('--clip_refiner_lambda', type=float, default=0.25, help='Blend ratio for refiner prior inside SAP attention')
    parser.add_argument('--clip_refiner_scd_weight', type=float, default=0.05, help='Weight for SCD-style spatial correlation distillation loss')

    ### 12--9-->6--10-->3--11-->1
    parser.add_argument('--merge_layer', type=str, default='8-9-10') # start from 0
    parser.add_argument('--merge_frame_num', type=str, default='2-2-3')
    ### R_c = 100% - 30% = 70%; R_I = 100% - 10% = 90%
    parser.add_argument('--merge_token_proportion', type=str, default='30-10')
    parser.add_argument('--frame_pos', type=int, default=1)
    
    args = parser.parse_args()
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    return args


def set_seed_logger(args):
    global logger
    startup_trace("set_seed_logger start")
    args.world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else args.world_size
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    startup_trace("cpu seeds set")

    if torch.cuda.is_available():
        startup_trace("cuda is available")
        visible_cuda_devices = torch.cuda.device_count()
        startup_trace("visible cuda devices: {}".format(visible_cuda_devices))
        if args.local_rank < 0 or args.local_rank >= visible_cuda_devices:
            raise RuntimeError(
                "Invalid local_rank {} with only {} visible CUDA device(s). "
                "Check CUDA_VISIBLE_DEVICES and --nproc_per_node/NPROC_PER_NODE.".format(
                    args.local_rank, visible_cuda_devices
                )
            )
        startup_trace("set_device({}) start".format(args.local_rank))
        torch.cuda.set_device(args.local_rank)
        startup_trace("set_device({}) done".format(args.local_rank))
        torch.cuda.manual_seed(args.seed)
        startup_trace("cuda seed set")
        if args.world_size > 1:
            startup_trace("init_process_group start")
            dist_timeout_seconds = int(os.environ.get("DIST_TIMEOUT_SECONDS", "300"))
            torch.distributed.init_process_group(
                backend=args.dist_backend,
                timeout=datetime.timedelta(seconds=dist_timeout_seconds),
            )
            startup_trace("init_process_group({}) done".format(args.dist_backend))
        args.device = torch.device("cuda", args.local_rank)
    logger.info("local_rank: {} world_size: {}".format(args.local_rank, args.world_size))

    if args.batch_size % args.world_size != 0 or args.batch_size_val % args.world_size != 0:
        raise ValueError(
            "Invalid batch_size/batch_size_val and world_size parameter: {}%{} and {}%{}, should be == 0".format(
                args.batch_size, args.world_size, args.batch_size_val, args.world_size))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args


def build_model(args):
    model = VTRModel(args)
    if args.init_model:
        if not exists(args.init_model):
            raise FileNotFoundError
        model_state_dict = torch.load(args.init_model, map_location='cpu')
        load_result = model.load_state_dict(model_state_dict, strict=False)
        if getattr(load_result, "missing_keys", None) or getattr(load_result, "unexpected_keys", None):
            logger.warning(
                "Checkpoint load mismatch: missing_keys=%s unexpected_keys=%s",
                load_result.missing_keys,
                load_result.unexpected_keys,
            )

    model.to(args.device)
    return model


def build_dataloader(args):
    ## ####################################
    # dataloader loading
    ## ####################################
    tokenizer = ClipTokenizer()
    assert args.datatype in DATALOADER_DICT

    assert DATALOADER_DICT[args.datatype]["test"] is not None or DATALOADER_DICT[args.datatype]["val"] is not None

    test_dataloader, test_length = None, 0
    if DATALOADER_DICT[args.datatype]["test"] is not None:
        test_subset = args.eval_split if not args.do_train else "test"
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer, subset=test_subset)

    if DATALOADER_DICT[args.datatype]["val"] is not None:
        val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, subset="val")
    else:
        val_dataloader, val_length = test_dataloader, test_length

    ## report validation results if the ["test"] is None
    if test_dataloader is None:
        test_dataloader, test_length = val_dataloader, val_length

    if isinstance(test_length, int):
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %d", val_length)
    elif len(test_length) == 2:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %dv %dt", test_length[0], test_length[1])
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d %d", len(test_dataloader[0]), len(test_dataloader[1]))
        logger.info("***** Running val *****")
        logger.info("  Num examples = %dv %dt", val_length[0], val_length[1])

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", len(train_dataloader) * args.epochs)
    else:
        train_dataloader, train_sampler = None, None

    return test_dataloader, val_dataloader, train_dataloader, train_sampler


def prep_optimizer(args, model, num_train_optimization_steps, local_rank):
    if hasattr(model, 'module'):
        model = model.module
    clip_lr = args.clip_lr  # 1e-7
    weight_decay = args.weight_decay  # 0.2
    warmup_proportion = args.warmup_proportion
    param_optimizer = list(model.named_parameters())

    for name, param in param_optimizer:
        if "TVPt" in name or "clip_refiner" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    
    optimizer_parameters_prompt = []
    enabled_prompt = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            enabled_prompt.append(name)
            optimizer_parameters_prompt.append(param)
    logger.info(f"Tuned Parameters: {sorted(enabled_prompt)}")

    optimizer_grouped_params = [
        {'params': optimizer_parameters_prompt, 'lr': args.clip_lr}
    ]

    optimizer = AdamW(optimizer_grouped_params, weight_decay=args.weight_decay)
    num_warmup_steps = int(warmup_proportion * num_train_optimization_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    if torch.cuda.is_available() and args.world_size > 1 and args.ddp_mode == "ddp":
        startup_trace("DistributedDataParallel wrap start")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                          find_unused_parameters=True)
        startup_trace("DistributedDataParallel wrap done")
    elif torch.cuda.is_available() and args.world_size > 1:
        logger.info("Using manual distributed gradient synchronization; skipping DDP parameter broadcast.")

    return optimizer, scheduler, model


def sync_trainable_gradients(args, model):
    if args.world_size < 2 or args.ddp_mode != "manual":
        return

    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        backend = torch.distributed.get_backend().lower()
        if backend == "gloo" and param.grad.is_cuda:
            grad = param.grad.detach().cpu()
            torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
            param.grad.copy_(grad.to(param.grad.device))
        else:
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)
        param.grad.div_(args.world_size)


def save_model(epoch, args, model, type_name=""):
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "{}.pth".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def prompt_save_model(epoch, args, model, type_name=""):
    assert "Not Implement" == 0
    # Only save the model it-self
    model_to_save = model.module if hasattr(model, 'module') else model
    output_model_file = join(args.output_dir, "{}.pth".format(type_name))
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    return output_model_file

def reduce_loss(loss, args):
    world_size = args.world_size
    if world_size < 2:
        return loss
    with torch.no_grad():
        reduced_loss = loss.detach().clone()
        backend = torch.distributed.get_backend().lower()
        if backend == "gloo" and reduced_loss.is_cuda:
            reduced_loss = reduced_loss.cpu()
        torch.distributed.all_reduce(reduced_loss, op=torch.distributed.ReduceOp.SUM)
        reduced_loss /= world_size
    return reduced_loss


def build_eval_score(tv_metrics, vt_metrics):
    tv_rsum = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10']
    vt_rsum = vt_metrics['R1'] + vt_metrics['R5'] + vt_metrics['R10']
    return {
        "r1": tv_metrics['R1'],
        "tv_rsum": tv_rsum,
        "vt_rsum": vt_rsum,
        "balance_rsum": tv_rsum + vt_rsum,
    }


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                scheduler, global_step, max_steps, val_dataloader):
    global logger
    global best_score
    global best_score_list
    global best_balance_score
    global best_balance_score_list
    global meters
    global sim_matrix_num
    global sim_name_list

    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    total_loss = 0

    end = time.time()
    logit_scale = 0
    if is_main_process():
        logger.info("[train] epoch %d/%d start, waiting for first batch", epoch, args.epochs)
    for step, batch in enumerate(train_dataloader, start=1):
        global_step += 1
        data_time = time.time() - end
        if step == 1 and is_main_process():
            logger.info("[train] first batch loaded in %.2fs", data_time)

        batch = tuple(
            t.to(device=device, non_blocking=True) if hasattr(t, "to") else t
            for t in batch
        )

        text_ids, text_mask, video, video_mask, inds, idx = batch
        optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(args.use_amp and torch.cuda.is_available())
        amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
            loss = model(text_ids, text_mask, video, video_mask, idx, global_step)
        
        if n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.

        if args.detect_anomaly:
            with torch.autograd.detect_anomaly():
                loss.backward()
        else:
            loss.backward()

        sync_trainable_gradients(args, model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        if scheduler is not None:
            scheduler.step()  # Update learning rate schedule

        if hasattr(model, 'module'):
            torch.clamp_(model.module.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.module.clip.logit_scale.exp().item()
        else:
            torch.clamp_(model.clip.logit_scale.data, max=np.log(100))
            logit_scale = model.clip.logit_scale.exp().item()

        batch_time = time.time() - end
        end = time.time()

        reduced_l = reduce_loss(loss, args)
        meters.update(time=batch_time, data=data_time, loss=float(reduced_l))

        eta_seconds = meters.time.global_avg * (max_steps - global_step)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if (global_step % log_step == 0 or global_step == 1) and is_main_process():
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "epoch: {epoch}/{max_epoch}",
                        "iteration: {iteration}/{max_iteration}",
                        "{meters}",
                        "lr: {lr}",
                        "logit_scale: {logit_scale:.2f}"
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    epoch=epoch,
                    max_epoch=args.epochs,
                    iteration=global_step,
                    max_iteration=max_steps,
                    meters=str(meters),
                    lr="/".join([str('%.9f' % itm) for itm in sorted(list(set(scheduler.get_last_lr())))]),
                    logit_scale=logit_scale,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )

        if (global_step % (log_step * 3) == 0)  or global_step == 1:
            eval_scores = eval_epoch(args, model, val_dataloader, args.device)
            if args.local_rank == 0:
                for list_idx in range(sim_matrix_num):
                    r1_score = eval_scores[list_idx]["r1"]
                    balance_score = eval_scores[list_idx]["balance_rsum"]
                    if best_score_list[list_idx] < r1_score:
                        best_score_list[list_idx] = r1_score
                    if best_balance_score_list[list_idx] < balance_score:
                        best_balance_score_list[list_idx] = balance_score
                    logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(r1_score, best_score_list[list_idx],sim_name_list[list_idx]))
                    logger.info("The balance R@sum is: {:.4f}\t| {:.4f}\tin {}".format(
                        balance_score, best_balance_score_list[list_idx], sim_name_list[list_idx]))

                current_best_r1 = max(score["r1"] for score in eval_scores)
                if best_score < current_best_r1:
                    best_score = current_best_r1
                    output_model_file = save_model(epoch, args, model, type_name="best")
                current_best_balance = max(score["balance_rsum"] for score in eval_scores)
                if best_balance_score < current_best_balance:
                    best_balance_score = current_best_balance
                    output_model_file = save_model(epoch, args, model, type_name="balance")
                logger.info("The best R1 is: {:.4f} at all".format(best_score))
                logger.info("The best balance R@sum is: {:.4f} at all".format(best_balance_score))

            synchronize()
            model.train()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def eval_epoch(args, model, test_dataloader, device):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)

    multi_sentence = False
    cut_off_points = []
    if hasattr(test_dataloader.dataset, 'multi_sentence_per_video') and test_dataloader.dataset.multi_sentence_per_video:
        multi_sentence = True
        cut_off_points = [itm - 1 for itm in test_dataloader.dataset.cut_off_points]
        logger.warning("Eval under the multi-sentence per video setting.")
        logger.warning("sentence num: {}, video num: {}".format(
            test_dataloader.dataset.sentence_num, test_dataloader.dataset.video_num))

    model.eval()
    # ----------------------------
    # 1. cache the features
    # ----------------------------
    batch_cls, batch_mask_t = [], []
    batch_video_feat, batch_mask_v = [], []
    batch_ids = []

    with torch.no_grad():
        tic = time.time()

        sim_matrix = []

        logger.info('[start] extract')
        eval_log_interval = int(os.environ.get("EVAL_LOG_INTERVAL", "50"))
        eval_total_steps = len(test_dataloader)
        for eval_step, batch in enumerate(tqdm(test_dataloader), start=1):
            batch = tuple(t.to(device) for t in batch)
            text_ids, text_mask, video, video_mask, inds, _ = batch
            cls, video_feat = model.stage1_eval(text_ids, text_mask, video, video_mask)
            batch_cls.append(cls)
            batch_mask_t.append(text_mask)
            batch_video_feat.append(video_feat)
            batch_mask_v.append(video_mask)
            batch_ids.append(inds)
            if (
                eval_log_interval > 0
                and (eval_step % eval_log_interval == 0 or eval_step == eval_total_steps)
                and is_main_process()
            ):
                elapsed = time.time() - tic
                logger.info(
                    "[extract] %d/%d batches, elapsed %s",
                    eval_step,
                    eval_total_steps,
                    str(datetime.timedelta(seconds=int(elapsed))),
                )

        if args.world_size > 1:
            synchronize()
        
        batch_ids = allgather(torch.cat(batch_ids, dim=0), args).squeeze()
        
        batch_cls = allgather(torch.cat(batch_cls, dim=0), args)
        batch_mask_t = allgather(torch.cat(batch_mask_t, dim=0), args)
        batch_video_feat = allgather(torch.cat(batch_video_feat, dim=0), args)
        batch_mask_v = allgather(torch.cat(batch_mask_v, dim=0), args)
        
        batch_cls[batch_ids] = batch_cls.clone()
        batch_mask_t[batch_ids] = batch_mask_t.clone()
        batch_video_feat[batch_ids] = batch_video_feat.clone()
        batch_mask_v[batch_ids] = batch_mask_v.clone()
        
        batch_cls = batch_cls[:batch_ids.max() + 1, ...]
        batch_mask_t = batch_mask_t[:batch_ids.max() + 1, ...]
        batch_video_feat = batch_video_feat[:batch_ids.max() + 1, ...]
        batch_mask_v = batch_mask_v[:batch_ids.max() + 1, ...]
        logger.info('[finish] extract')
        
        logger.info('[start] calculate the similarity')
        with torch.no_grad():
            mini_batch = args.batch_size_val
            sim_matrix = []
            
            batch_cls_split = torch.split(batch_cls, mini_batch)
            batch_mask_t_split = torch.split(batch_mask_t, mini_batch)
            if multi_sentence:
                video_inds = torch.tensor(cut_off_points, dtype=torch.long, device=batch_video_feat.device)
                batch_video_feat_for_eval = batch_video_feat[video_inds]
                batch_mask_v_for_eval = batch_mask_v[video_inds]
            else:
                batch_video_feat_for_eval = batch_video_feat
                batch_mask_v_for_eval = batch_mask_v

            batch_video_feat_split = torch.split(batch_video_feat_for_eval, mini_batch)
            batch_mask_v_split = torch.split(batch_mask_v_for_eval, mini_batch)
            
            for cls, text_mask in tqdm(zip(batch_cls_split, batch_mask_t_split)):
                each_row = []
                for video_feat, video_mask in zip(batch_video_feat_split, batch_mask_v_split):
                    logits = model.stage2_eval(cls, text_mask, video_feat, video_mask)
                    logits = logits.cpu().detach().numpy()
                    each_row.append(logits)
                each_row = np.concatenate(tuple(each_row), axis=-1)
                sim_matrix.append(each_row)
            sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        logger.info('[finish] calculate the similarity')
        
        
    logger.info('[start] compute_metrics')
    logger.info("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1])) 
    global sim_name_list
    
    eval_scores = []
    list_idx = 0
    if multi_sentence:
        logger.info("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
        cut_off_points2len = [itm + 1 for itm in cut_off_points]
        max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len[:-1], cut_off_points2len)])
        sim_matrix_new = []
        for s_, e_ in zip([0] + cut_off_points2len[:-1], cut_off_points2len):
            sim_matrix_new.append(np.concatenate((
                sim_matrix[s_:e_],
                np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)
            ), axis=0))
        sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
        logger.info("after reshape, sim matrix size: {} x {} x {}".format(
            sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))
        tv_metrics = tensor_text_to_video_metrics(sim_matrix)
        vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
    else:
        tv_metrics = compute_metrics(sim_matrix)
        vt_metrics = compute_metrics(sim_matrix.T)
    logger.info("Eval {} ...".format(sim_name_list[list_idx]))
    eval_score = build_eval_score(tv_metrics, vt_metrics)
    logger.info("Text-to-Video: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@sum: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], eval_score["tv_rsum"], tv_metrics['R50'], tv_metrics['MR'], tv_metrics['MeanR']))
    logger.info("Video-to-Text: R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - R@sum: {:.1f} - R@50: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}".
                format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], eval_score["vt_rsum"], vt_metrics['R50'], vt_metrics['MR'], vt_metrics['MeanR']))
    logger.info("Balance R@sum: {:.1f} = Text-to-Video R@sum {:.1f} + Video-to-Text R@sum {:.1f}".
                format(eval_score["balance_rsum"], eval_score["tv_rsum"], eval_score["vt_rsum"]))
    eval_scores.append(eval_score)

    return eval_scores

def main():
    global logger
    global best_score
    global best_score_list
    global best_balance_score
    global best_balance_score_list
    global meters
    global sim_matrix_num
    global sim_name_list

    sim_name_list = ['base'] 
    sim_matrix_num = len(sim_name_list)

    meters = MetricLogger(delimiter="  ")
    args = get_args()
    if not exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger('tvr', args.output_dir, args.local_rank)

    args = set_seed_logger(args)

    model = build_model(args)

    test_dataloader, val_dataloader, train_dataloader, train_sampler = build_dataloader(args)
    ## ####################################
    # train and eval
    ## ####################################
    if args.do_train:
        tic = time.time()
        max_steps = len(train_dataloader) * args.epochs
        _max_steps = len(train_dataloader) * args.epochs
        optimizer, scheduler, model = prep_optimizer(args, model, _max_steps, args.local_rank)

        best_score = 0.00001
        best_score_list = [0.00001 for _ in range(sim_matrix_num)]
        best_balance_score = 0.00001
        best_balance_score_list = [0.00001 for _ in range(sim_matrix_num)]
        best_output_model_file = "None"
        global_step = 0
        for epoch in range(args.epochs):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            synchronize()
            torch.cuda.empty_cache()
            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader,
                                               args.device, args.world_size, optimizer,
                                               scheduler, global_step, max_steps, val_dataloader)
            torch.cuda.empty_cache()

            eval_scores = eval_epoch(args, model, val_dataloader, args.device)
            torch.cuda.empty_cache()
            synchronize()

            if args.local_rank == 0:
                for list_idx in range(sim_matrix_num):
                    r1_score = eval_scores[list_idx]["r1"]
                    balance_score = eval_scores[list_idx]["balance_rsum"]
                    if best_score_list[list_idx] < r1_score:
                        best_score_list[list_idx] = r1_score
                    if best_balance_score_list[list_idx] < balance_score:
                        best_balance_score_list[list_idx] = balance_score
                    logger.info("The R1 is: {:.4f}\t| {:.4f}\tin {}".format(r1_score, best_score_list[list_idx],sim_name_list[list_idx]))
                    logger.info("The balance R@sum is: {:.4f}\t| {:.4f}\tin {}".format(
                        balance_score, best_balance_score_list[list_idx], sim_name_list[list_idx]))

                current_best_r1 = max(score["r1"] for score in eval_scores)
                if best_score < current_best_r1:
                    best_score = current_best_r1
                    output_model_file = save_model(epoch, args, model, type_name="best")
                current_best_balance = max(score["balance_rsum"] for score in eval_scores)
                if best_balance_score < current_best_balance:
                    best_balance_score = current_best_balance
                    output_model_file = save_model(epoch, args, model, type_name="balance")
                logger.info("The best R1 is: {:.4f} at all".format(best_score))
                logger.info("The best balance R@sum is: {:.4f} at all".format(best_balance_score))

            synchronize()

        toc = time.time() - tic
        training_time = time.strftime("%Hh %Mmin %Ss", time.gmtime(toc))
        logger.info("*" * 20 + '\n' + f'training finished with {training_time}' + "*" * 20 + '\n')

        if args.local_rank == 0:
            with open("{}_{}.txt".format(args.output_dir, best_score),'w') as f:
                f.write(' ')

    elif args.do_eval:
        eval_epoch(args, model, test_dataloader, args.device)


if __name__ == "__main__":
    main()

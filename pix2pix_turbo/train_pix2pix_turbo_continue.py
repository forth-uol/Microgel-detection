import argparse
import gc
import os
import re

import clip
import diffusers
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import wandb

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from torchvision import transforms
from tqdm.auto import tqdm

from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import PairedDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue pix2pix-turbo training from a generator checkpoint."
    )

    # Model and checkpoint.
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/sd-turbo",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        required=True,
        help="Path to model_STEP.pkl.",
    )

    # Dataset.
    parser.add_argument("--dataset_folder", type=str, required=True)
    parser.add_argument(
        "--train_image_prep",
        type=str,
        default="resized_crop_512",
    )
    parser.add_argument(
        "--test_image_prep",
        type=str,
        default="resized_crop_512",
    )
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    # Output and logging.
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_pix2pix_turbo",
    )

    # Training length.
    #
    # In this continuation script, max_train_steps means the number of
    # ADDITIONAL steps, not the final global step.
    parser.add_argument("--num_training_epochs", type=int, default=11)
    parser.add_argument("--max_train_steps", type=int, required=True)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # Evaluation and saving.
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--eval_freq", type=int, default=100)
    parser.add_argument("--num_samples_eval", type=int, default=100)
    parser.add_argument("--viz_freq", type=int, default=100)

    # Losses.
    parser.add_argument("--gan_disc_type", type=str, default="vagan_clip")
    parser.add_argument(
        "--gan_loss_type",
        type=str,
        default="multilevel_sigmoid_s",
    )
    parser.add_argument("--lambda_gan", type=float, default=0.5)
    parser.add_argument("--lambda_lpips", type=float, default=5.0)
    parser.add_argument("--lambda_l2", type=float, default=1.0)
    parser.add_argument("--lambda_clipsim", type=float, default=5.0)

    # Optimizer.
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--set_grads_to_none", action="store_true")

    # Learning-rate scheduler.
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)

    # Memory and performance.
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
    )
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def get_checkpoint_step(checkpoint_path):
    checkpoint_name = os.path.basename(checkpoint_path)
    match = re.fullmatch(r"model_(\d+)\.pkl", checkpoint_name)

    if match is None:
        raise ValueError(
            "Checkpoint filename must use the format model_STEP.pkl. "
            f"Received: {checkpoint_name}"
        )

    return int(match.group(1))


def load_partial_state(module, saved_state, module_name):
    current_state = module.state_dict()

    unexpected_keys = [
        key for key in saved_state
        if key not in current_state
    ]

    if unexpected_keys:
        preview = "\n".join(unexpected_keys[:20])
        raise KeyError(
            f"Checkpoint contains keys not present in {module_name}:\n"
            f"{preview}"
        )

    shape_mismatches = []

    for key, value in saved_state.items():
        if current_state[key].shape != value.shape:
            shape_mismatches.append(
                f"{key}: model={tuple(current_state[key].shape)}, "
                f"checkpoint={tuple(value.shape)}"
            )

    if shape_mismatches:
        preview = "\n".join(shape_mismatches[:20])
        raise ValueError(
            f"Checkpoint shape mismatch in {module_name}:\n{preview}"
        )

    current_state.update(saved_state)
    module.load_state_dict(current_state, strict=True)


def load_generator(checkpoint_path):
    print(f"Loading generator checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    required_keys = {
        "rank_unet",
        "rank_vae",
        "state_dict_unet",
        "state_dict_vae",
    }

    missing_keys = required_keys.difference(checkpoint.keys())

    if missing_keys:
        raise KeyError(
            "Generator checkpoint is missing required entries: "
            + ", ".join(sorted(missing_keys))
        )

    rank_unet = int(checkpoint["rank_unet"])
    rank_vae = int(checkpoint["rank_vae"])

    print(f"Checkpoint UNet LoRA rank: {rank_unet}")
    print(f"Checkpoint VAE LoRA rank:  {rank_vae}")

    # Construct the same architecture first, then load the saved trainable
    # generator weights. This avoids modifying pix2pix_turbo.py.
    net_pix2pix = Pix2Pix_Turbo(
        lora_rank_unet=rank_unet,
        lora_rank_vae=rank_vae,
    )

    load_partial_state(
        net_pix2pix.unet,
        checkpoint["state_dict_unet"],
        "UNet",
    )

    load_partial_state(
        net_pix2pix.vae,
        checkpoint["state_dict_vae"],
        "VAE",
    )

    net_pix2pix.set_train()

    print("Generator checkpoint loaded successfully")

    return net_pix2pix


def main(args):
    if args.pretrained_model_name_or_path != "stabilityai/sd-turbo":
        raise ValueError(
            "This continuation script currently supports "
            "stabilityai/sd-turbo only."
        )

    if not os.path.isfile(args.resume_from_checkpoint):
        raise FileNotFoundError(
            f"Checkpoint does not exist: {args.resume_from_checkpoint}"
        )

    if args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be greater than zero.")

    initial_global_step = get_checkpoint_step(
        args.resume_from_checkpoint
    )
    expected_final_step = (
        initial_global_step + args.max_train_steps
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(
            os.path.join(args.output_dir, "checkpoints"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(args.output_dir, "eval"),
            exist_ok=True,
        )

    net_pix2pix = load_generator(
        args.resume_from_checkpoint
    )

    if args.enable_xformers_memory_efficient_attention:
        if not is_xformers_available():
            raise ValueError(
                "xformers is not installed. Install it before using "
                "--enable_xformers_memory_efficient_attention."
            )

        net_pix2pix.unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.gan_disc_type == "vagan_clip":
        import vision_aided_loss

        net_disc = vision_aided_loss.Discriminator(
            cv_type="clip",
            loss_type=args.gan_loss_type,
            device="cuda",
        )
    else:
        raise NotImplementedError(
            f"Discriminator type {args.gan_disc_type} "
            "is not implemented."
        )

    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net="vgg").cuda()
    net_lpips.requires_grad_(False)
    net_lpips.eval()

    net_clip, _ = clip.load(
        "ViT-B/32",
        device="cuda",
    )
    net_clip.requires_grad_(False)
    net_clip.eval()

    # Keep exactly the same generator parameters as the original trainer.
    layers_to_opt = []

    for name, parameter in net_pix2pix.unet.named_parameters():
        if "lora" in name:
            assert parameter.requires_grad
            layers_to_opt.append(parameter)

    layers_to_opt += list(
        net_pix2pix.unet.conv_in.parameters()
    )

    for name, parameter in net_pix2pix.vae.named_parameters():
        if "lora" in name and "vae_skip" in name:
            assert parameter.requires_grad
            layers_to_opt.append(parameter)

    layers_to_opt += list(
        net_pix2pix.vae.decoder.skip_conv_1.parameters()
    )
    layers_to_opt += list(
        net_pix2pix.vae.decoder.skip_conv_2.parameters()
    )
    layers_to_opt += list(
        net_pix2pix.vae.decoder.skip_conv_3.parameters()
    )
    layers_to_opt += list(
        net_pix2pix.vae.decoder.skip_conv_4.parameters()
    )

    optimizer = torch.optim.AdamW(
        layers_to_opt,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    optimizer_disc = torch.optim.AdamW(
        net_disc.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=(
            args.lr_warmup_steps * accelerator.num_processes
        ),
        num_training_steps=(
            args.max_train_steps * accelerator.num_processes
        ),
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_disc,
        num_warmup_steps=(
            args.lr_warmup_steps * accelerator.num_processes
        ),
        num_training_steps=(
            args.max_train_steps * accelerator.num_processes
        ),
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    dataset_train = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_image_prep,
        split="train",
        tokenizer=net_pix2pix.tokenizer,
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    dataset_val = PairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.test_image_prep,
        split="test",
        tokenizer=net_pix2pix.tokenizer,
    )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    (
        net_pix2pix,
        net_disc,
        optimizer,
        optimizer_disc,
        dataloader_train,
        lr_scheduler,
        lr_scheduler_disc,
    ) = accelerator.prepare(
        net_pix2pix,
        net_disc,
        optimizer,
        optimizer_disc,
        dataloader_train,
        lr_scheduler,
        lr_scheduler_disc,
    )

    net_clip, net_lpips = accelerator.prepare(
        net_clip,
        net_lpips,
    )

    clip_normalize = transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    weight_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net_pix2pix.to(dtype=weight_dtype)
    net_disc.to(dtype=weight_dtype)
    net_lpips.to(dtype=weight_dtype)
    net_clip.to(dtype=weight_dtype)

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        tracker_config["initial_global_step"] = initial_global_step
        tracker_config["expected_final_step"] = expected_final_step

        accelerator.init_trackers(
            args.tracker_project_name,
            config=tracker_config,
        )

    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    global_step = initial_global_step
    steps_this_run = 0

    progress_bar = tqdm(
        total=args.max_train_steps,
        initial=0,
        desc="Additional steps",
        disable=not accelerator.is_local_main_process,
    )

    accelerator.print(
        f"Initial global step: {initial_global_step}"
    )
    accelerator.print(
        f"Additional steps:    {args.max_train_steps}"
    )
    accelerator.print(
        f"Expected final step: {expected_final_step}"
    )

    for epoch in range(args.num_training_epochs):
        for _, batch in enumerate(dataloader_train):
            if steps_this_run >= args.max_train_steps:
                break

            accumulation_models = [
                net_pix2pix,
                net_disc,
            ]

            with accelerator.accumulate(*accumulation_models):
                source_image = batch[
                    "conditioning_pixel_values"
                ]
                target_image = batch[
                    "output_pixel_values"
                ]

                batch_size = source_image.shape[0]

                # ---------------------------------------------------------
                # Generator reconstruction loss
                # ---------------------------------------------------------

                predicted_image = net_pix2pix(
                    source_image,
                    prompt_tokens=batch["input_ids"],
                    deterministic=True,
                )

                loss_l2 = (
                    F.mse_loss(
                        predicted_image.float(),
                        target_image.float(),
                        reduction="mean",
                    )
                    * args.lambda_l2
                )

                loss_lpips = (
                    net_lpips(
                        predicted_image.float(),
                        target_image.float(),
                    ).mean()
                    * args.lambda_lpips
                )

                loss = loss_l2 + loss_lpips

                if args.lambda_clipsim > 0:
                    predicted_clip = clip_normalize(
                        predicted_image * 0.5 + 0.5
                    )

                    predicted_clip = F.interpolate(
                        predicted_clip,
                        (224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )

                    caption_tokens = clip.tokenize(
                        batch["caption"],
                        truncate=True,
                    ).to(predicted_image.device)

                    clip_similarity, _ = net_clip(
                        predicted_clip,
                        caption_tokens,
                    )

                    loss_clipsim = (
                        1 - clip_similarity.mean() / 100
                    )

                    loss += (
                        loss_clipsim
                        * args.lambda_clipsim
                    )

                accelerator.backward(
                    loss,
                    retain_graph=False,
                )

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt,
                        args.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(
                    set_to_none=args.set_grads_to_none
                )

                # ---------------------------------------------------------
                # Generator adversarial loss
                # ---------------------------------------------------------

                predicted_image = net_pix2pix(
                    source_image,
                    prompt_tokens=batch["input_ids"],
                    deterministic=True,
                )

                loss_generator = (
                    net_disc(
                        predicted_image,
                        for_G=True,
                    ).mean()
                    * args.lambda_gan
                )

                accelerator.backward(loss_generator)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        layers_to_opt,
                        args.max_grad_norm,
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(
                    set_to_none=args.set_grads_to_none
                )

                # ---------------------------------------------------------
                # Discriminator real loss
                # ---------------------------------------------------------

                loss_discriminator_real = (
                    net_disc(
                        target_image.detach(),
                        for_real=True,
                    ).mean()
                    * args.lambda_gan
                )

                accelerator.backward(
                    loss_discriminator_real.mean()
                )

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        net_disc.parameters(),
                        args.max_grad_norm,
                    )

                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(
                    set_to_none=args.set_grads_to_none
                )

                # ---------------------------------------------------------
                # Discriminator fake loss
                # ---------------------------------------------------------

                loss_discriminator_fake = (
                    net_disc(
                        predicted_image.detach(),
                        for_real=False,
                    ).mean()
                    * args.lambda_gan
                )

                accelerator.backward(
                    loss_discriminator_fake.mean()
                )

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        net_disc.parameters(),
                        args.max_grad_norm,
                    )

                optimizer_disc.step()
                optimizer_disc.zero_grad(
                    set_to_none=args.set_grads_to_none
                )

                loss_discriminator = (
                    loss_discriminator_real
                    + loss_discriminator_fake
                )

            if accelerator.sync_gradients:
                global_step += 1
                steps_this_run += 1
                progress_bar.update(1)

                if accelerator.is_main_process:
                    logs = {
                        "lossG": loss_generator.detach().item(),
                        "lossD": loss_discriminator.detach().item(),
                        "loss_l2": loss_l2.detach().item(),
                        "loss_lpips": loss_lpips.detach().item(),
                        "additional_step": steps_this_run,
                    }

                    if args.lambda_clipsim > 0:
                        logs["loss_clipsim"] = (
                            loss_clipsim.detach().item()
                        )

                    progress_bar.set_postfix(
                        lossG=logs["lossG"],
                        lossD=logs["lossD"],
                        loss_l2=logs["loss_l2"],
                        loss_lpips=logs["loss_lpips"],
                    )

                    if global_step % args.viz_freq == 0:
                        logs["train/source"] = [
                            wandb.Image(
                                source_image[index]
                                .float()
                                .detach()
                                .cpu(),
                                caption=f"idx={index}",
                            )
                            for index in range(batch_size)
                        ]

                        logs["train/target"] = [
                            wandb.Image(
                                target_image[index]
                                .float()
                                .detach()
                                .cpu(),
                                caption=f"idx={index}",
                            )
                            for index in range(batch_size)
                        ]

                        logs["train/model_output"] = [
                            wandb.Image(
                                predicted_image[index]
                                .float()
                                .detach()
                                .cpu(),
                                caption=f"idx={index}",
                            )
                            for index in range(batch_size)
                        ]

                    if (
                        global_step
                        % args.checkpointing_steps
                        == 0
                    ):
                        checkpoint_path = os.path.join(
                            args.output_dir,
                            "checkpoints",
                            f"model_{global_step}.pkl",
                        )

                        accelerator.unwrap_model(
                            net_pix2pix
                        ).save_model(checkpoint_path)

                        print(
                            "Saved generator checkpoint: "
                            f"{checkpoint_path}"
                        )

                    if global_step % args.eval_freq == 0:
                        val_l2_values = []
                        val_lpips_values = []
                        val_clipsim_values = []

                        for val_index, val_batch in enumerate(
                            dataloader_val
                        ):
                            if (
                                val_index
                                >= args.num_samples_eval
                            ):
                                break

                            val_source = val_batch[
                                "conditioning_pixel_values"
                            ].cuda()

                            val_target = val_batch[
                                "output_pixel_values"
                            ].cuda()

                            with torch.no_grad():
                                val_prediction = (
                                    accelerator.unwrap_model(
                                        net_pix2pix
                                    )(
                                        val_source,
                                        prompt_tokens=val_batch[
                                            "input_ids"
                                        ].cuda(),
                                        deterministic=True,
                                    )
                                )

                                val_l2 = F.mse_loss(
                                    val_prediction.float(),
                                    val_target.float(),
                                    reduction="mean",
                                )

                                val_lpips = net_lpips(
                                    val_prediction.float(),
                                    val_target.float(),
                                ).mean()

                                val_prediction_clip = (
                                    clip_normalize(
                                        val_prediction
                                        * 0.5
                                        + 0.5
                                    )
                                )

                                val_prediction_clip = (
                                    F.interpolate(
                                        val_prediction_clip,
                                        (224, 224),
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                                )

                                val_caption_tokens = (
                                    clip.tokenize(
                                        val_batch["caption"],
                                        truncate=True,
                                    ).to(
                                        val_prediction.device
                                    )
                                )

                                val_clipsim, _ = net_clip(
                                    val_prediction_clip,
                                    val_caption_tokens,
                                )

                                val_l2_values.append(
                                    val_l2.item()
                                )
                                val_lpips_values.append(
                                    val_lpips.item()
                                )
                                val_clipsim_values.append(
                                    val_clipsim.mean().item()
                                )

                        logs["val/l2"] = np.mean(
                            val_l2_values
                        )
                        logs["val/lpips"] = np.mean(
                            val_lpips_values
                        )
                        logs["val/clipsim"] = np.mean(
                            val_clipsim_values
                        )

                        gc.collect()
                        torch.cuda.empty_cache()

                    accelerator.log(
                        logs,
                        step=global_step,
                    )

        if steps_this_run >= args.max_train_steps:
            break

    progress_bar.close()
    accelerator.wait_for_everyone()

    if steps_this_run != args.max_train_steps:
        raise RuntimeError(
            "Training ended before completing the requested "
            f"additional steps. Completed {steps_this_run}, "
            f"requested {args.max_train_steps}. Increase "
            "--num_training_epochs."
        )

    final_checkpoint = os.path.join(
        args.output_dir,
        "checkpoints",
        f"model_{global_step}.pkl",
    )

    if accelerator.is_main_process:
        accelerator.unwrap_model(
            net_pix2pix
        ).save_model(final_checkpoint)

        print("Training completed successfully")
        print(
            f"Initial global step: {initial_global_step}"
        )
        print(
            f"Additional steps:    {steps_this_run}"
        )
        print(f"Final global step:   {global_step}")
        print(f"Final checkpoint:    {final_checkpoint}")

    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
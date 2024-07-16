# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers import (
    ControlNetModel,
    DPMSolverMultistepScheduler,
    StableDiffusionLatentUpscalePipeline,
    StableDiffusionControlNetPipeline
)
from utils import (
    get_fp8_config,
    get_int8_config,
    load_calib_prompts,
)

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        # pipe(
        #     prompt=prompts,
        #     num_inference_steps=kwargs["n_steps"],
        #     negative_prompt=[
        #         "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
        #     ]
        #     * len(prompts),
        # ).images
        pipe(
            prompt=prompts,
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
            # output_type="latent",
            image=[torch.load("/root/Product_AIGC/test_control_temp_input/input_image_0.pt"), torch.load("/root/Product_AIGC/test_control_temp_input/input_image_1.pt")],
            generator=torch.manual_seed(11557),
            num_inference_steps=kwargs["n_steps"],
            guidance_scale=7.5,
            # strength=1.0,
            controlnet_conditioning_scale=[1.0, 1.0]).images
        # pipe(
        #     prompt=prompts,
        #     negative_prompt=[
        #         "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
        #     ]
        #     * len(prompts),
        #     image=(torch.load("/root/Product_AIGC/test_sr_temp_input/sr_input_latent.pt"))[0:2],
        #     generator=torch.manual_seed(11557),
        #     num_inference_steps=kwargs["n_steps"],
        #     guidance_scale=0).images
def main():
    parser = argparse.ArgumentParser()
    # Model hyperparameters
    parser.add_argument("--exp_name", default=None)
    parser.add_argument(
        "--model",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        choices=[
            "stabilityai/stable-diffusion-xl-base-1.0",
            "stabilityai/sdxl-turbo",
            "runwayml/stable-diffusion-v1-5",
            "RV_V51_noninpainting",
            "sd-x2-latent-upscaler"
        ],
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=30,
        help="Number of denoising steps, for SDXL-turbo, use 1-4 steps",
    )

    # Calibration and quantization parameters
    parser.add_argument("--format", type=str, default="int8", choices=["int8", "fp8"])
    parser.add_argument("--percentile", type=float, default=1.0, required=False)
    parser.add_argument(
        "--collect-method",
        type=str,
        required=False,
        default="default",
        choices=["global_min", "min-max", "min-mean", "mean-max", "default"],
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--calib-size", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0, help="SmoothQuant Alpha")
    parser.add_argument(
        "--quant-level",
        default=3.0,
        type=float,
        choices=[1.0, 2.0, 2.5, 3.0],
        help="Quantization level, 1: CNN, 2: CNN+FFN, 2.5: CNN+FFN+QKV, 3: CNN+FC",
    )

    args = parser.parse_args()

    args.calib_size = args.calib_size // args.batch_size

    if args.model == "runwayml/stable-diffusion-v1-5":
        pipe = StableDiffusionPipeline.from_pretrained(
            args.model, torch_dtype=torch.float16, safety_checker=None
        )
    elif args.model == "RV_V51_noninpainting":
        controlnets = []
        controlnets.append(ControlNetModel.from_pretrained("/checkpoints/control_v11p_sd15_inpaint", torch_dtype=torch.float16))
        controlnets.append(ControlNetModel.from_pretrained("/checkpoints/huggingface/hub/models--lllyasviel--sd-controlnet-canny/snapshots/7f2f69197050967007f6bbd23ab5e52f0384162a", torch_dtype=torch.float16))
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "/checkpoints/RV_V51_noninpainting",
            controlnet = controlnets,
            safety_checker = None,
            torch_dtype = torch.float16
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )
    elif args.model == "sd-x2-latent-upscaler":
        pipe = StableDiffusionLatentUpscalePipeline.from_pretrained("/checkpoints/sd-x2-latent-upscaler", torch_dtype=torch.float16)
    else:
        pipe = DiffusionPipeline.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    pipe.to("cuda")

    # This is a list of prompts
    cali_prompts = load_calib_prompts(args.batch_size, "./calib/calib_prompts.txt")
    extra_step = (
        1 if args.model == "runwayml/stable-diffusion-v1-5" or \
             args.model == "sd-x2-latent-upscaler" or args.model == "RV_V51_noninpainting" else 0
    )  # Depending on the scheduler. some schedulers will do n+1 steps
    if args.format == "int8":
        # Making sure to use global_min in the calibrator for SD 1.5
        if args.model == "runwayml/stable-diffusion-v1-5" or \
           args.model == "sd-x2-latent-upscaler" or args.model == "RV_V51_noninpainting":
            args.collect_method = "global_min"
        quant_config = get_int8_config(
            # pipe.controlnet,
            pipe.unet_controlnet,
            # pipe.unet,
            args.quant_level,
            args.alpha,
            args.percentile,
            args.n_steps + extra_step,
            collect_method=args.collect_method,
        )
    else:
        if args.collect_method == "default":
            quant_config = mtq.FP8_DEFAULT_CFG
        else:
            quant_config = get_fp8_config(
                pipe.unet,
                args.percentile,
                args.n_steps + extra_step,
                collect_method=args.collect_method,
            )

    def forward_loop(unet):
        # pipe.unet = unet
        # pipe.controlnet = unet
        pipe.unet_controlnet = unet
        do_calibrate(
            pipe=pipe,
            calibration_prompts=cali_prompts,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
        )

    # mtq.quantize(pipe.unet, quant_config, forward_loop)
    # mto.save(pipe.unet, f"./unet.state_dict.{args.exp_name}.pt")
    # mtq.quantize(pipe.controlnet, quant_config, forward_loop)
    # mto.save(pipe.controlnet, f"./unet.state_dict.{args.exp_name}.pt")
    mtq.quantize(pipe.unet_controlnet, quant_config, forward_loop)
    mto.save(pipe.unet_controlnet, f"./unet.state_dict.{args.exp_name}.pt")

if __name__ == "__main__":
    main()

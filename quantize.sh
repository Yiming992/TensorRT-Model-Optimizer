python quantize.py --model sd-x2-latent-upscaler --format int8 \
--batch-size 2 --calib-size 512 --collect-method min-mean \
--percentile 1.0 --alpha 1.0 --quant-level 2 \
--n_steps 10   --exp_name sr_2

python run_export.py --model sd-x2-latent-upscaler \
--quantized-ckpt /root/TensorRT-Model-Optimizer/diffusers/unet.state_dict.sr.pt \
--format int8 --quant-level 2 --onnx-dir sr_unet_int8

python quantize.py --model RV_V51_noninpainting --format int8 \
--batch-size 4 --calib-size 512 --collect-method min-mean \
--percentile 1.0 --alpha 1.0 --quant-level 2 \
--n_steps 20   --exp_name control_net

python quantize.py --model RV_V51_noninpainting --format int8 \
--batch-size 4 --calib-size 512 --collect-method min-mean \
--percentile 1.0 --alpha 1.0 --quant-level 2 \
--n_steps 20   --exp_name control_net_cr

python quantize.py --model RV_V51_noninpainting --format int8 \
--batch-size 4 --calib-size 512 --collect-method min-mean \
--percentile 1.0 --alpha 1.0 --quant-level 1 \
--n_steps 20   --exp_name control_merge

python run_export.py --model RV_V51_noninpainting \
--quantized-ckpt /root/TensorRT-Model-Optimizer/diffusers/unet.state_dict.control_net.pt \
--format int8 --quant-level 2 --onnx-dir controlnet_int8

python run_export.py --model RV_V51_noninpainting \
--quantized-ckpt /root/TensorRT-Model-Optimizer/diffusers/unet.state_dict.control_net_cr.pt \
--format int8 --quant-level 2 --onnx-dir controlnet_int8_cr

python run_export.py --model RV_V51_noninpainting \
--quantized-ckpt /root/TensorRT-Model-Optimizer/diffusers/unet.state_dict.control_merge.pt \
--format int8 --quant-level 1 --onnx-dir controlnet_int8_merge

# 超过2.5量化等级需要用优化一下
python onnx_utils/sdxl_graphsurgeon.py --onnx-path /root/TensorRT-Model-Optimizer/diffusers/sr_unet_int8_25/unet.onnx --output-onnx /root/TensorRT-Model-Optimizer/diffusers/sr_unet_int8_25/unet_fused.onnx
python onnx_utils/sdxl_graphsurgeon.py --onnx-path /root/TensorRT-Model-Optimizer/diffusers/controlnet_int8_merge/unet.onnx --output-onnx /root/TensorRT-Model-Optimizer/diffusers/controlnet_int8_merge/unet_fused.onnx
# INT8 SDXL Base or SDXL-turbo
# trtexec --onnx=./unet.onnx --shapes=sample:2x4x128x128,timestep:1,encoder_hidden_states:2x77x2048,text_embeds:2x1280,time_ids:2x6 --fp16 --int8 --builderOptimizationLevel=4 --saveEngine=unetxl.trt10.1.0.post12.dev1.engine
# INT8 SD 1.5
export LD_LIBRARY_PATH=/root/TensorRT-Model-Optimizer/TensorRT-10.1.0.27/lib/
./trtexec --onnx=/root/TensorRT-Model-Optimizer/diffusers/sr_unet_int8/unet.onnx --shapes=sample:2x8x128x128,timestep:1,encoder_hidden_states:2x77x768,timestep_cond:2x896 --fp16 --int8 --builderOptimizationLevel=4 --saveEngine=unet.trt10.1.0.engine
./trtexec --onnx=/root/TensorRT-Model-Optimizer/diffusers/sr_unet_int8_25/unet.onnx --shapes=sample:2x8x128x128,timestep:1,encoder_hidden_states:2x77x768,timestep_cond:2x896 --fp16 --int8 --builderOptimizationLevel=4 --saveEngine=unet.trt10.1.0.2.5.engine

./trtexec --onnx=/root/TensorRT-Model-Optimizer/diffusers/controlnet_int8_merge/unet.onnx \
--shapes=sample:8x4x64x64,encoder_hidden_states:8x77x768,images:2x8x3x512x512 \
--fp16 --int8 --builderOptimizationLevel=4 \
--saveEngine=control_unet_merge.trt10.1.0.engine



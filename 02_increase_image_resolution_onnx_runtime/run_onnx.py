import os
import onnx
import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

try:
    os.system("clear")
except:
    pass

# Loading ONNX Model
onnx_model = onnx.load("super_resolution.onnx")
onnx.checker.check_model(onnx_model)

# ONNX runtime inference session
ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# Running the model on an image using ONNX Runtime
img = Image.open("linkedin_dp.jpeg")

print("Size of Original Image: ", img.size)

resize = transforms.Resize([224, 224])
img = resize(img)

img_ycbcr = img.convert('YCbCr')
img_y, img_cb, img_cr = img_ycbcr.split()

to_tensor = transforms.ToTensor()
img_y = to_tensor(img_y)
img_y.unsqueeze_(0)

# Now, as a next step, let’s take the tensor representing the greyscale resized 
# cat image and run the super-resolution model in ONNX Runtime as explained previously.
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
ort_outs = ort_session.run(None, ort_inputs)
img_out_y = ort_outs[0]


# At this point, the output of the model is a tensor. Now, we’ll process the 
# output of the model to construct back the final output image from the output 
# tensor, and save the image. The post-processing steps have been adopted from 
# PyTorch implementation of super-resolution model here.

img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

# get the output image follow post-processing step from PyTorch implementation
final_img = Image.merge(
    "YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.Resampling.BICUBIC),
        img_cr.resize(img_out_y.size, Image.Resampling.BICUBIC),
    ]).convert("RGB")

# Save the image, we will compare this with the output image from mobile device
final_img.save("linkedin_dp_high_res.jpeg")


print("Size of Final Image: ", final_img.size)
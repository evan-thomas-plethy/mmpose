import torch
import coremltools as ct

# Load your TorchScript model
model_path = "rtmpose-m_traced.pt"
traced_model = torch.jit.load(model_path)
traced_model.eval()

# MMPose preprocessing
mean = [123.675, 116.28, 103.53]
std = [58.395, 57.12, 57.375]

global_std = sum(std)/len(std)        
scale = 1.0 / global_std                

bias = [- m / s for m, s in zip(mean, std)]

coreml_model = ct.convert(
    traced_model,
    convert_to="neuralnetwork",
    inputs=[
        ct.ImageType(
            shape=(1, 3, 256, 192),
            scale=scale,
            bias=bias
        )
    ]
)

# Save the converted CoreML model
coreml_model.save("rtmpose-m_Image.mlmodel")
print("Model successfully converted and saved as 'rtmpose-m_Image.mlmodel'")
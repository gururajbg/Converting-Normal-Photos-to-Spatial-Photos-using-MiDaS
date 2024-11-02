import torch

# Specify the model type you want to use
model_type = "DPT_Large"  # You can also use "MiDaS" for the smaller model

# Load the MiDaS model
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()  # Set the model to evaluation mode

# Save the model as a TorchScript model
dummy_input = torch.rand(1, 3, 384, 384)  # Dummy input for tracing
traced_model = torch.jit.trace(midas, dummy_input)
traced_model.save("midas_model.pt")  # Save the TorchScript model

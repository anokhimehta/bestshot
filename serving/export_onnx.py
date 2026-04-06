import torch
import torchvision.models as models
import onnx
import os

def export_efficientnet():
    print("Starting export...")
    model = models.efficientnet_b3(weights=None)
    model.eval()

    dummy_input = torch.randn(7, 3, 300, 300)

    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print("Success: model.onnx has been generated.")

if __name__ == "__main__":
    # 1. First, create the file
    export_efficientnet()

    # 2. Now, validate the file you just created
    if os.path.exists("model.onnx"):
        print("Validating model...")
        onnx_model = onnx.load("model.onnx")
        onnx.checker.check_model(onnx_model)
        print("ONNX structure is valid! Ready to push.")
    else:
        print("Export failed: model.onnx not found.")
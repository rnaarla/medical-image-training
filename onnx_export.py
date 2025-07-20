import torch
from src.core.model import get_model
# Project configuration for reorganized structure
sys.path.insert(0, str(Path(__file__).parent))
from utils.project_config import setup_imports



def export_model(checkpoint_path="best_model.pt", onnx_path="model.onnx"):
    model = get_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input__0'],
        output_names=['output__0'],
        dynamic_axes={
            'input__0': {0: 'batch_size'},
            'output__0': {0: 'batch_size'}
        }
    )
    print(f"ONNX model exported to {onnx_path}")

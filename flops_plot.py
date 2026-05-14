# Libraries
import os
import json
import hydra 
import torch 
from fvcore.nn import FlopCountAnalysis

# Custom functions 
from omegaconf import OmegaConf


def flatten_params(params):
    if isinstance(params, dict):
        return "_".join(f"{k}={v}" for k, v in params.items())
    return str(params)
_safe_globals = {
    "__builtins__": None,   # disable all other builtins
    "round": round,
}

# Now eval("…") will have access to round()
OmegaConf.register_new_resolver(
    "eval",
    lambda expr: eval(expr, _safe_globals, {})
)
OmegaConf.register_new_resolver("flatten_params", flatten_params)


# Hydra configuration 
@hydra.main(config_path="configs",
            version_base='1.2',
            config_name="default")
def main(cfg):

    # Set seed for reproducibility 
    torch.manual_seed(43422)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dataset parameters 
    batch_size = cfg.dataset.batch_size

    # Get model 
    model = hydra.utils.instantiate(cfg.model)

    # Get channel
    channel = hydra.utils.instantiate(cfg.communication.channel)
    
    # Apply method to the model 
    model = hydra.utils.instantiate(cfg.method.model,
                                    channel = channel,
                                    split_index = cfg.hyperparameters.split_index,
                                    model=model).to(device)

    inputs = torch.randn(batch_size, 3, 224, 224).to(device)  # Example input, adjust shape as needed
    


    # Extract model name, method name, and compression ratio from config
    model_name = cfg.model.model_name 
    method_name = cfg.method.name
    compression_ratio = model.compression
    
    # Normalize method name for comparison
    method_name_lower = method_name.lower()
    
    # Set model to training mode to enable FLOPS tracking for custom operations
    model.train()
    print(f"Model training mode: {model.training}")
    
    # Run forward pass before FlopCountAnalysis to populate FLOPS
    with torch.no_grad():
        model(inputs)
    
    # Check encoder and decoder FLOPS after forward pass
    encoder_flops_before = 0
    decoder_flops_before = 0

    if method_name_lower == "c3-sl":
        for module in model.model.modules():
            if hasattr(module, '__class__') and module.__class__.__name__ == 'Encoder':
                encoder_flops_before += getattr(module, 'conv_flops', 0)
                print(f"Found Encoder with conv_flops: {module.conv_flops}")
            elif hasattr(module, '__class__') and module.__class__.__name__ == 'Decoder':
                decoder_flops_before += getattr(module, 'corr_flops', 0)
                print(f"Found Decoder with corr_flops: {module.corr_flops}")
    
    flops = FlopCountAnalysis(model, inputs)
    
    # Get total FLOPS from FlopCountAnalysis
    standard_flops = flops.total()

    server_modules = [f"model.blocks.{i}" for i in range(4, 20)] + ["model.head", "model.norm"]
    edge_modules = [f"model.blocks.{i}" for i in range(4)] + ["model.patch_embed", "compressor_module"]

    server_flops = sum(flops.by_module().get(mod, 0) for mod in server_modules)
    edge_flops = sum(flops.by_module().get(mod, 0) for mod in edge_modules)
    
    
    # Get custom circular convolution FLOPS from encoder and decoder
    encoder_flops = 0
    decoder_flops = 0

    if method_name_lower == "c3-sl":
        print("CIAO")
        for module in model.model.modules():
            if hasattr(module, '__class__') and module.__class__.__name__ == 'Encoder':
                encoder_flops += getattr(module, 'conv_flops', 0)
            elif hasattr(module, '__class__') and module.__class__.__name__ == 'Decoder':
                decoder_flops += getattr(module, 'corr_flops', 0)
    
    # Add encoder FLOPS to edge, decoder FLOPS to server
    edge_flops += encoder_flops
    server_flops += decoder_flops

    if server_flops + edge_flops - encoder_flops - decoder_flops == standard_flops:
        print(f"FLOPS correctly partitioned: Server FLOPS = {server_flops}, Edge FLOPS = {edge_flops}")
    else:       
        print(f"FLOPS partitioning note:\n Standard FLOPS = {standard_flops},\n Server FLOPS (with custom) = {server_flops},\n Edge FLOPS (with custom) = {edge_flops}")
    
    if encoder_flops > 0:
        print(f"Encoder (Edge) circular convolution FLOPS: {encoder_flops}")
    if decoder_flops > 0:
        print(f"Decoder (Server) circular correlation FLOPS: {decoder_flops}")





    
    # Load existing FLOPS data or create new dict
    flops_file = "/home/federico/Desktop/Split_Learning/results/flops.json"
    os.makedirs(os.path.dirname(flops_file), exist_ok=True)
    
    if os.path.exists(flops_file) and os.path.getsize(flops_file) > 0:
        with open(flops_file, 'r') as f:
            try:
                flops_data = json.load(f)
            except json.JSONDecodeError:
                flops_data = {}
    else:
        flops_data = {}
    
    # Store FLOPS in nested structure: model -> method -> compression_ratio -> {server, edge}
    if model_name not in flops_data:
        flops_data[model_name] = {}
    if method_name not in flops_data[model_name]:
        flops_data[model_name][method_name] = {}
    
    flops_data[model_name][method_name][str(compression_ratio)] = {
        'server_flops': float(server_flops),
        'edge_flops': float(edge_flops)
    }
    
    # Save to JSON file
    with open(flops_file, 'w') as f:
        json.dump(flops_data, f, indent=2)
    
    print(f"FLOPS stored - Model: {model_name}, Method: {method_name}, Compression: {compression_ratio}")
    print(f"  Server FLOPS: {server_flops}")
    print(f"  Edge FLOPS: {edge_flops}")
    return 

    

# At the very bottom
if __name__ == "__main__":
    main()

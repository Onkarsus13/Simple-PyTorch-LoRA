import argparse
from .lora import LoRAConfig, apply_lora, merge_lora

def main():
    parser = argparse.ArgumentParser(description='Apply or merge LoRA adapters')
    parser.add_argument('--merge', action='store_true', help='Merge LoRA into base weights')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model directory')
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model_path)
    if args.merge:
        merge_lora(model)
        model.save_pretrained(args.model_path)
        print('Merged LoRA into', args.model_path)
    else:
        # Example: inject with default config
        cfg = LoRAConfig()
        apply_lora(model, cfg)
        model.save_pretrained(args.model_path)
        print('Injected LoRA into', args.model_path)

if __name__ == '__main__':
    main()

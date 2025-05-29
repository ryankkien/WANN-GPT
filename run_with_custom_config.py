import argparse
from pathlib import Path
import sys
from typing import Optional # Added for type hinting

# Add project root to path (if needed, like in run_demo.py)
project_root = Path(__file__).resolve().parent.parent # Adjust if your script is in a subdirectory
sys.path.insert(0, str(project_root))

from wann_gpt import load_config, EvolutionEngine, SharedWeightEvaluator, load_classification_data # Import necessary components

def main(config_file_path: Optional[str] = None, preset_name: Optional[str] = None): # Modified signature
    config = None # Initialize config
    
    try:
        if config_file_path:
            print(f"Loading configuration from file: {config_file_path}")
            config = load_config(config_path=config_file_path)
        elif preset_name:
            print(f"Loading configuration from preset: {preset_name}")
            config = load_config(preset=preset_name)
        else:
            # This case should be prevented by argparse mutually exclusive group
            print("Error: No configuration source specified. Use --config_file or --preset.")
            return
        
        if not config:
            print("Error: Failed to load configuration.")
            return

        print(f"Successfully loaded configuration: {config.experiment_name}")
        print(f"Number of generations set to: {config.evolution.num_generations}")
        print(f"Population size: {config.evolution.population_size}")

        # --- Example: Running Evolution (adapt as needed) ---
        print("\n--- Starting Evolution with Custom Config ---")

        # Ensure device is set (can also be in your config file)
        if not hasattr(config.training, 'device'):
            print("Device not specified in training config, defaulting to CPU.")
            device = "cpu"
        else:
            device = config.training.device
        
        print(f"Using device: {device}")

        # Load data (example for classification)
        # Adjust dataset_name, vocab_size, etc., or ensure they are in your config
        print("Loading dataset...")
        train_loader, test_loader, num_classes = load_classification_data(
            dataset_name=config.data.dataset_name, # Assumes 'dataset_name' is in your DataConfig
            vocab_size=config.data.vocab_size,
            max_length=config.data.max_length,
            subset_size=config.data.subset_size, # Can be None
            batch_size=config.data.batch_size
        )
        print(f"Dataset loaded. Train batches: {len(train_loader)}, Test batches: {len(test_loader)}, Num classes: {num_classes}")

        # Update model and evolution config with num_classes if it's a classification task
        if config.data.task_type == "classification":
            config.model.num_classes = num_classes
            config.evolution.num_classes = num_classes
            print(f"Set num_classes to {num_classes} for model and evolution.")
        
        # Create evolution components
        evaluator = SharedWeightEvaluator(device=device) # Pass device
        engine = EvolutionEngine(
            config.evolution, 
            evaluator, 
            save_dir=Path(config.logging.log_dir) / "evolution_results" # Example save directory
        )
        print(f"EvolutionEngine initialized. Population: {config.evolution.population_size}, Generations: {config.evolution.num_generations}")

        # Run evolution
        print("Starting evolution...")
        best_genome = engine.evolve(
            dataloader=train_loader,
            task_type=config.data.task_type,
            # log_wandb=config.logging.use_wandb # if you use wandb
        )
        
        print("\nEvolution completed!")
        if best_genome:
            fitness_key = 'classification' if config.data.task_type == "classification" else 'generation' # Adjust key as needed
            print(f"Best fitness ({fitness_key}): {best_genome.get_fitness(fitness_key):.4f}")
            print(f"Complexity: {best_genome.calculate_complexity()}")
        else:
            print("Evolution did not yield a best genome.")

        print("--- Evolution Demo Finished ---")

    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_file_path}") # Added FileNotFoundError back
    except Exception as e:
        print(f"An error occurred: {e}")
        # import traceback
        # traceback.print_exc() # For more detailed error information

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WANN-GPT evolution with a specific configuration.")
    group = parser.add_mutually_exclusive_group(required=True) # Make one of them required
    group.add_argument(
        "--config_file", 
        type=str, 
        help="Path to the YAML or JSON configuration file."
    )
    group.add_argument(
        "--preset",
        type=str,
        help="Name of the preset configuration to use (e.g., 'debug', 'classification_large')."
    )
    args = parser.parse_args()
    
    main(config_file_path=args.config_file, preset_name=args.preset) 
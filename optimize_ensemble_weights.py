#!/usr/bin/env python3
"""
Optimize ML Ensemble Weights for Different Market Conditions

This script optimizes the ensemble weights for different market regimes to maximize
prediction accuracy and trading performance across various market conditions.

It uses a combination of grid search and Bayesian optimization to find optimal weights.
"""

import os
import sys
import json
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logs/weight_optimization.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

# Import local modules
from backtest_ml_ensemble import MLEnsembleBacktester
from advanced_ensemble_model import DynamicWeightedEnsemble

class EnsembleWeightOptimizer:
    """
    Optimizes the ensemble weights for different market regimes
    
    This class analyzes historical data and tries different weight combinations
    to find optimal weights for different market conditions.
    """
    
    def __init__(self, trading_pair='SOL/USD', timeframe='1h',
                 output_dir='optimization_results/ensemble_weights',
                 fitness_metric='f1', # 'accuracy', 'f1', 'profit', 'sharpe'
                 initial_capital=10000.0,
                 population_size=20,
                 generations=10,
                 mutation_rate=0.2,
                 crossover_rate=0.7):
        """
        Initialize the weight optimizer
        
        Args:
            trading_pair: Trading pair to optimize weights for
            timeframe: Timeframe to optimize weights for
            output_dir: Directory to save optimization results
            fitness_metric: Metric to optimize for
            initial_capital: Initial capital for backtesting
            population_size: Size of the genetic algorithm population
            generations: Number of generations to run
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
        """
        self.trading_pair = trading_pair
        self.timeframe = timeframe
        self.output_dir = output_dir
        self.fitness_metric = fitness_metric
        self.initial_capital = initial_capital
        
        # GA parameters
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Define default model types
        self.model_types = ['tcn', 'cnn', 'lstm', 'gru', 'bilstm', 'attention', 'transformer', 'hybrid']
        
        # Define market regimes
        self.market_regimes = ['normal_trending_up', 'normal_trending_down', 
                             'volatile_trending_up', 'volatile_trending_down']
        
        # Load historical data
        logger.info(f"Initializing weight optimizer for {trading_pair} ({timeframe})")
        
        # Create a backtester for data loading and evaluation
        self.backtester = MLEnsembleBacktester(trading_pair, timeframe)
        
        # Create an ensemble model
        self.ensemble = DynamicWeightedEnsemble(trading_pair, timeframe)
    
    def _load_historical_data(self):
        """
        Load and preprocess historical data
        
        Returns:
            pd.DataFrame: Historical data with detected regimes
        """
        logger.info("Loading historical data...")
        
        # Load data using the backtester
        data = self.backtester.load_data()
        
        if data is None or len(data) == 0:
            logger.error(f"Failed to load historical data for {self.trading_pair} ({self.timeframe})")
            return None
        
        logger.info(f"Loaded {len(data)} data points")
        
        # Detect market regimes
        data = self.backtester._detect_market_regimes(data)
        logger.info("Detected market regimes in historical data")
        
        return data
    
    def _split_data_by_regime(self, data):
        """
        Split data into different market regimes
        
        Args:
            data: DataFrame with detected regimes
            
        Returns:
            dict: Dictionary of DataFrames for each regime
        """
        logger.info("Splitting data by market regime...")
        
        regime_data = {}
        
        for regime in self.market_regimes:
            regime_df = data[data['market_regime'] == regime].copy()
            regime_data[regime] = regime_df
            logger.info(f"  {regime}: {len(regime_df)} data points")
        
        return regime_data
    
    def _evaluate_weights(self, weights, data, regime=None):
        """
        Evaluate a set of weights using the backtester
        
        Args:
            weights: Dictionary of weights for each model type
            data: DataFrame to evaluate on
            regime: Specific regime to evaluate for (or None for all data)
            
        Returns:
            dict: Evaluation metrics
        """
        # Set the weights in the ensemble
        self.ensemble.update_model_weights(weights)
        
        # Run a backtest with the given weights
        results = self.backtester.run_backtest(
            data=data,
            ensemble_model=self.ensemble,
            initial_capital=self.initial_capital,
            plot_results=False
        )
        
        if not results:
            logger.warning("Backtest returned no results")
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'win_rate': 0,
                'max_drawdown': 1.0,  # Worst possible
            }
        
        # Extract metrics
        metrics = results.get('performance', {}).get('trading_metrics', {})
        model_metrics = results.get('performance', {}).get('model_metrics', {})
        
        return {
            'accuracy': model_metrics.get('accuracy', 0),
            'precision': model_metrics.get('precision', 0),
            'recall': model_metrics.get('recall', 0),
            'f1': model_metrics.get('f1', 0),
            'total_return': metrics.get('total_return_pct', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'win_rate': metrics.get('win_rate', 0),
            'max_drawdown': metrics.get('max_drawdown', 1.0),
        }
    
    def _get_fitness_metric_value(self, metrics):
        """
        Get the value of the fitness metric from evaluation metrics
        
        Args:
            metrics: Evaluation metrics dictionary
            
        Returns:
            float: Value of the fitness metric
        """
        if self.fitness_metric == 'accuracy':
            return metrics['accuracy']
        elif self.fitness_metric == 'f1':
            return metrics['f1']
        elif self.fitness_metric == 'profit':
            return metrics['total_return']
        elif self.fitness_metric == 'sharpe':
            return metrics['sharpe_ratio']
        else:
            logger.warning(f"Unknown fitness metric: {self.fitness_metric}, using accuracy")
            return metrics['accuracy']
    
    def _generate_random_weights(self):
        """
        Generate a random set of weights
        
        Returns:
            dict: Random weights for each model type
        """
        # Generate random weights
        raw_weights = {model_type: random.random() for model_type in self.model_types}
        
        # Normalize weights to sum to 1
        total = sum(raw_weights.values())
        normalized_weights = {model_type: weight / total for model_type, weight in raw_weights.items()}
        
        return normalized_weights
    
    def _crossover(self, parent1, parent2):
        """
        Perform crossover between two parents
        
        Args:
            parent1: First parent weights
            parent2: Second parent weights
            
        Returns:
            dict: Child weights
        """
        # Simple crossover: take weighted average of parents
        crossover_point = random.random()  # Weight for parent1
        
        child = {}
        for model_type in self.model_types:
            child[model_type] = parent1[model_type] * crossover_point + parent2[model_type] * (1 - crossover_point)
        
        # Normalize
        total = sum(child.values())
        normalized_child = {model_type: weight / total for model_type, weight in child.items()}
        
        return normalized_child
    
    def _mutate(self, weights):
        """
        Mutate weights
        
        Args:
            weights: Weights to mutate
            
        Returns:
            dict: Mutated weights
        """
        # Copy weights
        mutated = weights.copy()
        
        # Randomly select model types to mutate
        for model_type in self.model_types:
            if random.random() < self.mutation_rate:
                # Add random noise
                mutation = random.uniform(-0.2, 0.2)
                mutated[model_type] = max(0.01, min(0.99, mutated[model_type] + mutation))
        
        # Normalize
        total = sum(mutated.values())
        normalized = {model_type: weight / total for model_type, weight in mutated.items()}
        
        return normalized
    
    def optimize_weights_for_regime(self, regime, data):
        """
        Optimize weights for a specific market regime using a genetic algorithm
        
        Args:
            regime: Market regime to optimize for
            data: DataFrame for the regime
            
        Returns:
            tuple: (best_weights, best_fitness, all_fitness_history)
        """
        logger.info(f"Optimizing weights for {regime} regime...")
        
        if len(data) < 100:
            logger.warning(f"Insufficient data for {regime} regime (only {len(data)} points)")
            return self._generate_random_weights(), 0, []
        
        # Initialize population
        population = [self._generate_random_weights() for _ in range(self.population_size)]
        
        # Track fitness history
        fitness_history = []
        best_fitness = 0
        best_weights = None
        
        # Run genetic algorithm
        for generation in range(self.generations):
            logger.info(f"Generation {generation+1}/{self.generations}")
            
            # Evaluate fitness
            fitness_values = []
            for weights in tqdm(population, desc=f"Evaluating {regime} weights"):
                metrics = self._evaluate_weights(weights, data, regime)
                fitness = self._get_fitness_metric_value(metrics)
                fitness_values.append((weights, fitness, metrics))
            
            # Sort by fitness (descending)
            fitness_values.sort(key=lambda x: x[1], reverse=True)
            
            # Update best weights
            if fitness_values[0][1] > best_fitness:
                best_fitness = fitness_values[0][1]
                best_weights = fitness_values[0][0]
            
            # Track history
            generation_stats = {
                'generation': generation,
                'best_fitness': fitness_values[0][1],
                'avg_fitness': sum(f[1] for f in fitness_values) / len(fitness_values),
                'best_weights': fitness_values[0][0],
                'best_metrics': fitness_values[0][2]
            }
            fitness_history.append(generation_stats)
            
            logger.info(f"  Best fitness: {fitness_values[0][1]:.4f}, Avg fitness: {generation_stats['avg_fitness']:.4f}")
            
            # Early stopping if perfect fitness
            if best_fitness >= 0.99:
                logger.info("Early stopping: reached near-perfect fitness")
                break
            
            # Create next generation
            next_population = []
            
            # Elitism: keep best individual
            next_population.append(fitness_values[0][0])
            
            # Fill the rest with crossover and mutation
            while len(next_population) < self.population_size:
                # Tournament selection
                tournament_size = 3
                parent1 = max(random.sample(fitness_values, tournament_size), key=lambda x: x[1])[0]
                parent2 = max(random.sample(fitness_values, tournament_size), key=lambda x: x[1])[0]
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1.copy()
                
                # Mutation
                child = self._mutate(child)
                
                next_population.append(child)
            
            population = next_population
        
        logger.info(f"Optimization complete for {regime} regime")
        logger.info(f"Best fitness: {best_fitness:.4f}")
        logger.info(f"Best weights: {best_weights}")
        
        return best_weights, best_fitness, fitness_history
    
    def optimize_all_regimes(self):
        """
        Optimize weights for all market regimes
        
        Returns:
            dict: Optimized weights for each regime
        """
        logger.info("Starting optimization for all market regimes...")
        
        # Load historical data
        data = self._load_historical_data()
        if data is None:
            logger.error("Failed to load historical data")
            return None
        
        # Split data by regime
        regime_data = self._split_data_by_regime(data)
        
        # Optimize weights for each regime
        optimized_weights = {}
        optimization_results = {}
        
        for regime in self.market_regimes:
            if regime not in regime_data or len(regime_data[regime]) < 50:
                logger.warning(f"Insufficient data for {regime} regime, skipping optimization")
                optimized_weights[regime] = {model_type: 1.0 / len(self.model_types) for model_type in self.model_types}
                continue
            
            best_weights, best_fitness, fitness_history = self.optimize_weights_for_regime(
                regime, regime_data[regime])
            
            optimized_weights[regime] = best_weights
            optimization_results[regime] = {
                'best_fitness': best_fitness,
                'fitness_history': fitness_history
            }
        
        # Evaluate optimized weights on the entire dataset
        logger.info("Evaluating optimized weights on full dataset...")
        
        # Create ensemble with regime-specific weights
        self.ensemble.update_regime_weights(optimized_weights)
        
        # Run a final backtest
        final_results = self.backtester.run_backtest(
            data=None,  # Use default data
            ensemble_model=self.ensemble,
            initial_capital=self.initial_capital,
            plot_results=True,
            save_plot=True,
            plot_path=os.path.join(self.output_dir, f'optimized_backtest_{self.trading_pair.replace("/", "")}_{self.timeframe}.png')
        )
        
        if final_results:
            # Extract metrics for the optimized ensemble
            metrics = final_results.get('performance', {}).get('trading_metrics', {})
            model_metrics = final_results.get('performance', {}).get('model_metrics', {})
            
            logger.info("=== Optimized Ensemble Performance ===")
            logger.info(f"Accuracy: {model_metrics.get('accuracy', 0):.4f}")
            logger.info(f"F1 Score: {model_metrics.get('f1', 0):.4f}")
            logger.info(f"Total Return: {metrics.get('total_return_pct', 0) * 100:.2f}%")
            logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Win Rate: {metrics.get('win_rate', 0) * 100:.2f}%")
            logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        
        # Visualize optimization results
        self._visualize_optimization(optimization_results)
        
        # Save optimized weights
        self._save_optimized_weights(optimized_weights)
        
        logger.info("Optimization complete for all regimes")
        
        return optimized_weights
    
    def _visualize_optimization(self, optimization_results):
        """
        Visualize optimization results
        
        Args:
            optimization_results: Results from optimization
        """
        logger.info("Visualizing optimization results...")
        
        if not optimization_results:
            logger.warning("No optimization results to visualize")
            return
        
        # Create a figure
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Fitness history by regime
        plt.subplot(2, 2, 1)
        
        for regime, results in optimization_results.items():
            if 'fitness_history' not in results or not results['fitness_history']:
                continue
            
            # Extract fitness history
            generations = [entry['generation'] for entry in results['fitness_history']]
            best_fitness = [entry['best_fitness'] for entry in results['fitness_history']]
            avg_fitness = [entry['avg_fitness'] for entry in results['fitness_history']]
            
            # Plot best fitness
            plt.plot(generations, best_fitness, label=f"{regime} (best)", linewidth=2)
            # Plot average fitness
            plt.plot(generations, avg_fitness, label=f"{regime} (avg)", linestyle='--', alpha=0.7)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Optimization Progress by Regime')
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Final weights by regime
        plt.subplot(2, 2, 2)
        
        # Collect final weights
        regimes = []
        model_weights = {model_type: [] for model_type in self.model_types}
        
        for regime, results in optimization_results.items():
            if 'fitness_history' not in results or not results['fitness_history']:
                continue
            
            # Get best weights
            best_weights = results['fitness_history'][-1]['best_weights']
            
            regimes.append(regime)
            for model_type in self.model_types:
                model_weights[model_type].append(best_weights.get(model_type, 0))
        
        # Create bar positions
        x = np.arange(len(regimes))
        width = 0.1
        n_models = len(self.model_types)
        offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
        
        # Plot bars for each model
        for i, model_type in enumerate(self.model_types):
            plt.bar(x + offsets[i], model_weights[model_type], width, label=model_type)
        
        plt.xlabel('Market Regime')
        plt.ylabel('Weight')
        plt.title('Optimized Weights by Regime')
        plt.xticks(x, regimes, rotation=45)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Performance metrics by regime
        plt.subplot(2, 2, 3)
        
        # Collect metrics
        metrics_keys = ['accuracy', 'f1', 'total_return', 'sharpe_ratio']
        metrics_values = {key: [] for key in metrics_keys}
        
        for regime, results in optimization_results.items():
            if 'fitness_history' not in results or not results['fitness_history']:
                continue
            
            # Get best metrics
            best_metrics = results['fitness_history'][-1]['best_metrics']
            
            for key in metrics_keys:
                metrics_values[key].append(best_metrics.get(key, 0))
        
        # Create bar positions
        width = 0.2
        offsets = np.linspace(-(len(metrics_keys)-1)*width/2, (len(metrics_keys)-1)*width/2, len(metrics_keys))
        
        # Plot bars for each metric
        for i, key in enumerate(metrics_keys):
            plt.bar(x + offsets[i], metrics_values[key], width, label=key)
        
        plt.xlabel('Market Regime')
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics by Regime')
        plt.xticks(x, regimes, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Radar chart of model weights for each regime
        plt.subplot(2, 2, 4, polar=True)
        
        # Compute angles for each model (in radians)
        angles = np.linspace(0, 2*np.pi, len(self.model_types), endpoint=False).tolist()
        
        # Close the plot
        angles += angles[:1]
        
        # Plot each regime
        for i, regime in enumerate(regimes):
            # Get weights
            weights = [model_weights[model][i] for model in self.model_types]
            weights += weights[:1]  # Close the line
            
            plt.plot(angles, weights, label=regime, linewidth=2, marker='o')
        
        # Set ticks and labels
        plt.xticks(angles[:-1], self.model_types)
        plt.title('Model Weights Radar Chart')
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'optimization_results_{self.trading_pair.replace("/", "")}_{self.timeframe}.png'))
        plt.close()
        
        logger.info(f"Saved visualization to {os.path.join(self.output_dir, f'optimization_results_{self.trading_pair.replace('/', '')}_{self.timeframe}.png')}")
    
    def _save_optimized_weights(self, optimized_weights):
        """
        Save optimized weights to a file
        
        Args:
            optimized_weights: Dictionary of optimized weights by regime
        """
        logger.info("Saving optimized weights...")
        
        # Create filename
        filename = os.path.join(self.output_dir, f'optimized_weights_{self.trading_pair.replace("/", "")}_{self.timeframe}.json')
        
        # Add metadata
        output = {
            'trading_pair': self.trading_pair,
            'timeframe': self.timeframe,
            'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'fitness_metric': self.fitness_metric,
            'optimized_weights': optimized_weights
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved optimized weights to {filename}")

def main():
    """Main function to run the optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize ML ensemble weights')
    parser.add_argument('--pair', type=str, default='SOL/USD', help='Trading pair (default: SOL/USD)')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe (default: 1h)')
    parser.add_argument('--metric', type=str, default='f1', help='Fitness metric (default: f1)')
    parser.add_argument('--capital', type=float, default=10000.0, help='Initial capital (default: 10000.0)')
    parser.add_argument('--population', type=int, default=20, help='Population size (default: 20)')
    parser.add_argument('--generations', type=int, default=10, help='Number of generations (default: 10)')
    
    args = parser.parse_args()
    
    optimizer = EnsembleWeightOptimizer(
        trading_pair=args.pair,
        timeframe=args.timeframe,
        fitness_metric=args.metric,
        initial_capital=args.capital,
        population_size=args.population,
        generations=args.generations
    )
    
    optimized_weights = optimizer.optimize_all_regimes()
    
    if optimized_weights:
        print("\nOptimization completed successfully.")
        print("Optimized weights for each regime:")
        for regime, weights in optimized_weights.items():
            print(f"\n{regime}:")
            for model_type, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
                print(f"  {model_type}: {weight:.4f}")
    else:
        print("\nOptimization failed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
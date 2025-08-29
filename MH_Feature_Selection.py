#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MH_Feature_Selection.py - Simple Genetic Algorithm for Financial Feature Selection

This module implements a basic Genetic Algorithm (GA) for selecting optimal features
from financial time series data.

Author: Nicolas Cope
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

# Machine Learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Tensorflow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# For reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GeneticAlgorithm:
    """
    Simple Genetic Algorithm for feature selection in financial time series prediction.
    
    This class implements a basic GA that evolves a population of feature subsets.
    """
    
    def __init__(self, X_train, y_train, feature_names, model_type='random_forest', 
                 pop_size=50, crossover_rate=0.8, mutation_rate=0.2, elite_size=3,
                 max_generations=30, min_features=5, max_features=30, cv_folds=5,
                 random_state=42, adaptive_rates=True, diversity_threshold=0.15,
                 early_stopping_patience=10, debug=False):
        """
        Initialize the Genetic Algorithm for feature selection.
        
        Parameters:
        -----------
        X_train : pandas DataFrame or numpy array
            Training features
        y_train : pandas Series or numpy array
            Target variable
        feature_names : list
            Names of all available features
        model_type : str, default='random_forest'
            Model to use for fitness evaluation ('random_forest', 'xgboost', 'ann')
        pop_size : int, default=50
            Population size (increased for better exploration)
        crossover_rate : float, default=0.8
            Crossover rate
        mutation_rate : float, default=0.2
            Mutation rate (increased for better exploration)
        elite_size : int, default=3
            Number of elite individuals to preserve
        max_generations : int, default=30
            Maximum number of generations (increased)
        min_features : int, default=5
            Minimum number of features to use
        max_features : int, default=30
            Maximum number of features to use
        cv_folds : int, default=5
            Number of cross-validation folds (increased for stability)
        random_state : int, default=42
            Random seed for reproducibility
        adaptive_rates : bool, default=True
            Whether to use adaptive mutation/crossover rates
        diversity_threshold : float, default=0.15
            Minimum diversity threshold to maintain
        early_stopping_patience : int, default=10
            Generations to wait before early stopping
        """
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = feature_names
        self.model_type = model_type
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.max_generations = max_generations
        self.min_features = min_features
        self.max_features = min(len(feature_names), max_features)
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.adaptive_rates = adaptive_rates
        self.diversity_threshold = diversity_threshold
        self.early_stopping_patience = early_stopping_patience
        self.debug = debug
        
        # Adaptive rate parameters
        self.base_crossover_rate = crossover_rate
        self.base_mutation_rate = mutation_rate
        
        # Initialize tracking variables
        self.population = None
        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.best_features = []
        self.num_features_history = []
        self.diversity_history = []
        self.stagnation_count = 0
        
        # Set up cross-validation
        self.cv = TimeSeriesSplit(n_splits=cv_folds)
        
        np.random.seed(random_state)
    
    def initialize_population(self):
        """Initialize population with proper feature count distribution and some guided individuals"""
        population = []
        
        # Debug: Print what we're trying to create
        if hasattr(self, 'debug') and self.debug:
            print(f"Initializing population: {self.min_features}-{self.max_features} features")
        
        # Get feature importance ranking for guided initialization
        importance_ranking = self._get_feature_importance_ranking()
        
        # Create 25% guided individuals using feature importance
        guided_count = max(1, self.pop_size // 4)
        for i in range(guided_count):
            # Vary the number of top features to select
            num_features = self.min_features + (i * (self.max_features - self.min_features) // guided_count)
            num_features = max(self.min_features, min(self.max_features, num_features))
            
            # Select top features with some randomness
            individual = np.zeros(len(self.feature_names), dtype=bool)
            
            # Take top features but add some randomness
            top_features = importance_ranking[:num_features + 5]  # Get a few extra
            selected_indices = np.random.choice(top_features, size=num_features, replace=False)
            individual[selected_indices] = True
            population.append(individual)
        
        # Create remaining individuals with random distribution
        for i in range(guided_count, self.pop_size):
            # Calculate feature count for this individual
            # Distribute evenly across the range
            feature_range = self.max_features - self.min_features
            remaining_pop = self.pop_size - guided_count
            step = feature_range / remaining_pop if remaining_pop > 1 else 0
            target_features = int(self.min_features + ((i - guided_count) * step))
            
            # Add some randomness
            num_features = max(self.min_features, 
                              min(self.max_features, 
                                  target_features + np.random.randint(-2, 3)))
            
            # Create individual
            individual = np.zeros(len(self.feature_names), dtype=bool)
            selected_indices = np.random.choice(len(self.feature_names), 
                                              size=num_features, replace=False)
            individual[selected_indices] = True
            population.append(individual)
        
        # Debug: Check what we actually created
        if hasattr(self, 'debug') and self.debug:
            counts = [np.sum(ind) for ind in population]
            print(f"Created population with feature counts: {min(counts)}-{max(counts)} (avg: {np.mean(counts):.1f})")
            print(f"Guided individuals: {guided_count}/{self.pop_size}")
        
        self.population = population
        return population
    
    def _get_feature_importance_ranking(self):
        """Get feature importance ranking for initialization seeding"""
        try:
            model = self.create_model()
            model.fit(self.X_train, self.y_train)
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # Return indices sorted by importance (descending)
                return np.argsort(importances)[::-1]
            else:
                return np.arange(len(self.feature_names))
        except:
            return np.arange(len(self.feature_names))
    
    def create_model(self):
        """Create ML model based on model_type"""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        elif self.model_type == 'ann':
            # For ANN, we'll use a simplified approach with sklearn metrics
            return RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fitness_function(self, individual):
        """
        Enhanced fitness function with stability focus
        
        Parameters:
        -----------
        individual : numpy array
            Binary mask of selected features
        
        Returns:
        --------
        float
            Fitness score focusing on cross-validation stability
        """
        # Check if any features are selected
        num_features = np.sum(individual)
        if num_features < self.min_features or num_features > self.max_features:
            return 0.0
        
        try:
            # Get selected features
            if isinstance(self.X_train, pd.DataFrame):
                X_selected = self.X_train.iloc[:, individual]
            else:
                X_selected = self.X_train[:, individual]
            
            # Create model
            model = self.create_model()
            
            # Use cross-validation to get fitness score
            cv_scores = cross_val_score(
                model, 
                X_selected, 
                self.y_train,
                cv=self.cv, 
                scoring='accuracy'
            )
            
            # Enhanced fitness: prioritize stability and mean performance
            mean_accuracy = cv_scores.mean()
            cv_stability = 1.0 - cv_scores.std()  # Higher stability = less variance
            
            # Small complexity penalty but not too aggressive
            complexity_penalty = 0.005 * (num_features / len(individual))
            
            # Combine metrics with stability being important
            fitness = (0.7 * mean_accuracy + 0.3 * cv_stability) - complexity_penalty
            
            return max(0.0, fitness)
            
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"Error in fitness evaluation: {str(e)}")
            return 0.0
    
    def evaluate_population(self):
        """Evaluate fitness for entire population in parallel"""
        fitness_scores = Parallel(n_jobs=-1)(
            delayed(self.fitness_function)(individual) for individual in self.population
        )
        return fitness_scores
    
    def calculate_population_diversity(self):
        """Calculate population diversity using Hamming distance"""
        if not self.population or len(self.population) < 2:
            return 1.0
        
        total_distance = 0
        comparisons = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Hamming distance
                distance = np.sum(self.population[i] != self.population[j])
                total_distance += distance
                comparisons += 1
        
        # Normalize by maximum possible distance and number of comparisons
        max_distance = len(self.population[0])
        diversity = total_distance / (comparisons * max_distance)
        return diversity
    
    def inject_diversity(self, fitness_scores):
        """Inject diversity when population becomes too similar"""
        diversity = self.calculate_population_diversity()
        
        if diversity < self.diversity_threshold:
            # Find worst performers to replace
            worst_indices = np.argsort(fitness_scores)[:self.pop_size//4]  # Replace worst 25%
            
            for idx in worst_indices:
                # Create diverse individual
                if np.random.random() < 0.5:
                    # Random individual
                    num_features = np.random.randint(self.min_features, self.max_features + 1)
                    new_individual = np.zeros(len(self.feature_names), dtype=bool)
                    selected_indices = np.random.choice(len(self.feature_names), 
                                                      size=num_features, replace=False)
                    new_individual[selected_indices] = True
                    self.population[idx] = new_individual
                else:
                    # Flip random bits in existing individual to increase diversity
                    individual = self.population[idx].copy()
                    num_flips = np.random.randint(3, 8)  # Flip 3-7 features
                    flip_indices = np.random.choice(len(individual), size=num_flips, replace=False)
                    individual[flip_indices] = ~individual[flip_indices]
                    self._ensure_constraints(individual)
                    self.population[idx] = individual
    
    def update_adaptive_rates(self, generation):
        """Update mutation and crossover rates based on diversity and progress"""
        if not self.adaptive_rates:
            return
        
        diversity = self.calculate_population_diversity()
        
        # Increase mutation rate if diversity is low
        if diversity < self.diversity_threshold:
            self.mutation_rate = min(0.3, self.base_mutation_rate * 1.5)
            self.crossover_rate = max(0.6, self.base_crossover_rate * 0.8)
        else:
            # Gradually return to base rates
            self.mutation_rate = max(self.base_mutation_rate, self.mutation_rate * 0.95)
            self.crossover_rate = min(self.base_crossover_rate, self.crossover_rate * 1.05)
        
        # Adaptive rates based on generation progress
        progress = generation / self.max_generations
        if progress > 0.7:  # Late in evolution
            self.mutation_rate = max(0.05, self.mutation_rate * 0.9)  # Reduce mutation for fine-tuning
        elif progress < 0.3:  # Early in evolution
            self.mutation_rate = min(0.25, self.mutation_rate * 1.1)  # Increase exploration
    
    def select_parents(self, fitness_scores):
        """Simple tournament selection"""
        parents = []
        
        for _ in range(self.pop_size):
            # Tournament selection with size 3
            tournament_indices = np.random.choice(len(self.population), size=3, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx].copy())
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Simple uniform crossover"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Uniform crossover
        mask = np.random.random(len(parent1)) < 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        
        # Ensure constraints
        self._ensure_constraints(child1)
        self._ensure_constraints(child2)
        
        return child1, child2
    
    def _ensure_constraints(self, individual):
        """Ensure individual meets min/max feature constraints"""
        num_features = np.sum(individual)
        
        # Too few features - add some
        if num_features < self.min_features:
            zeros = np.where(individual == 0)[0]
            if len(zeros) > 0:
                to_flip = np.random.choice(
                    zeros, 
                    size=min(self.min_features - num_features, len(zeros)), 
                    replace=False
                )
                individual[to_flip] = True
        
        # Too many features - remove some
        elif num_features > self.max_features:
            ones = np.where(individual == 1)[0]
            to_flip = np.random.choice(
                ones, 
                size=num_features - self.max_features, 
                replace=False
            )
            individual[to_flip] = False
    
    def mutate(self, individual):
        """Simplified mutation with basic feature count management"""
        mutated = individual.copy()
        
        # Basic bit-flip mutation
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = not mutated[i]
        
        # Ensure we explore different feature counts
        if np.random.random() < 0.1:  # 10% chance to force feature count change
            current_features = np.sum(mutated)
            if current_features < self.max_features // 2:
                # Add features if we have few
                zeros = np.where(mutated == 0)[0]
                if len(zeros) > 0:
                    num_to_add = min(np.random.randint(1, 3), len(zeros))
                    to_flip = np.random.choice(zeros, size=num_to_add, replace=False)
                    mutated[to_flip] = True
            else:
                # Remove features if we have many
                ones = np.where(mutated == 1)[0]
                if len(ones) > self.min_features:
                    num_to_remove = min(np.random.randint(1, 2), len(ones) - self.min_features)
                    to_flip = np.random.choice(ones, size=num_to_remove, replace=False)
                    mutated[to_flip] = False
        
        # Ensure constraints
        self._ensure_constraints(mutated)
        
        return mutated
    
    def elitism(self, fitness_scores, new_population):
        """Simple elitism - preserve best individuals"""
        if self.elite_size > 0:
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elites = [self.population[i].copy() for i in elite_indices]
            
            # Replace worst individuals with elites
            for i, elite in enumerate(elites):
                new_population[i] = elite
        
        return new_population
    
    def evolve(self, generation):
        """Evolve the population for one generation with diversity management"""
        # Update adaptive rates (if enabled)
        self.update_adaptive_rates(generation)
        
        # Evaluate population
        fitness_scores = self.evaluate_population()
        
        # Check diversity and inject if needed (do this early)
        diversity = self.calculate_population_diversity()
        self.diversity_history.append(diversity)
        
        if diversity < self.diversity_threshold:
            self.inject_diversity(fitness_scores)
            # Re-evaluate after diversity injection
            fitness_scores = self.evaluate_population()
        
        # Update best individual and track stagnation
        best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[best_idx]
        
        if current_best_fitness > self.best_fitness:
            self.best_individual = self.population[best_idx].copy()
            self.best_fitness = current_best_fitness
            self.stagnation_count = 0
            
            # Update best features
            selected_features = [self.feature_names[i] for i in range(len(self.best_individual)) 
                                if self.best_individual[i]]
            self.best_features = selected_features
        else:
            self.stagnation_count += 1
        
        # Track history
        self.fitness_history.append(current_best_fitness)
        self.num_features_history.append(np.sum(self.best_individual) if self.best_individual is not None else 0)
        
        # Selection
        parents = self.select_parents(fitness_scores)
        
        # Create new population
        new_population = []
        
        # Crossover
        for i in range(0, self.pop_size, 2):
            if i+1 < self.pop_size:
                parent1, parent2 = parents[i], parents[i+1]
                child1, child2 = self.crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.append(parents[i])
        
        # Mutation
        for i in range(self.pop_size):
            new_population[i] = self.mutate(new_population[i])
        
        # Elitism
        new_population = self.elitism(fitness_scores, new_population)
        
        self.population = new_population
    
    def run(self, verbose=True):
        """Run the simplified genetic algorithm"""
        # Initialize population
        self.initialize_population()
        
        if verbose:
            print(f"Starting Streamlined GA:")
            print(f"  Population: {self.pop_size}, Generations: {self.max_generations}")
            print(f"  Features range: {self.min_features}-{self.max_features}")
            print(f"  Model: {self.model_type}, CV folds: {self.cv_folds}")
            
            # Show initial population stats
            initial_counts = [np.sum(ind) for ind in self.population]
            print(f"  Initial feature counts: {min(initial_counts)}-{max(initial_counts)} (avg: {np.mean(initial_counts):.1f})")
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.evolve(generation)
            
            if verbose:
                num_features = np.sum(self.best_individual) if self.best_individual is not None else 0
                
                # Calculate population feature statistics
                pop_feature_counts = [np.sum(ind) for ind in self.population]
                avg_pop_features = np.mean(pop_feature_counts)
                max_pop_features = np.max(pop_feature_counts)
                min_pop_features = np.min(pop_feature_counts)
                
                # Get current diversity
                current_diversity = self.diversity_history[-1] if self.diversity_history else 0
                
                print(f"Gen {generation:2d}: Fitness={self.best_fitness:.4f}, "
                      f"Best={num_features:2d}, Pop=[{min_pop_features:2d}-{avg_pop_features:.1f}-{max_pop_features:2d}], "
                      f"Div={current_diversity:.3f}, Stag={self.stagnation_count}")
            
            # Early stopping check
            if self.stagnation_count >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at generation {generation} (no improvement for {self.stagnation_count} generations)")
                break
        
        # Final results
        if verbose:
            print(f"\nGA completed!")
            print(f"Best fitness: {self.best_fitness:.4f}")
            print(f"Selected {len(self.best_features)} features: {self.best_features[:10]}{'...' if len(self.best_features) > 10 else ''}")
        
        return self.best_individual, self.best_fitness, self.best_features
    
    def plot_convergence(self):
        """Plot enhanced convergence history with diversity tracking"""
        plt.figure(figsize=(15, 10))
        
        # Fitness history
        plt.subplot(2, 2, 1)
        plt.plot(self.fitness_history, marker='o', linewidth=2, markersize=4)
        plt.title('Fitness Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness Score')
        plt.grid(True, alpha=0.3)
        
        # Number of features
        plt.subplot(2, 2, 2)
        plt.plot(self.num_features_history, marker='s', color='orange', linewidth=2, markersize=4)
        plt.title('Feature Count Evolution', fontsize=14, fontweight='bold')
        plt.xlabel('Generation')
        plt.ylabel('Number of Features')
        plt.grid(True, alpha=0.3)
        
        # Population diversity
        plt.subplot(2, 2, 3)
        if self.diversity_history:
            plt.plot(self.diversity_history, marker='^', color='green', linewidth=2, markersize=4)
            plt.axhline(y=self.diversity_threshold, color='red', linestyle='--', alpha=0.7, 
                       label=f'Diversity Threshold ({self.diversity_threshold})')
            plt.legend()
        plt.title('Population Diversity', fontsize=14, fontweight='bold')
        plt.xlabel('Generation')
        plt.ylabel('Diversity Score')
        plt.grid(True, alpha=0.3)
        
        # Fitness improvement rate
        plt.subplot(2, 2, 4)
        if len(self.fitness_history) > 1:
            improvement = np.diff(self.fitness_history)
            plt.plot(improvement, marker='d', color='purple', linewidth=2, markersize=4)
            plt.axhline(y=0, color='red', linestyle='-', alpha=0.5)
        plt.title('Fitness Improvement Rate', fontsize=14, fontweight='bold')
        plt.xlabel('Generation')
        plt.ylabel('Fitness Change')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Summary statistics
        if self.fitness_history:
            print(f"\nEvolution Summary:")
            print(f"Final fitness: {self.fitness_history[-1]:.4f}")
            print(f"Best fitness: {max(self.fitness_history):.4f}")
            print(f"Average fitness: {np.mean(self.fitness_history):.4f}")
            print(f"Fitness std: {np.std(self.fitness_history):.4f}")
            if self.diversity_history:
                print(f"Final diversity: {self.diversity_history[-1]:.3f}")
                print(f"Average diversity: {np.mean(self.diversity_history):.3f}")
            print(f"Total generations: {len(self.fitness_history)}")
            print(f"Early stopping at: {self.stagnation_count} generations without improvement")
    
    def plot_feature_importance(self, top_n=20):
        """Plot importance of selected features"""
        if not self.best_features:
            print("No features selected yet. Run the algorithm first.")
            return
        
        # Create model and fit on selected features
        model = self.create_model()
        
        if isinstance(self.X_train, pd.DataFrame):
            X_selected = self.X_train.iloc[:, self.best_individual]
        else:
            X_selected = self.X_train[:, self.best_individual]
            
        model.fit(X_selected, self.y_train)
        
        # Get feature importances (only works for tree-based models)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': self.best_features,
                'Importance': importances
            })
            feature_importance = feature_importance.sort_values('Importance', ascending=False)
            
            # Plot top N features
            plt.figure(figsize=(10, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(top_n))
            plt.title(f'Top {top_n} Selected Features by Importance')
            plt.tight_layout()
            plt.show()
        else:
            print("Feature importance plotting only available for tree-based models")
    
    def analyze_feature_selection_patterns(self):
        """Analyze patterns in the selected features"""
        if not self.best_features:
            print("No features selected yet. Run the algorithm first.")
            return
        
        print(f"\nFeature Selection Analysis:")
        print(f"{'='*50}")
        print(f"Total features available: {len(self.feature_names)}")
        print(f"Features selected: {len(self.best_features)}")
        print(f"Selection ratio: {len(self.best_features)/len(self.feature_names)*100:.1f}%")
        
        # Feature name analysis
        feature_types = {}
        for feature in self.best_features:
            # Extract feature type based on common naming patterns
            if 'sma' in feature.lower() or 'ma_' in feature.lower():
                feature_types['Moving Averages'] = feature_types.get('Moving Averages', 0) + 1
            elif 'rsi' in feature.lower():
                feature_types['RSI'] = feature_types.get('RSI', 0) + 1
            elif 'macd' in feature.lower():
                feature_types['MACD'] = feature_types.get('MACD', 0) + 1
            elif 'bb' in feature.lower() or 'bollinger' in feature.lower():
                feature_types['Bollinger Bands'] = feature_types.get('Bollinger Bands', 0) + 1
            elif 'atr' in feature.lower():
                feature_types['ATR'] = feature_types.get('ATR', 0) + 1
            elif 'volume' in feature.lower() or 'vol' in feature.lower():
                feature_types['Volume'] = feature_types.get('Volume', 0) + 1
            elif any(x in feature.lower() for x in ['high', 'low', 'close', 'open']):
                feature_types['Price-based'] = feature_types.get('Price-based', 0) + 1
            else:
                feature_types['Other'] = feature_types.get('Other', 0) + 1
        
        if feature_types:
            print(f"\nFeature Type Distribution:")
            for ftype, count in sorted(feature_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {ftype}: {count} ({count/len(self.best_features)*100:.1f}%)")
        
        return feature_types


# If run as main script
if __name__ == "__main__":
    print("MH_Feature_Selection.py - Simple Genetic Algorithm for Financial Feature Selection")
    print("This module should be imported, not run directly.")
    print("Example usage:")
    print("from MH_Feature_Selection import GeneticAlgorithm")
    print("ga = GeneticAlgorithm(X_train, y_train, feature_names)")
    print("best_mask, best_fitness, selected_features = ga.run()")

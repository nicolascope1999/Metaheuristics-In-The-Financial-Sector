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
                 pop_size=50, crossover_rate=0.8, mutation_rate=0.15, elite_size=3,
                 max_generations=25, min_features=5, max_features=30, cv_folds=5,
                 random_state=42, adaptive_rates=True, diversity_threshold=0.1,
                 early_stopping_patience=3):
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
        mutation_rate : float, default=0.15
            Mutation rate (increased for better exploration)
        elite_size : int, default=3
            Number of elite individuals to preserve
        max_generations : int, default=25
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
        diversity_threshold : float, default=0.1
            Minimum diversity threshold to maintain
        early_stopping_patience : int, default=3
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
        """Initialize population with diverse feature subsets using multiple strategies"""
        population = []
        
        # Strategy 1: Random initialization (50% of population)
        random_count = self.pop_size // 2
        for _ in range(random_count):
            num_features = np.random.randint(self.min_features, self.max_features + 1)
            mask = np.zeros(len(self.feature_names), dtype=bool)
            selected_indices = np.random.choice(
                len(self.feature_names), 
                size=num_features, 
                replace=False
            )
            mask[selected_indices] = True
            population.append(mask)
        
        # Strategy 2: Feature importance seeded initialization (30% of population)
        if hasattr(self, '_get_feature_importance_ranking'):
            importance_count = int(self.pop_size * 0.3)
            importance_ranking = self._get_feature_importance_ranking()
            
            for _ in range(importance_count):
                num_features = np.random.randint(self.min_features, self.max_features + 1)
                mask = np.zeros(len(self.feature_names), dtype=bool)
                
                # Select top features with some randomness
                top_features = min(num_features * 2, len(importance_ranking))
                selected_indices = np.random.choice(
                    importance_ranking[:top_features],
                    size=num_features,
                    replace=False
                )
                mask[selected_indices] = True
                population.append(mask)
        else:
            # Fallback to random if importance ranking not available
            for _ in range(int(self.pop_size * 0.3)):
                num_features = np.random.randint(self.min_features, self.max_features + 1)
                mask = np.zeros(len(self.feature_names), dtype=bool)
                selected_indices = np.random.choice(
                    len(self.feature_names), 
                    size=num_features, 
                    replace=False
                )
                mask[selected_indices] = True
                population.append(mask)
        
        # Strategy 3: Size-varied initialization with better distribution (remaining 20%)
        remaining_count = self.pop_size - len(population)
        feature_range = self.max_features - self.min_features + 1
        
        for i in range(remaining_count):
            # Create individuals with evenly distributed feature counts
            if i < remaining_count // 3:
                # Small feature sets (lower third of range)
                num_features = self.min_features + (i % ((feature_range // 3) + 1))
            elif i < 2 * remaining_count // 3:
                # Medium feature sets (middle third of range)
                start_mid = self.min_features + feature_range // 3
                num_features = start_mid + ((i - remaining_count // 3) % ((feature_range // 3) + 1))
            else:
                # Large feature sets (upper third of range)
                start_high = self.min_features + 2 * feature_range // 3
                num_features = start_high + ((i - 2 * remaining_count // 3) % ((feature_range // 3) + 1))
            
            num_features = max(self.min_features, min(self.max_features, num_features))
            
            mask = np.zeros(len(self.feature_names), dtype=bool)
            selected_indices = np.random.choice(
                len(self.feature_names), 
                size=num_features, 
                replace=False
            )
            mask[selected_indices] = True
            population.append(mask)
        
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
        Enhanced fitness function with multiple objectives and stability
        
        Parameters:
        -----------
        individual : numpy array
            Binary mask of selected features
        
        Returns:
        --------
        float
            Fitness score combining accuracy, stability, and complexity
        """
        # Check if any features are selected
        if not np.any(individual):
            return 0.0
        
        if np.sum(individual) < self.min_features:
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
            
            # Main fitness components
            mean_accuracy = cv_scores.mean()
            std_accuracy = cv_scores.std()
            
            # Stability bonus (lower std is better)
            stability_bonus = 0.1 * (1 / (1 + std_accuracy))
            
            # Complexity penalty (reduced to allow more features)
            num_features = np.sum(individual)
            complexity_ratio = num_features / len(individual)
            complexity_penalty = 0.01 * (complexity_ratio ** 1.5)  # Reduced penalty, less aggressive curve
            
            # Feature diversity bonus (check for correlated features)
            diversity_bonus = 0.0
            if num_features > 5:  # Only for larger feature sets
                try:
                    correlation_matrix = np.corrcoef(X_selected.T)
                    # Penalize highly correlated features
                    high_corr = np.sum(np.abs(correlation_matrix) > 0.8) - num_features  # Subtract diagonal
                    diversity_bonus = -0.01 * high_corr / (num_features * (num_features - 1))
                except:
                    diversity_bonus = 0.0
            
            # Combined fitness
            fitness = mean_accuracy + stability_bonus - complexity_penalty + diversity_bonus
            
            # Ensure fitness is positive
            fitness = max(0.0, fitness)
            
        except Exception as e:
            print(f"Error in fitness evaluation: {str(e)}")
            fitness = 0.0
        
        return fitness
    
    def evaluate_population(self):
        """Evaluate fitness for entire population"""
        fitness_scores = []
        for individual in self.population:
            fitness = self.fitness_function(individual)
            fitness_scores.append(fitness)
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
        """Enhanced parent selection using rank-based tournament selection"""
        parents = []
        
        # Rank individuals by fitness
        ranked_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        ranks = np.empty_like(ranked_indices)
        ranks[ranked_indices] = np.arange(len(ranked_indices))
        
        for _ in range(self.pop_size):
            # Adaptive tournament size based on diversity
            diversity = self.calculate_population_diversity()
            if diversity < self.diversity_threshold:
                tournament_size = max(2, min(5, self.pop_size // 4))  # Larger tournament for low diversity
            else:
                tournament_size = 3  # Standard tournament size
            
            # Tournament selection based on ranks
            tournament_indices = np.random.choice(len(self.population), size=tournament_size, replace=False)
            tournament_ranks = [ranks[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_ranks)]  # Best rank (lowest number)
            parents.append(self.population[winner_idx])
        
        return parents
    
    def crossover(self, parent1, parent2):
        """Enhanced crossover with multiple strategies"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        # Choose crossover strategy
        strategy = np.random.choice(['single_point', 'two_point', 'uniform'])
        
        if strategy == 'single_point':
            crossover_point = np.random.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
        elif strategy == 'two_point':
            point1, point2 = sorted(np.random.choice(len(parent1), size=2, replace=False))
            child1 = parent1.copy()
            child2 = parent2.copy()
            child1[point1:point2] = parent2[point1:point2]
            child2[point1:point2] = parent1[point1:point2]
        
        else:  # uniform
            mask = np.random.random(len(parent1)) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        
        # Ensure minimum number of features
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
        """Enhanced mutation with adaptive strategies and feature count management"""
        mutated = individual.copy()
        current_features = np.sum(mutated)
        
        # Choose mutation strategy
        diversity = self.calculate_population_diversity() if hasattr(self, 'population') else 0.5
        
        if diversity < self.diversity_threshold:
            # Aggressive mutation for low diversity
            mutation_rate = min(0.3, self.mutation_rate * 1.5)
        else:
            mutation_rate = self.mutation_rate
        
        # Bit-flip mutation
        for i in range(len(mutated)):
            if np.random.random() < mutation_rate:
                mutated[i] = not mutated[i]
        
        # Additional mutation strategies for low diversity
        if diversity < self.diversity_threshold * 0.5:
            # Block mutation: flip a random block of bits
            if np.random.random() < 0.3:
                block_size = np.random.randint(2, min(10, len(mutated) // 4))
                start_idx = np.random.randint(0, len(mutated) - block_size)
                for i in range(start_idx, start_idx + block_size):
                    mutated[i] = not mutated[i]
        
        # Feature count management: encourage exploration of different feature counts
        new_features = np.sum(mutated)
        if new_features == current_features and np.random.random() < 0.2:  # 20% chance to force change
            if current_features < (self.min_features + self.max_features) / 2:
                # Add features if we're below average
                zeros = np.where(mutated == 0)[0]
                if len(zeros) > 0:
                    num_to_add = min(np.random.randint(1, 4), len(zeros))
                    to_flip = np.random.choice(zeros, size=num_to_add, replace=False)
                    mutated[to_flip] = True
            else:
                # Remove features if we're above average
                ones = np.where(mutated == 1)[0]
                if len(ones) > self.min_features:
                    num_to_remove = min(np.random.randint(1, 3), len(ones) - self.min_features)
                    to_flip = np.random.choice(ones, size=num_to_remove, replace=False)
                    mutated[to_flip] = False
        
        # Ensure constraints
        self._ensure_constraints(mutated)
        
        return mutated
    
    def elitism(self, fitness_scores, new_population):
        """Enhanced elitism with diversity preservation"""
        if self.elite_size > 0:
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elites = [self.population[i].copy() for i in elite_indices]
            
            # Ensure elite diversity
            unique_elites = []
            for elite in elites:
                is_unique = True
                for existing in unique_elites:
                    # Check if too similar (Hamming distance < threshold)
                    similarity = np.sum(elite == existing) / len(elite)
                    if similarity > 0.9:  # 90% similarity threshold
                        is_unique = False
                        break
                if is_unique:
                    unique_elites.append(elite)
                if len(unique_elites) >= self.elite_size:
                    break
            
            # If not enough unique elites, fill with best available
            if len(unique_elites) < self.elite_size:
                remaining_indices = [i for i in elite_indices if not any(
                    np.array_equal(self.population[i], elite) for elite in unique_elites
                )]
                for idx in remaining_indices:
                    unique_elites.append(self.population[idx].copy())
                    if len(unique_elites) >= self.elite_size:
                        break
            
            # Replace worst individuals with unique elites
            worst_indices = np.argsort([self.fitness_function(ind) for ind in new_population])
            for i, elite in enumerate(unique_elites[:self.elite_size]):
                if i < len(worst_indices):
                    new_population[worst_indices[i]] = elite
        
        return new_population
    
    def evolve(self, generation):
        """Evolve the population for one generation with enhanced tracking"""
        # Update adaptive rates
        self.update_adaptive_rates(generation)
        
        # Calculate diversity
        diversity = self.calculate_population_diversity()
        self.diversity_history.append(diversity)
        
        fitness_scores = self.evaluate_population()
        
        # Update best individual and track stagnation
        best_idx = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[best_idx]
        
        if current_best_fitness > self.best_fitness:
            self.best_individual = self.population[best_idx].copy()
            self.best_fitness = current_best_fitness
            self.stagnation_count = 0  # Reset stagnation
            
            # Update best features
            selected_features = [self.feature_names[i] for i in range(len(self.best_individual)) 
                                if self.best_individual[i]]
            self.best_features = selected_features
        else:
            self.stagnation_count += 1
        
        # Track history
        self.fitness_history.append(current_best_fitness)
        self.num_features_history.append(np.sum(self.best_individual) if self.best_individual is not None else 0)
        
        # Check for feature count stagnation and encourage exploration
        if generation >= 3 and len(self.num_features_history) >= 3:
            recent_feature_counts = self.num_features_history[-3:]
            if len(set(recent_feature_counts)) == 1:  # Same feature count for 3 generations
                current_avg_features = np.mean([np.sum(ind) for ind in self.population])
                if current_avg_features < self.max_features * 0.7:  # Only if we're not near max
                    self._encourage_feature_exploration()
        
        # Diversity injection if needed
        if diversity < self.diversity_threshold * 0.5:
            self._inject_diversity()
        
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
    
    def _inject_diversity(self):
        """Inject diversity by replacing some individuals with random ones, including high-feature individuals"""
        num_to_replace = max(1, self.pop_size // 10)  # Replace 10% of population
        
        # Get worst individuals
        fitness_scores = self.evaluate_population()
        worst_indices = np.argsort(fitness_scores)[:num_to_replace]
        
        # Calculate current feature distribution
        current_feature_counts = [np.sum(ind) for ind in self.population]
        avg_features = np.mean(current_feature_counts)
        
        for i, idx in enumerate(worst_indices):
            # Create diverse individuals with different feature count strategies
            if i < num_to_replace // 2:
                # Half with high feature counts to encourage exploration
                num_features = np.random.randint(
                    max(self.min_features, int(avg_features)), 
                    self.max_features + 1
                )
            else:
                # Half with random feature counts
                num_features = np.random.randint(self.min_features, self.max_features + 1)
            
            mask = np.zeros(len(self.feature_names), dtype=bool)
            selected_indices = np.random.choice(
                len(self.feature_names), 
                size=num_features, 
                replace=False
            )
            mask[selected_indices] = True
            self.population[idx] = mask
    
    def _encourage_feature_exploration(self):
        """Encourage exploration of different feature counts when stagnated"""
        num_to_modify = max(1, self.pop_size // 5)  # Modify 20% of population
        
        # Get random individuals (not just worst ones)
        indices_to_modify = np.random.choice(self.pop_size, size=num_to_modify, replace=False)
        
        for idx in indices_to_modify:
            current_features = np.sum(self.population[idx])
            
            # Decide whether to add or remove features
            if current_features < self.max_features * 0.8:  # If below 80% of max, tend to add
                if np.random.random() < 0.7:  # 70% chance to add features
                    zeros = np.where(self.population[idx] == 0)[0]
                    if len(zeros) > 0:
                        num_to_add = min(
                            np.random.randint(1, 5),  # Add 1-4 features
                            len(zeros),
                            self.max_features - current_features
                        )
                        if num_to_add > 0:
                            to_add = np.random.choice(zeros, size=num_to_add, replace=False)
                            self.population[idx][to_add] = True
                else:
                    # 30% chance to remove features for diversity
                    ones = np.where(self.population[idx] == 1)[0]
                    if len(ones) > self.min_features:
                        num_to_remove = min(
                            np.random.randint(1, 3),  # Remove 1-2 features
                            len(ones) - self.min_features
                        )
                        if num_to_remove > 0:
                            to_remove = np.random.choice(ones, size=num_to_remove, replace=False)
                            self.population[idx][to_remove] = False
            else:
                # If we have many features, randomly add or remove
                if np.random.random() < 0.5:
                    # Add features
                    zeros = np.where(self.population[idx] == 0)[0]
                    if len(zeros) > 0 and current_features < self.max_features:
                        num_to_add = min(
                            np.random.randint(1, 3),
                            len(zeros),
                            self.max_features - current_features
                        )
                        if num_to_add > 0:
                            to_add = np.random.choice(zeros, size=num_to_add, replace=False)
                            self.population[idx][to_add] = True
    
    def run(self, verbose=True):
        """Run the enhanced genetic algorithm with early stopping"""
        # Initialize population
        self.initialize_population()
        
        if verbose:
            print(f"Starting Enhanced Genetic Algorithm with {self.pop_size} individuals...")
            print(f"Model type: {self.model_type}")
            print(f"Using {self.cv_folds}-fold cross-validation")
            print(f"Adaptive rates: {self.adaptive_rates}")
            print(f"Early stopping patience: {self.early_stopping_patience}")
        
        # Evolution loop with early stopping
        for generation in range(self.max_generations):
            self.evolve(generation)
            
            if verbose:
                num_features = np.sum(self.best_individual) if self.best_individual is not None else 0
                diversity = self.diversity_history[-1] if self.diversity_history else 0
                
                # Calculate population feature statistics
                pop_feature_counts = [np.sum(ind) for ind in self.population]
                avg_pop_features = np.mean(pop_feature_counts)
                max_pop_features = np.max(pop_feature_counts)
                min_pop_features = np.min(pop_feature_counts)
                
                print(f"Gen {generation:2d}: Fitness={self.best_fitness:.4f}, "
                      f"Best={num_features:2d}, Pop=[{min_pop_features:2d}-{avg_pop_features:.1f}-{max_pop_features:2d}], "
                      f"Div={diversity:.3f}, Mut={self.mutation_rate:.3f}")
                
                # Alert if feature counts have been stagnant
                if generation >= 3 and len(self.num_features_history) >= 3:
                    recent_counts = self.num_features_history[-3:]
                    if len(set(recent_counts)) == 1:
                        print(f"    → Feature count stagnant at {recent_counts[0]} for 3 generations")
            
            # Early stopping check
            if self.stagnation_count >= self.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at generation {generation} (stagnation: {self.stagnation_count})")
                break
        
        # Final results
        if verbose:
            print(f"\nEnhanced Genetic Algorithm completed!")
            print(f"Best fitness (enhanced validation score): {self.best_fitness:.4f}")
            print(f"Number of selected features: {len(self.best_features)}")
            print(f"Final diversity: {self.diversity_history[-1]:.3f}")
            print(f"Generations without improvement: {self.stagnation_count}")
            print(f"Selected features: {self.best_features[:15]}...")  # Show first 15
        
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
    
    def analyze_population_feature_counts(self):
        """Analyze the distribution of feature counts in the current population"""
        if not self.population:
            print("No population available. Run the algorithm first.")
            return
        
        feature_counts = [np.sum(individual) for individual in self.population]
        
        print(f"\nPopulation Feature Count Analysis:")
        print(f"{'='*40}")
        print(f"Population size: {len(self.population)}")
        print(f"Feature count range: {min(feature_counts)} - {max(feature_counts)}")
        print(f"Average features: {np.mean(feature_counts):.1f}")
        print(f"Median features: {np.median(feature_counts):.1f}")
        print(f"Standard deviation: {np.std(feature_counts):.1f}")
        
        # Feature count distribution
        from collections import Counter
        count_distribution = Counter(feature_counts)
        print(f"\nFeature Count Distribution:")
        for count in sorted(count_distribution.keys()):
            percentage = count_distribution[count] / len(self.population) * 100
            bar = "█" * int(percentage / 5)  # Scale bar
            print(f"  {count:2d} features: {count_distribution[count]:2d} individuals ({percentage:4.1f}%) {bar}")
        
        # Check if we're exploring the full range
        range_explored = max(feature_counts) - min(feature_counts)
        max_possible_range = self.max_features - self.min_features
        exploration_ratio = range_explored / max_possible_range if max_possible_range > 0 else 1.0
        
        print(f"\nRange Exploration:")
        print(f"  Explored range: {range_explored} out of {max_possible_range} possible")
        print(f"  Exploration ratio: {exploration_ratio:.1%}")
        
        if exploration_ratio < 0.5:
            print(f"  ⚠️  Warning: Low feature count exploration! Consider increasing diversity.")
        
        return {
            'feature_counts': feature_counts,
            'distribution': dict(count_distribution),
            'stats': {
                'mean': np.mean(feature_counts),
                'median': np.median(feature_counts),
                'std': np.std(feature_counts),
                'min': min(feature_counts),
                'max': max(feature_counts)
            },
            'exploration_ratio': exploration_ratio
        }
    
    def get_selection_stability_report(self, num_runs=5):
        """Run the GA multiple times to assess feature selection stability"""
        print(f"Running stability analysis with {num_runs} independent runs...")
        
        feature_selection_counts = {name: 0 for name in self.feature_names}
        fitness_scores = []
        all_selected_features = []
        
        original_seed = self.random_state
        
        for run in range(num_runs):
            # Use different random seed for each run
            self.random_state = original_seed + run * 100
            np.random.seed(self.random_state)
            
            # Reset tracking variables
            self.population = None
            self.best_individual = None
            self.best_fitness = -np.inf
            self.fitness_history = []
            self.best_features = []
            self.num_features_history = []
            self.diversity_history = []
            self.stagnation_count = 0
            
            # Run GA
            _, fitness, features = self.run(verbose=False)
            
            # Collect results
            fitness_scores.append(fitness)
            all_selected_features.append(features)
            for feature in features:
                feature_selection_counts[feature] += 1
            
            print(f"Run {run+1}: Fitness={fitness:.4f}, Features={len(features)}")
        
        # Restore original seed
        self.random_state = original_seed
        
        # Analysis
        print(f"\nStability Analysis Results:")
        print(f"{'='*50}")
        print(f"Average fitness: {np.mean(fitness_scores):.4f} ± {np.std(fitness_scores):.4f}")
        print(f"Best fitness: {np.max(fitness_scores):.4f}")
        print(f"Worst fitness: {np.min(fitness_scores):.4f}")
        print(f"Fitness coefficient of variation: {np.std(fitness_scores)/np.mean(fitness_scores)*100:.1f}%")
        
        # Most consistently selected features
        consistent_features = {k: v for k, v in feature_selection_counts.items() if v >= num_runs // 2}
        print(f"\nMost Consistently Selected Features (>={num_runs//2} times):")
        for feature, count in sorted(consistent_features.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {count}/{num_runs} ({count/num_runs*100:.1f}%)")
        
        return {
            'fitness_scores': fitness_scores,
            'feature_selection_counts': feature_selection_counts,
            'consistent_features': consistent_features,
            'all_selected_features': all_selected_features
        }


# If run as main script
if __name__ == "__main__":
    print("MH_Feature_Selection.py - Simple Genetic Algorithm for Financial Feature Selection")
    print("This module should be imported, not run directly.")
    print("Example usage:")
    print("from MH_Feature_Selection import GeneticAlgorithm")
    print("ga = GeneticAlgorithm(X_train, y_train, feature_names)")
    print("best_mask, best_fitness, selected_features = ga.run()")
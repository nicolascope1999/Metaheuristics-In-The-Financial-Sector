#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MH_Feature_Selection.py - Simple Genetic Algorithm for Financial Feature Selection

This module implements a basic Genetic Algorithm (GA) for selecting optimal features
from financial time series data.

Author: Nicolas
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
                 pop_size=20, crossover_rate=0.8, mutation_rate=0.1, elite_size=2,
                 max_generations=10, min_features=5, max_features=30, cv_folds=3,
                 random_state=42):
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
        pop_size : int, default=20
            Population size
        crossover_rate : float, default=0.8
            Crossover rate
        mutation_rate : float, default=0.1
            Mutation rate
        elite_size : int, default=2
            Number of elite individuals to preserve
        max_generations : int, default=10
            Maximum number of generations
        min_features : int, default=5
            Minimum number of features to use
        max_features : int, default=30
            Maximum number of features to use
        cv_folds : int, default=3
            Number of cross-validation folds
        random_state : int, default=42
            Random seed for reproducibility
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
        
        # Initialize tracking variables
        self.population = None
        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        self.best_features = []
        self.num_features_history = []
        
        # Set up cross-validation
        self.cv = TimeSeriesSplit(n_splits=cv_folds)
        
        np.random.seed(random_state)
    
    def initialize_population(self):
        """Initialize random population of binary feature masks"""
        population = []
        for _ in range(self.pop_size):
            # Decide how many features to use for this individual
            num_features = np.random.randint(self.min_features, self.max_features + 1)
            
            # Create a binary mask with num_features 1s and the rest 0s
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
        Calculate fitness based on cross-validation accuracy
        
        Parameters:
        -----------
        individual : numpy array
            Binary mask of selected features
        
        Returns:
        --------
        float
            Fitness score (validation accuracy)
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
            accuracy = cv_scores.mean()
            
            # Small penalty for complexity
            num_features = np.sum(individual)
            complexity_penalty = 0.001 * num_features / len(individual)
            
            fitness = accuracy - complexity_penalty
            
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
    
    def select_parents(self, fitness_scores):
        """Select parents using tournament selection"""
        parents = []
        for _ in range(self.pop_size):
            # Tournament selection (k=3)
            tournament_indices = np.random.choice(len(self.population), size=3, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        return parents
    
    def crossover(self, parent1, parent2):
        """Perform single-point crossover between parents"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = np.random.randint(1, len(parent1))
        
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        
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
        """Apply bit-flip mutation"""
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] = not mutated[i]
        
        # Ensure constraints
        self._ensure_constraints(mutated)
        
        return mutated
    
    def elitism(self, fitness_scores, new_population):
        """Preserve elite individuals"""
        if self.elite_size > 0:
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elites = [self.population[i] for i in elite_indices]
            
            # Replace worst individuals with elites
            for i, elite in enumerate(elites):
                new_population[-(i+1)] = elite.copy()
        
        return new_population
    
    def evolve(self):
        """Evolve the population for one generation"""
        fitness_scores = self.evaluate_population()
        
        # Update best individual
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_individual = self.population[best_idx].copy()
            self.best_fitness = fitness_scores[best_idx]
            
            # Update best features
            selected_features = [self.feature_names[i] for i in range(len(self.best_individual)) 
                                if self.best_individual[i]]
            self.best_features = selected_features
        
        # Track history
        self.fitness_history.append(np.max(fitness_scores))
        self.num_features_history.append(np.sum(self.best_individual))
        
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
        """Run the genetic algorithm"""
        # Initialize population
        self.initialize_population()
        
        if verbose:
            print(f"Starting Genetic Algorithm with {self.pop_size} individuals...")
            print(f"Model type: {self.model_type}")
            print(f"Using {self.cv_folds}-fold cross-validation")
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.evolve()
            
            if verbose:
                num_features = np.sum(self.best_individual)
                print(f"Generation {generation}: Best fitness = {self.best_fitness:.4f}, "
                      f"Features = {num_features}")
        
        # Final results
        if verbose:
            print(f"\nGenetic Algorithm completed!")
            print(f"Best fitness (validation accuracy): {self.best_fitness:.4f}")
            print(f"Number of selected features: {len(self.best_features)}")
            print(f"Selected features: {self.best_features[:10]}...")  # Show first 10
        
        return self.best_individual, self.best_fitness, self.best_features
    
    def plot_convergence(self):
        """Plot convergence history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.fitness_history, marker='o')
        plt.title('Fitness History')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Validation Accuracy)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.num_features_history, marker='o', color='orange')
        plt.title('Number of Features')
        plt.xlabel('Generation')
        plt.ylabel('Number of Features')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
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


# If run as main script
if __name__ == "__main__":
    print("MH_Feature_Selection.py - Simple Genetic Algorithm for Financial Feature Selection")
    print("This module should be imported, not run directly.")
    print("Example usage:")
    print("from MH_Feature_Selection import GeneticAlgorithm")
    print("ga = GeneticAlgorithm(X_train, y_train, feature_names)")
    print("best_mask, best_fitness, selected_features = ga.run()")

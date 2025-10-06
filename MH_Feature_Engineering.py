"""
Metaheuristic Feature Engineering using Genetic Algorithms
=========================================================

This module implements Genetic Algorithm-based feature engineering for financial markets.
It creates new technical indicators by optimizing their parameters to maximize predictive power
while maintaining time-series integrity and cross-validation stability.

Features Implemented:
- Adaptive Moving Average (AMA): Volatility-adjusted moving average
- Fractal Dimension Indicator (FDI): Market roughness measurement

Author: Financial ML Metaheuristics Project
Date: September 2025
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import random
from typing import List, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineeringGA:
    """
    Genetic Algorithm for optimizing financial feature engineering parameters.
    
    This class implements a GA specifically designed for creating and optimizing
    new financial features while respecting time-series constraints.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 target: pd.Series, 
                 feature_type: str = 'AMA',
                 pop_size: int = 50,
                 max_generations: int = 30,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.2,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize the Feature Engineering GA.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The financial data containing OHLCV and other features
        target : pd.Series
            The target variable (e.g., Bullish/Bearish)
        feature_type : str
            Type of feature to engineer ('AMA' or 'FDI')
        pop_size : int
            Population size for GA
        max_generations : int
            Maximum number of generations
        crossover_rate : float
            Probability of crossover
        mutation_rate : float
            Probability of mutation
        cv_folds : int
            Number of time-series cross-validation folds
        random_state : int
            Random seed for reproducibility
        """
        self.data = data.copy()
        self.target = target.copy()
        self.feature_type = feature_type
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Set random seeds
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Define parameter bounds based on feature type
        self._set_parameter_bounds()
        
        # Initialize population
        self.population = self._initialize_population()
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
    def _set_parameter_bounds(self):
        """Set parameter bounds based on feature type."""
        if self.feature_type == 'AMA':
            # AMA parameters: [smoothing_factor, volatility_window, fast_period, slow_period]
            self.param_bounds = [
                (0.01, 0.9),   # smoothing_factor
                (5, 50),       # volatility_window
                (2, 10),       # fast_period
                (10, 50)       # slow_period
            ]
            self.param_types = ['float', 'int', 'int', 'int']
        elif self.feature_type == 'FDI':
            # FDI parameters: [window_size, high_low_factor, scaling_factor]
            self.param_bounds = [
                (10, 100),     # window_size
                (0.1, 2.0),    # high_low_factor
                (0.5, 2.0)     # scaling_factor
            ]
            self.param_types = ['int', 'float', 'float']
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def _initialize_population(self) -> List[List]:
        """Initialize random population within parameter bounds."""
        population = []
        for _ in range(self.pop_size):
            individual = []
            for i, (low, high) in enumerate(self.param_bounds):
                if self.param_types[i] == 'int':
                    individual.append(random.randint(int(low), int(high)))
                else:
                    individual.append(random.uniform(low, high))
            population.append(individual)
        return population
    
    def _create_ama_feature(self, params: List) -> pd.Series:
        """
        Create Adaptive Moving Average feature.
        
        AMA adjusts its smoothing based on market volatility and directional movement.
        
        Parameters:
        -----------
        params : List
            [smoothing_factor, volatility_window, fast_period, slow_period]
        """
        smoothing_factor, volatility_window, fast_period, slow_period = params
        volatility_window = int(volatility_window)
        fast_period = int(fast_period)
        slow_period = int(slow_period)
        
        close = self.data['Close'].values
        n = len(close)
        ama = np.zeros(n)
        ama[0] = close[0]
        
        for i in range(1, n):
            # Calculate efficiency ratio (directional movement / total movement)
            if i >= volatility_window:
                direction = abs(close[i] - close[i - volatility_window])
                total_movement = sum(abs(close[j] - close[j-1]) for j in range(i - volatility_window + 1, i + 1))
                
                if total_movement > 0:
                    efficiency_ratio = direction / total_movement
                else:
                    efficiency_ratio = 0
                
                # Calculate smoothing constant
                fast_sc = 2.0 / (fast_period + 1)
                slow_sc = 2.0 / (slow_period + 1)
                smooth_const = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2
                
                # Apply adaptive smoothing
                ama[i] = ama[i-1] + smooth_const * (close[i] - ama[i-1])
            else:
                # Use simple moving average for initial values
                ama[i] = np.mean(close[:i+1])
        
        return pd.Series(ama, index=self.data.index)
    
    def _create_fdi_feature(self, params: List) -> pd.Series:
        """
        Create Fractal Dimension Indicator feature.
        
        FDI measures the "roughness" of price movements to distinguish
        between trending and ranging markets.
        
        Parameters:
        -----------
        params : List
            [window_size, high_low_factor, scaling_factor]
        """
        window_size, high_low_factor, scaling_factor = params
        window_size = int(window_size)
        
        high = self.data['High'].values
        low = self.data['Low'].values
        close = self.data['Close'].values
        n = len(close)
        fdi = np.zeros(n)
        
        for i in range(window_size, n):
            # Calculate price changes and distances
            price_changes = close[i-window_size:i]
            
            # Calculate total distance (sum of absolute price changes)
            total_distance = sum(abs(price_changes[j] - price_changes[j-1]) 
                               for j in range(1, len(price_changes)))
            
            # Calculate high-low range factor
            hl_range = sum(high[i-window_size:i] - low[i-window_size:i])
            range_factor = hl_range * high_low_factor
            
            # Calculate fractal dimension
            if total_distance > 0 and range_factor > 0:
                n_segments = window_size - 1
                dimension = np.log(n_segments) / (np.log(n_segments) + np.log(total_distance / range_factor))
                fdi[i] = dimension * scaling_factor
            else:
                fdi[i] = 1.0  # Default value
        
        # Fill initial values
        fdi[:window_size] = fdi[window_size] if window_size < n else 1.0
        
        return pd.Series(fdi, index=self.data.index)
    
    def _create_feature(self, params: List) -> pd.Series:
        """Create feature based on feature type."""
        if self.feature_type == 'AMA':
            return self._create_ama_feature(params)
        elif self.feature_type == 'FDI':
            return self._create_fdi_feature(params)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")
    
    def _evaluate_fitness(self, params: List) -> float:
        """
        Evaluate fitness of feature parameters using time-series cross-validation.
        
        Fitness combines:
        - Mean accuracy across CV folds
        - Stability (1 - std of CV scores)
        - Feature variance (to avoid constant features)
        """
        try:
            # Create the feature
            feature = self._create_feature(params)
            
            # Check for invalid values
            if feature.isna().any() or np.isinf(feature).any():
                return -1.0
            
            # Check for constant feature (low variance)
            if feature.var() < 1e-6:
                return -1.0
            
            # Prepare feature for cross-validation
            X = feature.values.reshape(-1, 1)
            y = self.target.values
            
            # Remove any NaN values
            valid_idx = ~(np.isnan(X.flatten()) | np.isnan(y))
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 50:  # Need minimum samples
                return -1.0
            
            # Time-series cross-validation
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            cv_scores = []
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train simple model
                model = LogisticRegression(random_state=self.random_state, max_iter=1000)
                model.fit(X_train_scaled, y_train)
                
                # Predict and score
                y_pred = model.predict(X_test_scaled)
                score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cv_scores.append(score)
            
            # Calculate fitness components
            mean_score = np.mean(cv_scores)
            cv_stability = max(0.0, 1.0 - np.std(cv_scores))
            feature_variance = min(1.0, feature.var())  # Normalize variance component
            
            # Combined fitness with weights
            fitness = (0.6 * mean_score + 0.3 * cv_stability + 0.1 * feature_variance)
            
            return fitness
            
        except Exception as e:
            # Return poor fitness for invalid parameters
            return -1.0
    
    def _tournament_selection(self, tournament_size: int = 3) -> List:
        """Select individual using tournament selection."""
        tournament = random.sample(self.population, tournament_size)
        tournament_fitness = [self._evaluate_fitness(ind) for ind in tournament]
        winner_idx = np.argmax(tournament_fitness)
        return tournament[winner_idx].copy()
    
    def _crossover(self, parent1: List, parent2: List) -> Tuple[List, List]:
        """Perform uniform crossover between two parents."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1[i], child2[i] = child2[i], child1[i]
        
        return child1, child2
    
    def _mutate(self, individual: List) -> List:
        """Mutate individual parameters within bounds."""
        mutated = individual.copy()
        
        for i in range(len(individual)):
            if random.random() < self.mutation_rate:
                low, high = self.param_bounds[i]
                if self.param_types[i] == 'int':
                    mutated[i] = random.randint(int(low), int(high))
                else:
                    # Gaussian mutation with bounds
                    sigma = (high - low) * 0.1  # 10% of range as std
                    new_val = individual[i] + random.gauss(0, sigma)
                    mutated[i] = max(low, min(high, new_val))
        
        return mutated
    
    def run(self, verbose: bool = True) -> Tuple[List, float, pd.Series]:
        """
        Run the Genetic Algorithm for feature engineering.
        
        Returns:
        --------
        Tuple[List, float, pd.Series]
            Best parameters, best fitness, and the engineered feature
        """
        if verbose:
            print(f"Starting GA Feature Engineering for {self.feature_type}")
            print(f"Population: {self.pop_size}, Generations: {self.max_generations}")
            print("-" * 60)
        
        for generation in range(self.max_generations):
            # Evaluate population fitness
            fitness_scores = [self._evaluate_fitness(ind) for ind in self.population]
            
            # Track best individual
            max_fitness_idx = np.argmax(fitness_scores)
            max_fitness = fitness_scores[max_fitness_idx]
            
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.best_individual = self.population[max_fitness_idx].copy()
            
            self.fitness_history.append(max_fitness)
            
            if verbose and generation % 5 == 0:
                avg_fitness = np.mean(fitness_scores)
                print(f"Generation {generation:3d}: Best={max_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Create next generation
            new_population = []
            
            # Elitism: keep best individuals
            elite_size = max(1, self.pop_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(self.population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.pop_size:
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            self.population = new_population[:self.pop_size]
        
        # Create final feature with best parameters
        best_feature = self._create_feature(self.best_individual)
        
        if verbose:
            print("-" * 60)
            print(f"GA completed! Best fitness: {self.best_fitness:.4f}")
            print(f"Best parameters: {self.best_individual}")
        
        return self.best_individual, self.best_fitness, best_feature


def engineer_adaptive_moving_average(data: pd.DataFrame, 
                                   target: pd.Series, 
                                   **ga_params) -> Tuple[pd.Series, dict]:
    """
    Engineer Adaptive Moving Average feature using GA optimization.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Financial data with OHLCV columns
    target : pd.Series
        Target variable
    **ga_params : dict
        Additional GA parameters
    
    Returns:
    --------
    Tuple[pd.Series, dict]
        Engineered AMA feature and optimization details
    """
    ga = FeatureEngineeringGA(data, target, feature_type='AMA', **ga_params)
    best_params, best_fitness, ama_feature = ga.run()
    
    optimization_details = {
        'feature_type': 'AMA',
        'best_parameters': best_params,
        'best_fitness': best_fitness,
        'fitness_history': ga.fitness_history,
        'parameter_names': ['smoothing_factor', 'volatility_window', 'fast_period', 'slow_period']
    }
    
    return ama_feature, optimization_details


def engineer_fractal_dimension_indicator(data: pd.DataFrame, 
                                        target: pd.Series, 
                                        **ga_params) -> Tuple[pd.Series, dict]:
    """
    Engineer Fractal Dimension Indicator feature using GA optimization.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Financial data with OHLCV columns
    target : pd.Series
        Target variable
    **ga_params : dict
        Additional GA parameters
    
    Returns:
    --------
    Tuple[pd.Series, dict]
        Engineered FDI feature and optimization details
    """
    ga = FeatureEngineeringGA(data, target, feature_type='FDI', **ga_params)
    best_params, best_fitness, fdi_feature = ga.run()
    
    optimization_details = {
        'feature_type': 'FDI',
        'best_parameters': best_params,
        'best_fitness': best_fitness,
        'fitness_history': ga.fitness_history,
        'parameter_names': ['window_size', 'high_low_factor', 'scaling_factor']
    }
    
    return fdi_feature, optimization_details


if __name__ == "__main__":
    # Example usage
    print("Metaheuristic Feature Engineering Module")
    print("This module provides GA-based feature engineering for financial data.")
    print("Import this module in your notebook to use the feature engineering functions.")
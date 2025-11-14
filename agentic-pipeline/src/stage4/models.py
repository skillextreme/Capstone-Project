"""
Model Training Module

Handles training of machine learning models:
- Linear Regression (Ridge, Lasso)
- Gradient Boosting (XGBoost)
- Random Forest (optional)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains and evaluates machine learning models.

    Example:
        >>> trainer = ModelTrainer(config={'models': ['linear_regression', 'xgboost']})
        >>> results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Model Trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = {
            'models': {
                'linear_regression': {
                    'enabled': True,
                    'regularization': 'ridge',
                    'alpha': 1.0
                },
                'gradient_boosting': {
                    'enabled': True,
                    'library': 'xgboost',
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1
                },
                'random_forest': {
                    'enabled': False,
                    'n_estimators': 100,
                    'max_depth': 10
                }
            },
            'cross_validation': {
                'enabled': True,
                'cv_folds': 5
            },
            'random_seed': 42
        }

        if config:
            # Deep merge config
            for key, value in config.items():
                if isinstance(value, dict) and key in self.config:
                    self.config[key].update(value)
                else:
                    self.config[key] = value

        self.models = {}
        self.results = {}

        logger.info("Initialized Model Trainer")

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all enabled models and evaluate on test set.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with model results
        """
        logger.info(f"Training models on {len(X_train)} samples, {len(X_train.columns)} features")

        results = {}

        # Train linear regression
        if self.config['models']['linear_regression']['enabled']:
            results['linear_regression'] = self._train_linear_regression(
                X_train, y_train, X_test, y_test
            )

        # Train gradient boosting
        if self.config['models']['gradient_boosting']['enabled']:
            results['gradient_boosting'] = self._train_gradient_boosting(
                X_train, y_train, X_test, y_test
            )

        # Train random forest
        if self.config['models']['random_forest']['enabled']:
            results['random_forest'] = self._train_random_forest(
                X_train, y_train, X_test, y_test
            )

        self.results = results

        logger.info(f"Trained {len(results)} models")

        return results

    def _train_linear_regression(
        self,
        X_train, y_train, X_test, y_test
    ) -> Dict[str, Any]:
        """Train linear regression model."""
        logger.info("Training Linear Regression...")

        model_config = self.config['models']['linear_regression']
        reg_type = model_config['regularization']
        alpha = model_config['alpha']

        # Choose regularization
        if reg_type == 'ridge':
            model = Ridge(alpha=alpha, random_state=self.config['random_seed'])
        elif reg_type == 'lasso':
            model = Lasso(alpha=alpha, random_state=self.config['random_seed'])
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        metrics = {
            'train': self._calculate_metrics(y_train, y_pred_train),
            'test': self._calculate_metrics(y_test, y_pred_test)
        }

        # Cross-validation
        if self.config['cross_validation']['enabled']:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config['cross_validation']['cv_folds'],
                scoring='neg_mean_absolute_error'
            )
            metrics['cv_mae'] = -cv_scores.mean()
            metrics['cv_mae_std'] = cv_scores.std()

        # Feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)

        # Store model
        self.models['linear_regression'] = model

        logger.info(f"  Train MAE: {metrics['train']['mae']:.4f}, Test MAE: {metrics['test']['mae']:.4f}")

        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'feature_importance': feature_importance.to_dict('records')
        }

    def _train_gradient_boosting(
        self,
        X_train, y_train, X_test, y_test
    ) -> Dict[str, Any]:
        """Train gradient boosting model."""
        logger.info("Training Gradient Boosting (XGBoost)...")

        model_config = self.config['models']['gradient_boosting']

        # Create XGBoost model
        model = xgb.XGBRegressor(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            learning_rate=model_config['learning_rate'],
            random_state=self.config['random_seed'],
            verbosity=0
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        metrics = {
            'train': self._calculate_metrics(y_train, y_pred_train),
            'test': self._calculate_metrics(y_test, y_pred_test)
        }

        # Cross-validation
        if self.config['cross_validation']['enabled']:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=self.config['cross_validation']['cv_folds'],
                scoring='neg_mean_absolute_error'
            )
            metrics['cv_mae'] = -cv_scores.mean()
            metrics['cv_mae_std'] = cv_scores.std()

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store model
        self.models['gradient_boosting'] = model

        logger.info(f"  Train MAE: {metrics['train']['mae']:.4f}, Test MAE: {metrics['test']['mae']:.4f}")

        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'feature_importance': feature_importance.to_dict('records')
        }

    def _train_random_forest(
        self,
        X_train, y_train, X_test, y_test
    ) -> Dict[str, Any]:
        """Train random forest model."""
        logger.info("Training Random Forest...")

        model_config = self.config['models']['random_forest']

        model = RandomForestRegressor(
            n_estimators=model_config['n_estimators'],
            max_depth=model_config['max_depth'],
            random_state=self.config['random_seed'],
            n_jobs=-1
        )

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        metrics = {
            'train': self._calculate_metrics(y_train, y_pred_train),
            'test': self._calculate_metrics(y_test, y_pred_test)
        }

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Store model
        self.models['random_forest'] = model

        logger.info(f"  Train MAE: {metrics['train']['mae']:.4f}, Test MAE: {metrics['test']['mae']:.4f}")

        return {
            'model': model,
            'metrics': metrics,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'feature_importance': feature_importance.to_dict('records')
        }

    def _calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate regression metrics."""
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # MAPE (avoid division by zero)
        mask = y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = np.nan

        return {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }

    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best model based on test MAE.

        Returns:
            Tuple of (model_name, model_results)
        """
        if not self.results:
            return None, None

        best_model_name = min(
            self.results.keys(),
            key=lambda m: self.results[m]['metrics']['test']['mae']
        )

        return best_model_name, self.results[best_model_name]

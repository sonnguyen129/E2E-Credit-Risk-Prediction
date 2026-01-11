"""
Focal Loss Implementation for Binary Classification

Focal Loss được giới thiệu trong paper "Focal Loss for Dense Object Detection" (Lin et al., 2017)
để xử lý class imbalance bằng cách giảm weight của các easy examples và tập trung vào hard examples.

Focal Loss = -α * (1-p)^γ * log(p)

Với:
- α: balancing factor cho class weights  
- γ (gamma): focusing parameter (γ >= 0). Khi γ=0, focal loss = cross entropy
- p: predicted probability
"""

import numpy as np


def focal_loss_binary(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Tính Focal Loss cho binary classification
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True labels (0 or 1)
    y_pred : array-like, shape (n_samples,)
        Predicted probabilities for positive class
    alpha : float, default=0.25
        Balancing factor. Thường dùng α=0.25 cho positive class
    gamma : float, default=2.0
        Focusing parameter. Thường dùng γ=2.0
        
    Returns:
    --------
    loss : float
        Average focal loss
    """
    # Clip predictions để tránh log(0)
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Tính focal loss
    # For positive class (y=1): -α * (1-p)^γ * log(p)
    # For negative class (y=0): -(1-α) * p^γ * log(1-p)
    focal_loss_pos = -alpha * np.power(1 - y_pred, gamma) * np.log(y_pred)
    focal_loss_neg = -(1 - alpha) * np.power(y_pred, gamma) * np.log(1 - y_pred)
    
    # Chọn loss dựa trên true label
    loss = np.where(y_true == 1, focal_loss_pos, focal_loss_neg)
    
    return np.mean(loss)


def focal_loss_lgb(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss cho LightGBM custom objective function
    
    LightGBM yêu cầu trả về gradient và hessian
    
    Parameters:
    -----------
    y_true : array-like, shape (n_samples,)
        True labels (0 or 1)
    y_pred : array-like, shape (n_samples,)
        Raw predictions (logits, not probabilities)
    alpha : float, default=0.25
        Balancing factor
    gamma : float, default=2.0
        Focusing parameter
        
    Returns:
    --------
    grad : array, shape (n_samples,)
        First order gradient
    hess : array, shape (n_samples,)
        Second order gradient (hessian)
    """
    # Convert logits to probabilities
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Clip probabilities
    epsilon = 1e-7
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    
    # Tính gradient và hessian
    # Gradient: ∂L/∂z where z is logit
    grad = np.where(
        y_true == 1,
        -alpha * (gamma * np.power(1 - y_pred_prob, gamma - 1) * np.log(y_pred_prob) + 
                  np.power(1 - y_pred_prob, gamma)) * y_pred_prob * (1 - y_pred_prob),
        (1 - alpha) * (gamma * np.power(y_pred_prob, gamma - 1) * np.log(1 - y_pred_prob) + 
                       np.power(y_pred_prob, gamma)) * y_pred_prob * (1 - y_pred_prob)
    )
    
    # Hessian (approximation)
    # Để đơn giản hóa, sử dụng approximation: p*(1-p)
    hess = y_pred_prob * (1 - y_pred_prob)
    
    return grad, hess


def focal_loss_xgb(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss cho XGBoost custom objective function
    
    Simplified approach: Use weighted cross-entropy as approximation
    This is more stable than full focal loss gradient
    
    Parameters:
    -----------
    y_pred : array-like, shape (n_samples,)
        Raw predictions (logits)
    dtrain : xgboost.DMatrix
        Training data with labels
    alpha : float, default=0.25
        Balancing factor for positive class
    gamma : float, default=2.0
        Focusing parameter (not used in simplified version for stability)
        
    Returns:
    --------
    grad : array, shape (n_samples,)
        First order gradient
    hess : array, shape (n_samples,)
        Second order gradient (hessian)
    """
    # Get true labels
    y_true = dtrain.get_label()
    
    # Convert logits to probabilities using sigmoid
    y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred))
    
    # Clip probabilities to avoid log(0)
    epsilon = 1e-7
    y_pred_prob = np.clip(y_pred_prob, epsilon, 1 - epsilon)
    
    # Compute modulating factor (1-p)^gamma for positive, p^gamma for negative
    # This downweights easy examples
    modulating_pos = np.power(1 - y_pred_prob, gamma)
    modulating_neg = np.power(y_pred_prob, gamma)
    
    # Gradient of weighted cross-entropy with focal modulation
    # For y=1: grad = α * (p - 1) * modulating_factor
    # For y=0: grad = (1-α) * p * modulating_factor
    grad = np.where(
        y_true == 1,
        alpha * modulating_pos * (y_pred_prob - 1),
        (1 - alpha) * modulating_neg * y_pred_prob
    )
    
    # Hessian: second derivative approximation
    # Use p*(1-p) as base, scaled by modulation
    hess = np.where(
        y_true == 1,
        alpha * modulating_pos * y_pred_prob * (1 - y_pred_prob),
        (1 - alpha) * modulating_neg * y_pred_prob * (1 - y_pred_prob)
    )
    
    # Add small constant to prevent division by zero
    hess = np.maximum(hess, 1e-7)
    
    return grad, hess


def focal_loss_lgb_eval(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss evaluation metric cho LightGBM
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted probabilities
    alpha : float, default=0.25
        Balancing factor
    gamma : float, default=2.0
        Focusing parameter
        
    Returns:
    --------
    metric_name : str
        Name of the metric
    metric_value : float
        Value of the metric
    is_higher_better : bool
        Whether higher value is better
    """
    loss = focal_loss_binary(y_true, y_pred, alpha, gamma)
    return 'focal_loss', loss, False  # Lower is better


def focal_loss_xgb_eval(y_pred, dtrain, alpha=0.25, gamma=2.0):
    """
    Focal Loss evaluation metric cho XGBoost
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted probabilities
    dtrain : xgboost.DMatrix
        Training data with labels
    alpha : float, default=0.25
        Balancing factor
    gamma : float, default=2.0
        Focusing parameter
        
    Returns:
    --------
    metric_name : str
        Name of the metric
    metric_value : float
        Value of the metric
    """
    y_true = dtrain.get_label()
    
    # Convert logits to probabilities if needed
    if y_pred.max() > 1 or y_pred.min() < 0:
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    
    loss = focal_loss_binary(y_true, y_pred, alpha, gamma)
    return 'focal_loss', loss


# Helper functions để tạo objective và eval functions với custom alpha/gamma
def create_focal_loss_lgb(alpha=0.25, gamma=2.0):
    """
    Tạo LightGBM focal loss objective function với custom alpha và gamma
    """
    def focal_loss_objective(y_true, y_pred):
        return focal_loss_lgb(y_true, y_pred, alpha, gamma)
    return focal_loss_objective


def create_focal_loss_lgb_eval(alpha=0.25, gamma=2.0):
    """
    Tạo LightGBM focal loss eval function với custom alpha và gamma
    """
    def focal_loss_eval(y_true, y_pred):
        return focal_loss_lgb_eval(y_true, y_pred, alpha, gamma)
    return focal_loss_eval


def create_focal_loss_xgb(alpha=0.25, gamma=2.0):
    """
    Tạo XGBoost focal loss objective function với custom alpha và gamma
    """
    def focal_loss_objective(y_pred, dtrain):
        return focal_loss_xgb(y_pred, dtrain, alpha, gamma)
    return focal_loss_objective


def create_focal_loss_xgb_eval(alpha=0.25, gamma=2.0):
    """
    Tạo XGBoost focal loss eval function với custom alpha và gamma
    """
    def focal_loss_eval(y_pred, dtrain):
        return focal_loss_xgb_eval(y_pred, dtrain, alpha, gamma)
    return focal_loss_eval

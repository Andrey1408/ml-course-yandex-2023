import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    
    eigenvector = np.random.rand(data.shape[0])
    
    for _ in range(num_steps):
        eigenvector = np.dot(data, eigenvector)
        
        norm = np.sqrt(np.sum(eigenvector ** 2))

        eigenvector /= norm

    eigenvalue = np.dot(np.dot(eigenvector.T, data), eigenvector)
    
    return  float(eigenvalue), eigenvector
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 11:46:06 2023

@author: ayala

This program computes eigenvectors & eigenvalues of a symmetric matrix using QR algorithm w/ shifts and 
QR Factorization using Householder method.

Preprocessing using Hessenberg algorithm for tridiagonalization of symmetric matrix.

"""

import numpy as np
from time import time 
    
def vec_norm(v):
    return np.sqrt(np.sum(np.power(v,2)))

def v_vec_norm(v):
    if (v.shape[0] > 1):
        return np.sqrt(np.power(v[0], 2) + np.power(v[1], 2))
    else:
        return np.sqrt(np.power(v[0],2))
    
def v_vec_dot(v):
    if (v.shape[0] > 1):
        return np.power(v[0], 2) + np.power(v[1], 2)
    else:
        return v[0]*v[0]
    
def hh_reflection_vector(x):
    v = np.copy(x)
    v[0] += np.sign(x[0])*v_vec_norm(x)
    return v, 2.0 / v_vec_dot(v)

def qr_factorization_householder(A):
    """
    Performs QR factorization of a matrix A using Householder reflections.
    
    Parameters
    ----------
    A : numpy.ndarray
        Input matrix of shape (n, n).
    
    Returns
    -------
    Q : numpy.ndarray
        Orthogonal matrix of shape (n, n) where Q^T * Q = I.
    R : numpy.ndarray
        Upper triangular matrix of shape (n, n) representing the factorization A = QR.
    """
    n = A.shape[0]
    Q = np.eye(n)
    R = np.copy(A)
    
    for j in range(n):
        v, c = hh_reflection_vector(R[j:, j])
        update_R(v, c, R[j:, j:]) # Apply Householder transformation to eliminate entries below the diagonal in the jth column
        update_Q(v, c, Q[:, j:])

    return Q, R

def update_Q(v, c, Q):
    if (v.shape[0]==1):
        Q[:0] -= c *Q[:0] * np.power(v[0],2)
        return
    
    dd2 = np.copy(Q[:, :2]) # copy only what's relevant: Q's first two columns.
    Q[:, 0] -= c * (dd2[:,0]*np.power(v[0],2)+dd2[:,1]*(v[0]*v[1]))
    Q[:, 1] -= c * (dd2[:,0]*(v[0]*v[1])+dd2[:,1]*np.power(v[1],2))
 
def update_R(v, c, R):
    if (R.shape[0]==1):
        # R[0,0] -= c*(np.power(v[0],2)*R[0,0])
        return
        
    dd2 = np.copy(R[:2, :]) # copy only what's relevant: R's first two rows.
    R[0, :] -= c * (np.power(v[0],2) * dd2[0,:] + (v[0]*v[1])*dd2[1,:])
    R[1, :] -= c * ((v[0]*v[1])*dd2[0,:] + np.power(v[1],2)*dd2[1,:]) # V
     
def hessenberg_form(A, epsilon=1e-6):
    """
    Computes the Hessenberg form of a symmetric matrix A using Householder reflections.
    
    Parameters
    ----------
    A : numpy.ndarray
        Symmetric matrix of shape (n, n).
    
    Returns
    -------
    H : numpy.ndarray
        Hessenberg form of matrix A.
    Q : numpy.ndarray
        Orthogonal matrix such that Q^T * A * Q = H.
    """
    n = A.shape[0]
    H = np.copy(A)
    Q = np.eye(n)

    for k in range(n - 2):
        x = H[k+1:, k]
        v = np.copy(x)
        v[0] += np.sign(x[0]) * vec_norm(x)
        v /= vec_norm(v)
        H[k+1:, k:] -= 2.0 * np.outer(v, v @ H[k+1:, k:])
        H[:, k+1:] -= 2.0 * np.outer(H[:, k+1:] @ v, v)
        Q[:, k+1:] -= 2.0 * np.outer(Q[:, k+1:] @ v, v)

    # Create the mask for tridiagonal elements
    mask = np.eye(n, dtype=bool) | np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool)
    
    # Assign zeros to the masked elements
    H[~mask] = np.where(np.abs(H[~mask]) < epsilon, 0, H[~mask])
    
    return H, Q

def my_eigen_recursive(A, epsilon=1e-6):
    """
    Computes eigenvalues and eigenvectors of matrix A using QR algorithm w/ shifts
    
    Parameters
    ----------
    A : symmetric matrix.
    epsilon : TYPE, optional
        DESCRIPTION. The default is 1e-6.
        
    Returns
    -------
    eigenvalues and eigenvectors of matrix A.
    """
    n = A.shape[0]
    I = np.eye(n)   # Identity matrix nxn
    Q = np.eye(n) 
    
    if (n==1): 
        return A[:,0], I

    eigenvectors = np.eye(n)
    u=0.0

    while np.min(np.abs(np.diag(A, 1))) > epsilon:        
        uI = u*I
        Q, R = qr_factorization_householder(A- uI)  # Apply QR factorization on shifted A
        A = R @ Q + uI                              # Shift back the result
        u=A[n-1,n-1]                                # Choose shift: shift = A(n,n)
        eigenvectors = eigenvectors @ Q             # Store eigenvectors for final result

    diag_arr_position = np.argmin(np.abs(np.diag(A, 1))) # get index where min value is in upper diag    

    upper_mat = A[:diag_arr_position+1,:diag_arr_position+1] # upper block
    low_mat = A[diag_arr_position+1:,diag_arr_position+1:] # lower block

    eigenvalues_upper, eigenvector_upper = my_eigen_recursive(upper_mat,epsilon)
    eigenvalues_lower, eigenvector_lower = my_eigen_recursive(low_mat,epsilon)
    
    eigenvalues = np.concatenate([eigenvalues_upper, eigenvalues_lower])
    
    v1 = np.r_[eigenvector_upper, np.zeros((eigenvector_lower.shape[0],eigenvector_upper.shape[1]))]
    v2 = np.r_[np.zeros((eigenvector_upper.shape[0],eigenvector_lower.shape[1])), eigenvector_lower]
    
    return eigenvalues, eigenvectors @ np.c_[v1,v2]

def sort_by_same_order(eigenvalues, eigenvectors):
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    return sorted_eigenvalues, sorted_eigenvectors

def create_symmetric_matrix(n):
    # Generate random values for the upper triangular part of the matrix
    upper_triangular = np.random.rand(n, n)

    # Create a symmetric matrix by copying the upper triangular part to the lower triangular part
    symmetric_matrix = np.triu(upper_triangular) + np.triu(upper_triangular, 1).T

    return symmetric_matrix

def is_symmetric(matrix):
    # Convert the matrix to a NumPy array
    matrix = np.array(matrix)
    
    # Check if the matrix is equal to its transpose
    return np.array_equal(matrix, matrix.T)

if __name__ == '__main__':
    matrix = np.loadtxt("inv_matrix(800 x 800).txt", dtype='f', delimiter=' ')
    # matrix = create_symmetric_matrix(800)
    matrix = matrix[:3, :3]
    print(is_symmetric(matrix))

    print("Converting matrix to Hessenberg form...")
    hessen_mat, H = hessenberg_form(matrix)
    
    print("Calculating eigenvalues & eigenvectors of matrix using QR Algorithm...")
    tt = time()
    values, vectors = my_eigen_recursive(hessen_mat, epsilon=1e-6)
    vectors = H @ vectors

    duration=time()-tt
    print("\nmy_eigen_recursive:\t", duration)

    tt = time()
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    duration=time()-tt
    print("np.linalg.eig: \t\t", duration)
    
    sorted_values, sorted_vectors = sort_by_same_order(values, vectors)
    sorted_eigenvalues, sorted_eigenvectors = sort_by_same_order(eigenvalues, eigenvectors)
    print("\nEigenvector norm difference : \t", np.linalg.norm(np.abs(sorted_vectors) - np.abs(sorted_eigenvectors)))
    print("Eigenvalue norm difference  : \t", np.linalg.norm(np.abs(sorted_values)- np.abs(sorted_eigenvalues))) 

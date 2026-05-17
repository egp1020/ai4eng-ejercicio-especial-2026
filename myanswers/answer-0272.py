import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reduce_genomic_dimensions(X, variance_target=0.90):
    """
    Reduce dimensionalidad preservando al menos variance_target de la varianza.
    Retorna una tupla: (pca_model, X_reduced)
    """
    # 1. Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. PCA completo para calcular varianza acumulada
    pca_full = PCA(svd_solver="full")
    pca_full.fit(X_scaled)

    evr_cum = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = np.argmax(evr_cum >= variance_target) + 1

    # 3. PCA final con el número mínimo de componentes
    pca_model = PCA(n_components=n_components, svd_solver="full")
    X_reduced = pca_model.fit_transform(X_scaled)

    return pca_model, X_reduced
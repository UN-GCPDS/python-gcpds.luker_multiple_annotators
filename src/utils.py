import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Rectangle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor

def create_dfs(df_base: pd.DataFrame, fq_vars: list, sens_vars: list, num_annotators: int, how_dropna: str = 'any') -> tuple:
    """Creates dataframes for feature variables, multi-annotator responses, and annotator list.

    Args:
        df_base (pd.DataFrame): The base dataframe containing all data.
        fq_vars (list): List of feature variables.
        sens_vars (list): List of sensory variables.
        num_annotators (int): Number of annotators to consider.
        how_dropna (str, optional): How to handle missing values when dropping rows. Defaults to 'any'.

    Returns:
        tuple: (df_vars, df_ma, annotators)
            - df_vars (pd.DataFrame): Dataframe containing feature variables.
            - df_ma (pd.DataFrame): Multi-annotator dataframe.
            - annotators (list): List of selected annotators.
    """
    fq_vars = list(fq_vars)
    df_anotadores = df_base[['codigo evaluador', 'codigo sampler'] + sens_vars]
    anotadores = df_base[df_base.loc[:, sens_vars].notna().values]['codigo evaluador'].value_counts().index[:num_annotators].to_list()
    df_anotadores = df_anotadores[df_anotadores['codigo evaluador'].isin(anotadores)]
    df_ma = pd.pivot_table(df_anotadores, columns='codigo evaluador', index='codigo sampler')
    df_ma.dropna(inplace=True, how=how_dropna)
    df_vars = df_base[['codigo sampler'] + fq_vars].set_index('codigo sampler').loc[df_ma.index, :].drop_duplicates().dropna()
    df_ma = df_ma.loc[df_vars.index, :]

    if len(sens_vars) > 1:
        mask = pd.Series(True, index=df_ma.index)
        for col in df_ma.columns.get_level_values(0).unique():
            mask *= ~df_ma[col].isna().all(axis=1)
        mask = mask.astype('bool')
        df_vars, df_ma = df_vars.loc[mask], df_ma.loc[mask]

    return df_vars, df_ma, anotadores

def create_dfs_impute(df_fq: pd.DataFrame, df_sens: pd.DataFrame, annotators: list, fq_vars: list, sens_vars: list, family: str = None) -> tuple:
    """Creates and imputes dataframes for feature variables and multi-annotator responses.

    Args:
        df_fq (pd.DataFrame): Dataframe with feature variables.
        df_sens (pd.DataFrame): Dataframe with sensory variables.
        annotators (list): List of annotators.
        fq_vars (list): List of feature variables.
        sens_vars (list): List of sensory variables.
        family (str, optional): Optional filter for a specific family. Defaults to None.

    Returns:
        tuple: (df_vars, df_ma)
            - df_vars (pd.DataFrame): Imputed dataframe of feature variables.
            - df_ma (pd.DataFrame): Imputed multi-annotator dataframe.
    """
    df_comb = df_sens.merge(df_fq, left_on='codigo sampler', right_on='CodMuestra')
    df_fam = df_comb[df_comb.Matriz == family] if family else df_comb
    cods_sampler = df_fam.CodMuestra.unique()
    df_fq_fam = df_fq[df_fq['CodMuestra'].isin(cods_sampler)]
    df_vars = df_fq_fam[['CodMuestra'] + fq_vars].set_index('CodMuestra')

    try:
        df_vars.loc[df_vars['humedad - determinador halogeno'] > 1, :] = np.nan
    except KeyError:
        pass

    # Imputation for feature variables
    imputer = IterativeImputer()
    df_vars[:] = imputer.fit_transform(df_vars)

    # Processing multi-annotator dataframe
    df_sens_fam = df_sens[df_sens['codigo sampler'].isin(cods_sampler)]
    df_sens_fam = df_sens_fam[['codigo sampler', 'codigo evaluador'] + sens_vars]
    df_sens_fam = df_sens_fam[df_sens_fam['codigo evaluador'].str.isnumeric()]
    df_sens_fam.iloc[:, 1:] = df_sens_fam.iloc[:, 1:].apply(pd.to_numeric)
    df_sens_fam = df_sens_fam[df_sens_fam['codigo evaluador'].isin(annotators)]
    df_ma = pd.pivot_table(df_sens_fam, columns='codigo evaluador', index='codigo sampler')
    
    idx = df_ma.index.intersection(df_vars.index)
    df_ma = df_ma.loc[idx]
    df_vars = df_vars.loc[idx]

    # Impute missing sensory values using K-Nearest Neighbors
    for ann in annotators:
        for var in sens_vars:
            knn = KNeighborsRegressor()
            mask_test = df_ma.loc[:, (var, ann)].isna()
            mask_train = ~mask_test
            x_train = df_vars[mask_train].values
            x_test = df_vars[mask_test].values
            y_train = df_ma.loc[:, (var, ann)][mask_train]
            knn.fit(x_train, y_train)
            df_ma.loc[:, (var, ann)][mask_test] = np.round(knn.predict(x_test) * 2) / 2

    return df_vars, df_ma

def get_iAnn(y: np.ndarray) -> np.ndarray:
    """Generates a binary mask indicating the presence of annotations.

    Args:
        y (np.ndarray): Sensory data matrix.

    Returns:
        np.ndarray: Binary mask where 1 indicates presence and 0 indicates missing data.
    """
    return (~np.isnan(y)).astype(float)

def plot_clusters(X_red: np.ndarray, productos: pd.Series, k: int = 2) -> np.ndarray:
    """Performs k-means clustering and plots the results.

    Args:
        X_red (np.ndarray): Reduced feature space for clustering.
        productos (pd.Series): Product labels corresponding to data points.
        k (int, optional): Number of clusters. Defaults to 2.

    Returns:
        np.ndarray: Cluster labels assigned by k-means.
    """
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X_red)
    labels = kmeans.labels_

    # Plot clusters
    for i in range(k):
        plt.plot(*X_red[labels == i].T, '.', ms=10, label=f'Grupo {i+1}')
    plt.title('Agrupamiento K-medias')
    plt.legend()
    plt.show()

    return labels

def plot_MA_red(X: np.ndarray, y: np.ndarray, anotadores: list, markers: list, 
                productos: np.ndarray, productos_unicos: list, method: str = 'pca') -> np.ndarray:
    """Plots multi-annotator data using dimensionality reduction (PCA or t-SNE).

    This function reduces the feature space using PCA or t-SNE and visualizes 
    the results for each annotator, grouping products with different markers.

    Args:
        X (np.ndarray): Input data matrix (samples, features).
        y (np.ndarray): Sensory data matrix (samples, annotators).
        anotadores (list): List of annotators.
        markers (list): List of markers for different products.
        productos (np.ndarray): Array of product labels corresponding to samples.
        productos_unicos (list): List of unique product labels.
        method (str, optional): Dimensionality reduction method ('pca' or 'tsne'). Defaults to 'pca'.

    Returns:
        np.ndarray: Transformed data (samples, 2) after dimensionality reduction.
    """
    if method == 'pca':
        red = PCA(n_components=2)
    elif method == 'tsne':
        red = TSNE(n_components=2, perplexity=5)
    X_red = red.fit_transform(X)
    n_rows = len(anotadores) // 2 + 1

    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 4.5 * n_rows), sharex=True, sharey=True)
    axs = axs.flatten()
    
    for i, anotador in enumerate(anotadores):
        for j, product in enumerate(productos_unicos):
            mask = productos == product
            scatter = axs[i].scatter(X_red[mask, 0], X_red[mask, 1], c=y[mask, i], cmap='viridis', 
                                     marker=markers[j], label=product.split(' / ')[0], 
                                     vmin=np.nanmin(y), vmax=np.nanmax(y))
        axs[i].set_title(f'Representación {method.upper()} para el anotador {anotador}')

    handles = []
    labels = []
    for j, product in enumerate(productos_unicos):
        mask = productos == product
        scatter = axs[-1].scatter(X_red[mask, 0], X_red[mask, 1], c=np.ones(y[mask, 0].shape) * np.nanmin(y), 
                                  cmap='viridis', marker=markers[j], label=product.split(' / ')[0], 
                                  vmin=np.nanmin(y), vmax=np.nanmax(y))
        handles.append(scatter)
        labels.append(product.split(' / ')[0])

    fig.colorbar(scatter, ax=axs[-1], label='Variable sensorial')
    axs[-1].legend(handles, labels, loc=(0.15, 0.25), title='Productos', prop={'size': 8})
    
    rect = Rectangle((axs[-1].get_xlim()[0], axs[-1].get_ylim()[0]),  
                     axs[-1].get_xlim()[1] - axs[-1].get_xlim()[0],   
                     axs[-1].get_ylim()[1] - axs[-1].get_ylim()[0],   
                     facecolor='white', edgecolor='none')  
    axs[-1].add_patch(rect)

    for i in range(2 * n_rows - len(anotadores)):
        axs[-i-1].axis('off')
    axs[-1].axis('off')

    plt.tight_layout()
    plt.show()
    return X_red

def plot_kde_clusters(df_vars: pd.DataFrame, labels: np.ndarray) -> None:
    """Plots Kernel Density Estimation (KDE) distributions for each cluster.

    This function overlays KDE plots for different clusters, allowing visualization 
    of variable distributions within each cluster.

    Args:
        df_vars (pd.DataFrame): Dataframe containing variables for each sample.
        labels (np.ndarray): Cluster labels assigned to each sample.

    Returns:
        None: The function generates plots but does not return any value.
    """
    df_vars['cluster'] = labels
    num_clusters = len(df_vars['cluster'].unique())

    # Iterate through each variable and plot KDE for each cluster
    for variable, series in df_vars.drop(columns='cluster').items():
        plt.figure(figsize=(8, 4))
        for cluster in range(num_clusters):
            ax = plt.gca()
            cluster_data = series[df_vars['cluster'] == cluster]
            sns.kdeplot(cluster_data, ax=ax, label=f'Cluster {cluster+1}')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.title(variable)
        plt.show()

    plt.tight_layout()
    plt.show()

def transform_data(training_data):
    """Transforms training data into structured arrays for model training.

    Args:
        training_data: Training dataset containing sensory and physicochemical data.

    Returns:
        tuple: (sensory_data_dict, physicochemical_array)
            - sensory_data_dict (dict): Dictionary mapping sensory variables to numpy arrays.
            - physicochemical_array (np.ndarray): Array containing physicochemical data.
    """
    result = {}
    physicochemical_result = []

    # Extract unique annotator user IDs
    unique_users = set()
    for sample in training_data.data:
        unique_users.update(sample.sensory_data_by_user.keys())
    unique_users = sorted(unique_users)

    # Process each sample
    for i, sample in enumerate(training_data.data):
        sensory_data_by_user = sample.sensory_data_by_user
        physicochemical_data = sample.physicochemical_data
        physicochemical_result.append([physicochemical_data.dict()[var] for var in physicochemical_data.dict()])

        # Process sensory data
        for user_id, user_data in sensory_data_by_user.items():
            if user_data.sensory_data:
                for sensory_var, value in user_data.sensory_data[0].dict().items():
                    if sensory_var not in result:
                        result[sensory_var] = pd.DataFrame(columns=unique_users)
                    result[sensory_var].loc[i, user_id] = value

    # Convert data structures to numpy arrays
    for key in result.keys():
        result[key] = result[key].to_numpy(dtype='float')
    physicochemical_result = pd.DataFrame(physicochemical_result).to_numpy()

    return result, physicochemical_result

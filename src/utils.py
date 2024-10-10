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

def create_dfs(df_base, fq_vars, sens_vars, num_annotators, how_dropna='any'):
    fq_vars = list(fq_vars)
    df_anotadores = df_base[['codigo evaluador', 'codigo sampler'] + sens_vars]
    anotadores = df_base[df_base.loc[:,sens_vars].notna().values]['codigo evaluador'].value_counts().index[:num_annotators].to_list()
    df_anotadores = df_anotadores[df_anotadores['codigo evaluador'].isin(anotadores)]
    df_ma = pd.pivot_table(df_anotadores, columns='codigo evaluador', index='codigo sampler')
    df_ma.dropna(inplace=True, how=how_dropna)
    df_vars = df_base[['codigo sampler'] + fq_vars].set_index('codigo sampler').loc[df_ma.index, :].drop_duplicates().dropna()
    df_ma = df_ma.loc[df_vars.index,:]
    if len(sens_vars) > 1:
        mask = pd.Series(True, index=df_ma.index)
        for col in df_ma.columns.get_level_values(0).unique():
            mask *= ~df_ma[col].isna().all(axis=1)
        mask = mask.astype('bool')
        df_vars, df_ma = df_vars.loc[mask], df_ma.loc[mask]
    return df_vars, df_ma, anotadores

def create_dfs_impute(df_fq, df_sens, annotators, fq_vars, sens_vars, family):
    df_comb = df_sens.merge(df_fq, left_on='codigo sampler', right_on='CodMuestra')
    df_fam = df_comb[df_comb.Matriz == family]
    cods_sampler = df_fam.CodMuestra.unique()
    df_fq_fam = df_fq[df_fq['CodMuestra'].isin(cods_sampler)]
    df_vars = df_fq_fam[['CodMuestra'] + fq_vars].set_index('CodMuestra')
    try:
        df_vars.loc[df_vars['humedad - determinador halogeno'] > 1,:] = np.nan
    except:
        pass
    # imputation fq_vars
    imputer = IterativeImputer()
    df_vars[:] = imputer.fit_transform(df_vars)

    # df_ma
    df_sens_fam = df_sens[df_sens['codigo sampler'].isin(cods_sampler)]
    df_sens_fam = df_sens_fam.loc[:,['codigo sampler', 'codigo evaluador'] + sens_vars]
    df_sens_fam = df_sens_fam[df_sens_fam['codigo evaluador'].str.isnumeric()]
    df_sens_fam.iloc[:,1:] = df_sens_fam.iloc[:,1:].apply(pd.to_numeric)
    df_sens_fam = df_sens_fam[df_sens_fam['codigo evaluador'].isin(annotators)]
    df_ma = pd.pivot_table(df_sens_fam, columns='codigo evaluador', index='codigo sampler')
    idx = df_ma.index.intersection(df_vars.index)
    df_ma = df_ma.loc[idx]
    df_vars = df_vars.loc[idx]
    for ann in annotators:
        for var in sens_vars:
            knn = KNeighborsRegressor()
            mask_test = df_ma.loc[:,(var, ann)].isna()
            mask_train = ~mask_test
            x_train = df_vars[mask_train].values
            x_test = df_vars[mask_test].values
            y_train = df_ma.loc[:,(var, ann)][mask_train]
            knn.fit(x_train, y_train)
            df_ma.loc[:,(var, ann)][mask_test] = np.round(knn.predict(x_test)*2)/2
    return df_vars, df_ma

def get_iAnn(y):
    return (~np.isnan(y)).astype(float)

def plot_clusters(X_red, productos, k=2):
    # plot clusters
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(X_red)
    labels = kmeans.labels_
    for i in range(k):
        plt.plot(*X_red[labels == i].T, '.', ms=10, label=f'Grupo {i+1}')
    plt.title(f'Agrupamiento K-medias')
    plt.legend()
    plt.show()

    # plot distributions on each cluster
    df = pd.DataFrame({'products' : productos.values, 'cluster':labels})

    # Group the DataFrame by 'products' and 'cluster' and count occurrences
    counts = df.groupby(['products', 'cluster']).size().unstack(fill_value=0)

    # Get unique products and clusters
    products = df['products'].unique()
    clusters = df['cluster'].unique()

    # Set the y positions for the bars
    y = np.arange(len(products))

    # Calculate the width of each cluster's bar
    num_clusters = len(clusters)
    bar_width = 1 / (num_clusters + 1)  # +1 to add space between clusters

    # Plot the bars for each cluster
    for i, cluster_value in enumerate(clusters):
        plt.barh(y +(k-i-1) * bar_width, counts[cluster_value], height=bar_width, label=f'Grupo {i+1}')

    # Add labels and title
    plt.ylabel('Productos')
    plt.xlabel('Conteo')
    plt.title('Productos por grupo')
    plt.yticks(y + bar_width * (num_clusters - 1) / 2, products)

    # Add legend
    plt.legend()
    plt.show()
    return labels

def plot_MA_red(X, y, anotadores, markers, productos, productos_unicos, method='pca'):
    if method == 'pca':
        red = PCA(n_components=2)
    elif method == 'tsne':
        red = TSNE(n_components=2, perplexity=5)
    X_red = red.fit_transform(X)
    n_rows = len(anotadores) // 2 + 1

    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 4.5*n_rows), sharex=True, sharey=True)
    axs = axs.flatten()
    for i, anotador in enumerate(anotadores):
        for j, product in enumerate(productos_unicos):
            mask = productos == product
            scatter = axs[i].scatter(X_red[mask, 0], X_red[mask, 1], c=y[mask, i], cmap='viridis', marker=markers[j],
                                    label=product.split(' / ')[0], vmin=np.nanmin(y), vmax=np.nanmax(y))
            # if i == 0:
            #     handles.append(scatter)
            #     labels.append(product)
        # axs[i].set_xlabel('Principal Component 1')
        # axs[i].set_ylabel('Principal Component 2')
        axs[i].set_title(f'Representación {method.upper()} para el anotador {anotador}')
    handles = []
    labels = []
    for j, product in enumerate(productos_unicos):
        mask = productos == product
        scatter = axs[-1].scatter(X_red[mask, 0], X_red[mask, 1], c=np.ones(y[mask,0].shape) * np.nanmin(y), cmap='viridis', marker=markers[j],
                                        label=product.split(' / ')[0], vmin=np.nanmin(y), vmax=np.nanmax(y))
        handles.append(scatter)
        labels.append(product.split(' / ')[0])
    fig.colorbar(scatter, ax=axs[-1], label='Variable sensorial')
    axs[-1].legend(handles, labels, loc=(0.15, 0.25), title='Productos', prop={'size': 8})
    rect = Rectangle((axs[-1].get_xlim()[0], axs[-1].get_ylim()[0]),  # Bottom-left corner
                    axs[-1].get_xlim()[1] - axs[-1].get_xlim()[0],   # Width
                    axs[-1].get_ylim()[1] - axs[-1].get_ylim()[0],   # Height
                    facecolor='white', edgecolor='none')  # Color and transparency
    axs[-1].add_patch(rect)
    for i in range(2*n_rows - len(anotadores)):
        axs[-i-1].axis('off')
    axs[-1].axis('off')
    plt.tight_layout()
    plt.show()
    return X_red

def plot_kde_clusters(df_vars, labels):

    # Add cluster labels as a new column to the DataFrame
    df_vars['cluster'] = labels

    num_clusters = len(df_vars['cluster'].unique())

    # Determine the number of variables (excluding the 'cluster' column)
    num_variables = len(df_vars.columns) - 1

    # Create a subplot grid based on the number of variables
    # fig, axes = plt.subplots(nrows=num_variables, ncols=1, figsize=(8, 4*num_variables))

    # Iterate through each variable and plot histograms for each cluster
    for i, (variable, series) in enumerate(df_vars.drop(columns='cluster').items()):
        plt.figure(figsize=(8, 4))
        for cluster in range(num_clusters):
            ax=plt.gca()
            cluster_data = series[df_vars['cluster'] == cluster]
            sns.kdeplot(cluster_data, ax=ax, label=f'Cluster {cluster+1}')
        plt.xlabel('Valor')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.title(variable)
        plt.show()

    plt.tight_layout()
    plt.show()

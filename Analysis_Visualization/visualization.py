import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

def plot_regression_results(y_test, y_pred, weights, target_clm = 'Target Column'):
    '''Produces three plots to analyze the results of linear regression:
        -True vs predicted
        -Raw residual histogram
        -Weight histogram

    Inputs:
        y_test: (n_observations,) numpy array with true values
        y_pred: (n_observations,) numpy array with predicted values
        weights: (n_weights) numpy array with regression weights'''

    print('MSE: ', mean_squared_error(y_test, y_pred))
    print('r^2: ', r2_score(y_test, y_pred))

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    # predicted vs true
    ax[0].scatter(y_test, y_pred, s=2)
    ax[0].set_title('True vs. Predicted')
    ax[0].set_xlabel('True %s' % (target_clm))
    ax[0].set_ylabel('Predicted %s' % (target_clm))

    # residuals
    error = np.squeeze(np.array(y_test)) - np.squeeze(np.array(y_pred))
    ax[1].hist(np.array(error), bins=30)
    ax[1].set_title('Raw residuals')
    ax[1].set_xlabel('(true-predicted)')

    # weight histogram
    ax[2].hist(weights, bins=30)
    ax[2].set_title('weight histogram')

    plt.tight_layout()




def plot_mse_alphas(alphas, errors):
    # Plot of MSE  vs. alphas
    # alphas of type 'numpy.ndarray'
    # errors of type 'numpy.ndarray'

    plt.figure(figsize=(7, 5))
    plt.plot(alphas, errors, marker='o')
    plt.xscale("log")
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Test Set MSE')
    plt.title('Test Set MSE vs. Regularization Strength (Alpha)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    # find minumum
    min_error_idx = np.argmin(errors)

    print(f"Minumum MSE = {errors[min_error_idx]:.2f}\nat alpha = {alphas[min_error_idx]:.2F}")


# TODO write a plot function
def plot_losses_epochs(n_epoch,train_losses, val_losses):
    # n_epoch of type int
    # train_losses of type list
    # val_losses of type list
    epochs = [e+1 for e in range(n_epoch)]

    # Create DataFrame for Seaborn
    loss_df = pd.DataFrame({
        'Epoch': epochs * 2,
        'Loss': train_losses + val_losses,
        'Set': ['Training results'] * len(epochs) + ['Validation results'] * len(epochs)
    })
   
    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=loss_df,
        x='Epoch',
        y='Loss',
        hue='Set',
        style='Set',
        markers=True,
        dashes=True
    )
    plt.title("Epoch vs. Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(1, len(epochs))
    plt.show()


def plot_multi_model_losses(models_losses, n_epochs):
    # Losses vs Epochs for several models
    """
    - models_losses: Dict with model name as key and a dict as value:
        {
            'model_name': {
                'train': [list of train losses],
                'val': [list of val losses]
            }
        }
    - n_epochs: int, number of epochs
    """
    epochs = list(range(1, n_epochs + 1))
    all_data = []

    for model_name, losses in models_losses.items():
        train = losses['train']
        val = losses['val']


        # Append both train and val losses with model name
        for epoch, loss in zip(epochs, train):
            all_data.append({'Epoch': epoch, 'Loss': loss, 'Set': 'Training', 'Model': model_name})
        for epoch, loss in zip(epochs, val):
            all_data.append({'Epoch': epoch, 'Loss': loss, 'Set': 'Validation', 'Model': model_name})

    df = pd.DataFrame(all_data)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Loss',
        hue='Model',
        style='Set',
        markers=True,
        dashes=True
    )
    plt.title("Epoch vs. Loss for Multiple Models")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(title='Model / Set')
    plt.tight_layout()
    plt.show()


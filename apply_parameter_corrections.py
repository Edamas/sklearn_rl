import pandas as pd
import ast

# Memorized corrections content (the ~30 records)
memorized_parameters_tsv_content = """param_name	param_dtype	param_standard	param_min	param_max	param_list	param_required	descrição do parâmetro	apt_for_training	observações	from_sklearn_docs	estimators_list
alpha	float,array	0.0000000001	0.00000000000000000001	9999	[]	False	Value added to the diagonal of the kernel matrix during fitting. This can prevent a potential numerical issue during fitting, by ensuring that the calculated values form a positive definite matrix. It can also be interpreted as the variance of additional Gaussian measurement noise on the training observations.	True	If an array is passed, it must have the same number of entries as the data used for fitting and is used as datapoint-dependent noise level.	True	['GaussianProcessRegressor','Ridge']
alpha	float	0.0001	0.0	9999	[]	False	Constant that multiplies the regularization term if regularization is used.	True	Only used if penalty is not None.	True	['Perceptron','SGDClassifier']
alpha	float	0.0001	0.0	9999	[]	False	Constant that multiplies the regularization term.	True	The higher the value, the stronger the regularization. Also used for learning rate when learning_rate='optimal'.	True	['SGDClassifier','SGDRegressor']
alpha	float	0.0001	0.0	9999	[]	False	Strength of the squared L2 regularization. Note that the penalty is equal to alpha * ||w||^2.	True	Must be in the range [0, inf).	True	['HuberRegressor']
alpha	float	0.0001			[]	False	Strength of the L2 regularization term.	True	The L2 regularization term is divided by the sample size when added to the loss.	True	['MLPClassifier']
alpha	float	0.01	0.0	9999	[]	False	The regularization parameter: the higher alpha, the more regularization, the sparser the inverse covariance. Range is (0, inf].	True		True	['GraphicalLasso']
alpha	float	0.05	0.0	1.0	[]	False	The highest uncorrected p-value for features to keep.	True		True	['SelectFdr','SelectFwe']
alpha	float	0.2	0.0	1.0	[]	False	Clamping factor. A value in (0, 1) that specifies the relative amount that an instance should adopt the information from its neighbors as opposed to its initial label.	True	alpha=0 means keeping the initial label information; alpha=1 means replacing all initial information.	True	['LabelSpreading']
alpha	float	0.9	0.0	1.0	[]	False	The alpha-quantile of huber loss and quantile loss functions.	True	Only if loss='huber' or loss='quantile'. Values must be in (0.0, 1.0).	True	['GradientBoostingRegressor']
alpha	float	1.0	-9999	9999	[]	False	A distance scaling parameter as used in robust single linkage. See [3] for more information.	True		True	['HDBSCAN']
alpha	float	1.0	0	9999	[]	False	Hyperparameter of the ridge regression that learns the inverse transform (when fit_inverse_transform=True).	True		True	['KernelPCA']
alpha	float	1.0	0.0	9999	[]	False	Sparsity controlling parameter. Higher values lead to sparser components.	True		True	['SparsePCA']
alpha	float	1.0	0.0	9999	[]	False	Constant that multiplies the penalty terms. alpha = 0 is equivalent to ordinary least square. For numerical reasons, using alpha = 0 with the Lasso object is not advised - use LinearRegression instead.	True	Controls the overall strength of regularization. Higher values increase regularization.	True	['ElasticNet','Ridge']
alpha	float	1.0	0.0	9999	[]	False	Constant that multiplies the L1 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf). When alpha = 0, the objective is equivalent to ordinary least squares, solved by the LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso object is not advised. Instead, you should use the LinearRegression object.	True		True	['Lasso']
alpha	float	1.0	0.0	9999	[]	False	Constant that multiplies the L1/L2 term.	True		True	['MultiTaskLasso']
alpha	float	1.0	0.0	9999	[]	False	Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization. Alpha corresponds to 1 / (2C) in other linear models such as LogisticRegression or LinearSVC.	True		True	['RidgeClassifier']
alpha	float	1.0	0.0	9999	[]	False	Constant that multiplies the penalty term.	True	alpha = 0 is equivalent to an ordinary least square, solved by LinearRegression. For numerical reasons, using alpha = 0 with the LassoLars object is not advised.	True	['LassoLars']
alpha	float	1.0	0.0	9999	[]	False	Regularization constant that multiplies the L1 penalty term.	True		True	['QuantileRegressor']
alpha	float_or_array	1.0	0.0	9999	[]	False	Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).	True	When alpha = 0, the objective is equivalent to ordinary least squares. For numerical reasons, using alpha = 0 with the Ridge object is not advised.	True	['Ridge']
alpha	float_or_array	1.0	0.0	9999	[]	False	Regularization strength; must be a positive float.	True	Regularization improves the conditioning of the problem and reduces the variance of the estimates. Larger values specify stronger regularization. Alpha corresponds to 1 / (2C) in other linear models such as LogisticRegression or LinearSVC. If an array is passed, penalties are assumed to be specific to the targets.	True	['KernelRidge']
alpha	float	1.0	0.0	9999	[]	False	Constant that multiplies the L1/L2 term. Defaults to 1.0.	True		True	['MultiTaskElasticNet','ElasticNet']
alpha	float	1.0	0	9999	[]	False	Constant that multiplies the L2 penalty term and determines the regularization strength.	True	alpha = 0 is equivalent to unpenalized GLMs. In this case, the design matrix X must have full column rank (no collinearities). Values of alpha must be in the range [0.0, inf).	True	['TweedieRegressor','GammaRegressor','PoissonRegressor']
alpha	float	1.0	0	9999	[]	False	Sparsity controlling parameter.	True		True	['MiniBatchDictionaryLearning','dict_learning_online']
alpha	float_or_array	1.0	0	9999	[]	False	Additive (Laplace/Lidstone) smoothing parameter (set alpha=0 and force_alpha=True, for no smoothing).	True		True	['ComplementNB','MultinomialNB','BernoulliNB']
alpha	float	1.0	0.0000000001	9999	[]	False	Sparsity controlling parameter.	True		True	['DictionaryLearning']
alpha	int	1	0	9999	[int]	False	Sparsity controlling parameter.	True	Higher values lead to sparser components.	True	['MiniBatchSparsePCA']
class_weight	dynamic	None			['balanced',dict,None]	False	Weights associated with classes.	True	'balanced' adjusts weights inversely proportional to class frequencies. Added in 0.17.	True	['PassiveAggressiveClassifier']
copy_X	bool	True			[True,False]	False	If True, X will be copied; else may be overwritten.	True	Recommended to keep as True to preserve original data.	True	['LassoCV','ElasticNetCV']
warm_start	bool	False			[True,False]	False	When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.	True	Useful for successive fitting with slightly different parameters or data.	True	['ElasticNet']
n_jobs	int	None	-1	9999	[int]	False	Number of CPUs for OVA computation in multi-class problems.	False	None means 1, -1 means using all processors.	True	['PassiveAggressiveClassifier']
"""

# Read memorized content into a DataFrame
memorized_df = pd.read_csv(pd.io.common.StringIO(memorized_parameters_tsv_content), sep='\t')

# Convert 'estimators_list' from string to actual list for both DataFrames
# Handle potential NaN values in estimators_list before literal_eval
memorized_df['estimators_list'] = memorized_df['estimators_list'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# Function to apply corrections
def apply_corrections(full_df, corrections_df):
    updated_df = full_df.copy()
    for index, correction_row in corrections_df.iterrows():
        param_name = correction_row['param_name']
        estimators_list = correction_row['estimators_list']

        # Find matching rows in the full DataFrame
        # We need to ensure that the estimators_list in the full_df contains all estimators from the correction_row's estimators_list
        # This is a more robust way to handle partial matches or different orderings
        
        # Convert updated_df's estimators_list to actual list for comparison
        updated_df['estimators_list_parsed'] = updated_df['estimators_list'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

        match_indices = updated_df[
            (updated_df['param_name'] == param_name) &
            (updated_df['estimators_list_parsed'].apply(lambda x: all(elem in x for elem in estimators_list)))
        ].index

        if not match_indices.empty:
            # Apply all columns from the correction_row to the matched rows
            for col in corrections_df.columns:
                if col != 'estimators_list': # Don't directly copy the string version of estimators_list
                    updated_df.loc[match_indices, col] = correction_row[col]
            # For estimators_list, ensure it's updated with the exact list from correction_row
            updated_df.loc[match_indices, 'estimators_list'] = str(estimators_list) # Convert back to string for saving

    # Drop the temporary parsed column
    updated_df = updated_df.drop(columns=['estimators_list_parsed'])
    return updated_df

# Main execution
if __name__ == "__main__":
    try:
        # Load the full parameters.tsv provided by the user
        full_parameters_path = 'D_training/parameters.tsv'
        full_df = pd.read_csv(full_parameters_path, sep='\t')

        # Apply corrections
        corrected_df = apply_corrections(full_df, memorized_df)

        # Save the updated DataFrame back to the original path
        corrected_df.to_csv(full_parameters_path, sep='\t', index=False, encoding='utf-8')

        print("Correções aplicadas com sucesso ao parameters.tsv.")
    except FileNotFoundError:
        print(f"Erro: O arquivo {full_parameters_path} não foi encontrado. Certifique-se de que o arquivo completo foi adicionado.")
    except Exception as e:
        print(f"Ocorreu um erro ao aplicar as correções: {e}")

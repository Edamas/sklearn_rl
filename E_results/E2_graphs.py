import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler # Assuming StandardScaler is always part of the pipeline
from sklearn.utils import estimator_html_repr
import pydoc


DATA_DIR = Path("A_inputs/A1_datasets")

def graphs_app():
    st.header("📊 Gráficos")

    # Selecionar dataset
    data_files = [f.name for f in DATA_DIR.glob("*.csv")] + [f.name for f in DATA_DIR.glob("*.tsv")]
    if not data_files:
        st.warning(f"Nenhum arquivo .csv ou .tsv encontrado na pasta '{DATA_DIR}'.")
        st.stop()

    selected_file = st.selectbox("Selecione um arquivo de dados:", data_files)

    if selected_file:
        file_path = DATA_DIR / selected_file
        try:
            if file_path.suffix == '.tsv':
                df = pd.read_csv(file_path, sep='\t')
            else:
                df = pd.read_csv(file_path)
        except Exception as e:
            error_message = f"Erro ao ler o arquivo '{file_path.name}': {e}"
            st.error(error_message)
            st.exception(e)
            st.stop()

        st.dataframe(df.head())

        st.subheader("5.4 Gráficos")

        plot_type = st.radio("Selecione o tipo de gráfico:", ["Scatterplot", "Histograma", "Gráfico de Barras"])

        columns = df.columns.tolist()

        if plot_type == "Scatterplot":
            x_col = st.selectbox("Eixo X:", columns)
            y_col = st.selectbox("Eixo Y:", columns)
            color_col = st.selectbox("Cor (opcional):", [None] + columns)
            symbol_col = st.selectbox("Símbolo (opcional):", [None] + columns)

def show_pipeline_graph(pipeline_steps_repr, height=400, show_params=True): # Add parameters
    
    
    # Reconstruct the pipeline object from the representation
    steps = []
    for step_name, class_path, params in pipeline_steps_repr: # Unpack params
        try:
            # Dynamically locate the class
            estimator_class = pydoc.locate(class_path)
            if estimator_class:
                # Instantiate with parameters for accurate representation
                # Only pass params if show_params is True, otherwise pass empty dict
                if show_params:
                    steps.append((step_name, estimator_class(**params)))
                else:
                    steps.append((step_name, estimator_class())) # Instantiate without params
            else:
                st.warning(f"Could not locate class for step: {class_path}")
                steps.append((step_name, class_path)) # Fallback to string
        except Exception as e:
            st.warning(f"Error reconstructing step {step_name} ({class_path}) with params {params}: {e}")
            steps.append((step_name, class_path)) # Fallback to string

    if steps:
        try:
            # Create a dummy pipeline for visualization
            dummy_pipeline = Pipeline(steps)
            
            # Get the HTML representation
            html_repr = estimator_html_repr(dummy_pipeline)
            
            # Display in Streamlit
            st.components.v1.html(html_repr, height=height) # Use passed height
        except Exception as e:
            error_message = f"Erro ao gerar o diagrama do pipeline: {e}"
            st.error(error_message)
            
    else:
        st.info("Nenhuma etapa de pipeline para exibir.")

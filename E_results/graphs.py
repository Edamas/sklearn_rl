import plotly.express as px

def plot_scatter(df, x_col, y_col, color_col=None, symbol_col=None):
    """Gera um gráfico de dispersão."""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=1, symbol=symbol_col)
    return fig

def plot_histogram(df, x_col, color_col=None):
    """Gera um histograma."""
    fig = px.histogram(df, x=x_col, color=color_col)
    return fig

def plot_bar(df, x_col, y_col, color_col=None):
    """Gera um gráfico de barras."""
    fig = px.bar(df, x=x_col, y=y_col, color=color_col)
    return fig
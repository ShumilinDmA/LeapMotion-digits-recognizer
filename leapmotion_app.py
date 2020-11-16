import streamlit as st
from numpy import genfromtxt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from model import NeuralNet
from dataset import inference_picture, data_to_picture

net = NeuralNet(1296, random_state=2020)
net.load_weights()
# noinspection SpellCheckingInspection
st.set_option('deprecation.showfileUploaderEncoding', False)


def image_preprocessing(data):
    """
    Prepare data to inference
    :param data: Data in numpy array
    :return: Numpy array
    """
    inference_path = data_to_picture(data)
    inference_png = inference_picture(inference_path)
    return inference_png


def inference(model, image):
    """
    Make inference
    :param model: NeuralNet
    :param image: Data to process
    :return: probability for classes and class prediction
    """
    probability, prediction = model.forward_step(image, None, mode='inference')
    return probability[0], prediction[0]


def plot_digits(data):
    """
    Make plot of given data
    :param data: Numpy array
    :return:
    """
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("2D data", "3D data"),
                        specs=[[{'type': 'xy'}, {'type': 'surface'}]])  # Create the box for future plots
    fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], name='2D', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], name='3D',
                               line=dict(color='darkblue', width=2),
                               mode='lines'), row=1, col=2)

    fig.update_layout(height=400, width=800, showlegend=True, template='simple_white')
    st.plotly_chart(figure_or_data=fig, use_container_width=False)
    return


def plot_distribution(prob):
    """
    Plot probability distribution using bar plot
    :param prob: Probabilities
    :return:
    """
    fig = px.bar(x=range(0, 10), y=prob, title="Probability distribution",
                 labels={'x': 'Classes', 'y': 'Probability'})
    fig.update_traces(marker_color='yellow')
    st.plotly_chart(figure_or_data=fig, use_container_width=False)
    return


def main():
    st.header('Welcome to LeapMotion Digits recognizer!')
    st.sidebar.subheader('Choose file to classify:')
    file = st.sidebar.file_uploader("", key='file_loader')
    distribution = st.sidebar.checkbox("Show probability distribution?")
    if file is not None:
        numpy_data = genfromtxt(file, delimiter=',')
    else:
        numpy_data = genfromtxt('training_data/stroke_7_0015.csv', delimiter=',')

    image = image_preprocessing(numpy_data)
    probability, prediction = inference(net, image)

    st.markdown(f"<h3 style='text-align: center; color: Black;'>Predicted class of the data is: {prediction}</h3>",
                unsafe_allow_html=True)
    plot_digits(numpy_data)

    if distribution:
        plot_distribution(probability)


if __name__ == '__main__':
    main()

import streamlit as st
from numpy import genfromtxt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from model import NeuralNet
from dataset import inference_picture, data_to_picture

net = NeuralNet(1296, random_state=2020)
net.load_weights()
# noinspection SpellCheckingInspection
st.set_option('deprecation.showfileUploaderEncoding', False)


def image_preprocessing(data):
    inference_path = data_to_picture(data)
    inference_png = inference_picture(inference_path)
    return inference_png


def inference(model, image):
    probability, prediction = model.forward_step(image, None, mode='inference')
    return probability[0], prediction[0]


def plot_digits(data):
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("2D data", "3D data"),
                        specs=[[{'type': 'xy'}, {'type': 'surface'}]])  # Create the box for future plots
    fig.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], name='2D', mode='lines'), row=1, col=1)
    fig.add_trace(go.Scatter3d(x=data[:, 0], y=data[:, 1], z=data[:, 2], name='3D',
                               line=dict(color='darkblue', width=2)), row=1, col=2)

    fig.update_layout(height=400, width=800, showlegend=True, template='simple_white')
    st.plotly_chart(figure_or_data=fig, use_container_width=False)
    return


def main():
    st.header('Welcome to LeapMotion Digits recognizer!')
    file = st.sidebar.file_uploader("Choose file:", key='file_loader')
    numpy_data = genfromtxt(file, delimiter=',')

    plot_digits(numpy_data)
    image = image_preprocessing(numpy_data)
    probability, prediction = inference(net, image)
    st.text(f"Class of the data is: {prediction}")


if __name__ == '__main__':
    main()

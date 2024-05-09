from functools import lru_cache

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact
from transformers import GPT2Model, GPT2Tokenizer

# Load the tokenizer and the GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2Model.from_pretrained("gpt2-large")


# Function to apply the selected transformation
def apply_transformation(attention, transform_type):
    if transform_type == "sqrt":
        return np.sqrt(attention)
    elif transform_type == "sigmoid":
        return 1 / (1 + np.exp(-attention))
    elif transform_type == "tanh":
        return np.tanh(attention)
    elif transform_type == "cbrt":
        return np.cbrt(attention)
    elif transform_type == "normalize":
        return attention / np.max(attention)
    elif transform_type == "square":
        return np.square(attention)
    else:
        return attention


aggregation_dropdown = widgets.Dropdown(
    options=["none", "mean", "max", "median"],
    value="none",
    description="Aggregation:",
    disabled=False,
)

sum_heads_checkbox = widgets.Checkbox(
    value=False, description="Sum Heads", disabled=False
)


# Widget to select the transformation type
transform_dropdown = widgets.Dropdown(
    options=["none", "sqrt", "square", "sigmoid", "tanh", "cbrt", "normalize"],
    value="none",
    description="Transformation:",
    disabled=False,
)


def apply_aggregation(attention, aggregation_type, weights=None):
    if aggregation_type == "mean":
        return np.mean(attention, axis=0)
    elif aggregation_type == "max":
        return np.max(attention, axis=0)
    elif aggregation_type == "median":
        return np.median(attention, axis=0)
    else:
        return attention


# Function to visualize the attention matrix
def visualize_attention(attention, tokens):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(attention[: len(tokens), : len(tokens)], cmap="viridis")
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    fig.colorbar(im, ax=ax)
    return fig


# Update the get_attention_matrix function
def get_attention_matrix(text, layer, head, aggregation_type, transform_type):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    attention = outputs.attentions[layer].squeeze().detach().numpy()
    if aggregation_type != "none":
        attention = apply_aggregation(attention, aggregation_type)
    else:
        attention = attention[head]
    attention = apply_transformation(attention, transform_type)
    return attention


@lru_cache(300)
def cached_tokenizer(text):
    return tokenizer.tokenize(text)


# Update the cached_atte function
@lru_cache(300)
def cached_atte(text, layer, head, aggregation_type, transform_type):
    return get_attention_matrix(text, layer, head, aggregation_type, transform_type)


# Update the visualization function
def vis_attention(text, layer, head, aggregation_type, transform_type):
    tokens = cached_tokenizer(text)
    if len(tokens) < 2:
        return "Escribe más tokens"
    attention = cached_atte(text, layer, head, aggregation_type, transform_type)
    return visualize_attention(attention, tokens)


# Update the interact function
def interact_attention():
    interact(
        vis_attention,
        text="",
        layer=IntSlider(min=0, max=29, step=1, value=0),
        head=IntSlider(min=0, max=11, step=1, value=0),
        sum_heads=sum_heads_checkbox,
        aggregation_type=aggregation_dropdown,
        transform_type=transform_dropdown,
    )


def ex1():
    print(
        "Ejemplo 1: - Soy Alejandro. Tu profesor de Python - Mi profesor me da clases de \n Capa 13"
    )
    vis_attention(
        "- Soy Alejandro. Tu profesor de Python - Mi profesor me da clases de ",
        13,
        0,
        "none",
        "tanh",
    )


def ex2():
    print(
        "Ejemplo 2: - Soy Alejandro. Tu profesor de Python - Mi profesor me da clases de \n Capa 26"
    )
    vis_attention(
        "- Soy Alejandro. Tu profesor de Python - Mi profesor me da clases de ",
        26,
        0,
        "max",
        "tanh",
    )


def ex3():
    print(
        "Ejemplo 3: - Soy tu profesor de Python. Mi profesor me da clases de \n Capa 27"
    )
    vis_attention(
        "-Soy tu profesor de Python. Mi profesor me da clases de ", 27, 0, "max", "tanh"
    )


def ex4():
    print("Ejemplo 4: - 1+1=3.Fix it \n Capa 13")
    vis_attention("1+1=3.Fix it ", 13, 0, "max", "sqrt")


def ex5():
    vis_attention("1+1=3. Fix it:", 29, 6, "none", "normalize")


# exports ex1 ex2 ex3 interact_attention

__all__ = ["ex1", "ex2", "ex3", "ex4", "ex5", "interact_attention"]

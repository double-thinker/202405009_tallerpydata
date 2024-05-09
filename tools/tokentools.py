from functools import lru_cache


@lru_cache(maxsize=100)
def split_tokens(
    text: str, tokenizer_name: str = "cl100k_base", encoder=None, decoder=None
):
    """
    Split a given text into subword tokens and their corresponding token IDs.

    Args:
        text (str): The input text to be tokenized.
        tokenizer_name (str, optional): The name of the tokenizer to use. If provided, the function
            will use the tiktoken library for tokenization. Default is `cl100k_base`.
        encoder (callable, optional): A custom encoder function that takes a string and returns a list
            of token IDs. Required if tokenizer_name is not provided.
        decoder (callable, optional): A custom decoder function that takes a list of token IDs and
            returns the corresponding string. Required if tokenizer_name is not provided.

    Returns:
        tuple: A tuple containing two lists:
            - fragments (list): A list of strings representing the subword tokens.
            - tokenids (list): A list of integers representing the token IDs.

    Raises:
        ValueError: If neither tokenizer_name nor custom encoder and decoder functions are provided.
    """

    if tokenizer_name is not None:
        import tiktoken

        tokenizer = tiktoken.get_encoding(tokenizer_name)
        encoder = tokenizer.encode
        decoder = tokenizer.decode
    elif encoder is None and decoder is None:
        raise ValueError(
            "Either tokenizer_name or custom encoder and decoder functions must be provided."
        )

    tokenids = tokenizer.encode(text)
    fragments = [
        repr(tokenizer.decode([t]).replace(" ", "␣")).strip("'") for t in tokenids
    ]

    return fragments, tokenids

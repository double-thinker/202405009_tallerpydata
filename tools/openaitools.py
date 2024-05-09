def check_openai(verbose=True):
    try:
        import openai
    except ImportError:
        if verbose:
            print(
                "La librería openai no está instalada. Por favor, instálala con pip install openai y reinicia el kernel."
            )
        return False

    try:
        client = openai.OpenAI()
    except Exception as e:
        raise e

    # list models to check if the API key is working
    try:
        response = client.models.list()
    except openai.AuthenticationError as e:
        if verbose:
            print(
                f'Añade tu api key con:\n\nimport os\nos.environ["OPENAI_API_KEY"] = "TU_API_KEY"\n\nError: {e}'
            )
        return False

    if response.data:
        if verbose:
            print("Todo está ok! :)")
        return True
    else:
        if verbose:
            print("API key is not working: ", response)
        return False

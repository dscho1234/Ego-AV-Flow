import pickle


def read_pickle(file_name):
    """
    Read data from a pickle file.

    :param file_name: Name of the file to read from.
    :return: The data unpickled from the file.
    """
    with open(file_name, "rb") as file:
        return pickle.load(file)


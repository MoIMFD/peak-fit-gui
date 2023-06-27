from view import CLIObject, MatplotlibGUI
from model import load_dataframe, get_delimiter, FindPeaksWrapper


filepath = "./dummy_data.csv"
delimiter = get_delimiter(filepath)
df = load_dataframe(filepath, separator=delimiter)


fpw = FindPeaksWrapper()
gui = MatplotlibGUI(df, fpw, 2, 5)

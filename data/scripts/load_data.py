import pandas as pd
import pandas as np

from settings import GLASS_ORIGINAL, DIABETS_ORIGINAL, WINE_ORIGINAL


def load_glass():
    glass = pd.read_csv(GLASS_ORIGINAL)
    return {'data': glass[glass.columns.values[1:-1]].as_matrix(),
            'target': glass[glass.columns.values[-1]].as_matrix()}


def load_diabets():
    diabets = pd.read_csv(DIABETS_ORIGINAL)
    return {'data': diabets[diabets.columns.values[:-1]].as_matrix(),
            'target': diabets[diabets.columns.values[-1]].as_matrix()}


def load_wine():
    wine = pd.read_csv(WINE_ORIGINAL)
    return {'data': wine[wine.columns.values[1:]].as_matrix(),
            'target': wine[wine.columns.values[1]].as_matrix()}


if __name__ == "__main__":
    glass = load_glass()
    data = glass['data']


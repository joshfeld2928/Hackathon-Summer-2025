def get_test_data(df):
    return [pair.split('+') for pair in df.iloc[:, 0]]
import pandas as pd

input_path = "data/result/phrase_result.csv"

input_data = pd.read_csv(input_path)
#
# print(input_data[:6])

input_data = input_data[:6]

def lateral_explode(dataframe, fieldname):
    temp_fieldname = fieldname + '_made_tuple_'
    dataframe[temp_fieldname] = dataframe[fieldname].apply(tuple)
    list_of_dataframes = []
    for values in dataframe[temp_fieldname].unique().tolist():
        list_of_dataframes.append(pd.DataFrame({
            temp_fieldname: [values] * len(values),
            fieldname: list(values),
        }))
    dataframe = dataframe[list(set(dataframe.columns) - set([fieldname]))]\
        .merge(pd.concat(list_of_dataframes), how='left', on=temp_fieldname)
    del dataframe[temp_fieldname]

    return dataframe

expected = lateral_explode(input_data,'phrase')


# expected =(input_data.phrase.apply(pd.Series)
#               .stack()
#               .reset_index(drop=True)
#               .to_frame('phrase'))

# def unnest(df, col, reset_index=False):
#     col_flat = pd.DataFrame([[i, x]
#                        for i, y in df[col].apply(list).iteritems()
#                            for x in y], columns=['I', col])
#     col_flat = col_flat.set_index('I')
#     df = df.drop(col, 1)
#     df = df.merge(col_flat, left_index=True, right_index=True)
#     if reset_index:
#         df = df.reset_index(drop=True)
#     return df
#
# input = pd.DataFrame({'A': [1, 2], 'B': [['a', 'b'], 'c']})
#
# print(input)
print(input_data)

# expected = unnest(input,'B')

# expected = unnest(input_data, 'phrase')
#
print("expected****",expected)



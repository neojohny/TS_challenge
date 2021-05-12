from tsfresh import extract_features


def fresh_feature(df, id='merchant', sort_col = 'time'):
    
    extracted_features = extract_features(df, column_id=id, column_sort=sort_col)
    return extract_features

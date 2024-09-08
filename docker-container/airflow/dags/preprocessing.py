import pandas as pd
import numpy as np
from sklearn import preprocessing

def clean_data(filename):
    df = pd.read_csv(filename)

    df['date_time'] = df['date']+' '+df['time']
    df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True, infer_datetime_format=True)
    df['Week_Number'] = df['date_time'].dt.isocalendar().week
    # divide time into periods according to uk rush hours
    df.loc[(df['time'] >= '06:00') & (df['time'] <= '10:00'), 'time_period'] = 'Morning'
    df.loc[(df['time'] >= '10:01') & (df['time'] <= '14:00'), 'time_period'] = 'Noon'
    df.loc[(df['time'] >= '14:01') & (df['time'] <= '18:00'), 'time_period'] = 'Rush'
    df.loc[(df['time'] >= '18:01') & (df['time'] <= '22:00'), 'time_period'] = 'Night'
    df.loc[(df['time'] >= '22:01') & (df['time'] <= '23:59'), 'time_period'] = 'Midnight'
    df.loc[(df['time'] >= '00:00') & (df['time'] <= '02:00'), 'time_period'] = 'Midnight'
    df.loc[(df['time'] >= '02:01') & (df['time'] <= '05:59'), 'time_period'] = 'Early Morning'
    # data cleaning
    def find_null(df):
        df_null_count = df.isna().sum()
        df_null_column = df_null_count[df_null_count > 0].reset_index()
        df_null_column = df_null_column.rename(columns={0:'Percentage'})
        df_null_column['Percentage'] = round((df_null_column['Percentage'] / len(df)) * 100, 2)
        return df_null_column

    def find_missing(df):
        missing = []
        for (index, column) in enumerate(df):
            col_values = df[column].value_counts(normalize=False).reset_index().rename(columns={'index':column,column:'count'})
            if(len(col_values) == len(df)):
                continue
            for i in range(len(col_values)):
                value = col_values[column][i]
                if(type(value) == str):
                    if(('miss' in value.lower()) | ('-1' == value) | ('zero' in value.lower())):
                        missing.append((column , value, col_values['count'][i]))
                else:
                    if(value == -1):
                        missing.append((column , value, col_values['count'][i]))
        return missing

    # duplicates
    df.drop_duplicates(subset=['longitude','latitude','date_time'], keep='last', inplace=True)
    df = df.drop(columns=['accident_reference', 'date', 'time', 
                        'local_authority_ons_district', 'local_authority_highway', 'lsoa_of_accident_location'])
    # missing data
    def handle_missing(df, missing, df_null_column):
        for i in range(len(df_null_column)):
            if(df_null_column['Percentage'][i] < 2):
                df = df[df[df_null_column['index'][i]].notna()]
            else:
                df[df_null_column['index'][i]] = df[df_null_column['index'][i]].fillna(value=-1)
        for i in range(len(missing)):
            if (missing[i][1] == 'Data missing or out of range'):
                if(missing[i][0] == 'junction_control'):
                    df[missing[i][0]] = df[missing[i][0]].replace(missing[i][1], 'Not a junction')
                elif(missing[i][0] == 'trunk_road_flag'):
                    df[missing[i][0]] = df[missing[i][0]].replace(missing[i][1], 'Non-trunk')
                elif(((missing[i][2]/len(df)) * 100) < 2):
                    df = df[df[missing[i][0]] != missing[i][1]]
            elif('zero' in missing[i][1]):
                df[missing[i][0]] = df[missing[i][0]].replace(missing[i][1], 0)
        return df
    df_clean = handle_missing(df,find_missing(df),find_null(df))
    # outliers
    df_clean = df_clean[(df_clean['number_of_vehicles'] <= 5) & (df_clean['number_of_casualties'] <= 5)]

    df_clean['month'] = df_clean['date_time'].dt.month
    df_clean['day'] = df_clean['date_time'].dt.day
    df_clean['hour'] = df_clean['date_time'].dt.hour
    df_clean['minute'] = df_clean['date_time'].dt.minute
    df_clean.drop(columns = ['date_time'], inplace=True)

    df_clean['special_conditions_at_site'].replace(['None'],[0], inplace=True)
    df_clean.loc[df_clean['special_conditions_at_site'] != 0,'special_conditions_at_site'] = 1

    df_clean['carriageway_hazards'].replace(['None'],[0], inplace=True)
    df_clean.loc[df_clean['carriageway_hazards'] != 0,'carriageway_hazards'] = 1

    df_clean['pedestrian_crossing_human_control'].replace(['None within 50 metres '],[0], inplace=True)
    df_clean.loc[df_clean['pedestrian_crossing_human_control'] != 0,'pedestrian_crossing_human_control'] = 1

    df_clean['pedestrian_crossing_physical_facilities'].replace(['No physical crossing facilities within 50 metres'],[0], inplace=True)
    df_clean.loc[df_clean['pedestrian_crossing_physical_facilities'] != 0,'pedestrian_crossing_physical_facilities'] = 1

    one_hot_encoded = [('first_road_class',4), ('road_type',0), ('junction_detail',0), ('junction_control',3), ('second_road_class',4),
            ('light_conditions',3) ,('road_surface_conditions',4),('urban_or_rural_area',0)]


    label_encoded = ['police_force', 'accident_severity', 'day_of_week', 'local_authority_district', 
                    'weather_conditions' ,'did_police_officer_attend_scene_of_accident','trunk_road_flag', 'time_period']
    lookup = []
    def calculate_top_categories(df, variable, how_many):
        return [
            x for x in df[variable].value_counts().sort_values(
                ascending=False).head(how_many).index
        ]

    def label_encode(df):
        for i in range(len(label_encoded)):
            col = label_encoded[i]        
            label = preprocessing.LabelEncoder()
            df[col] = label.fit_transform(df[col])
            lookup.append((label_encoded[i], list(label.classes_), list(range(len(list(label.classes_))))))
    def one_hot_encode(df):
        for i in range(len(one_hot_encoded)):
            topx = calculate_top_categories(df, one_hot_encoded[i][0], one_hot_encoded[i][1])
            for label in topx:
                df[one_hot_encoded[i][0] + '_' + label] = np.where(
                    df[one_hot_encoded[i][0]] == label, 1, 0) 
            df.drop(columns=one_hot_encoded[i][0], inplace=True)
            
    label_encode(df_clean)
    one_hot_encode(df_clean)

    for i,col in enumerate(df_clean):
        if(col == 'accident_index'): continue
        df_clean[col] = pd.to_numeric(df_clean[col])
        
    df_clean = df_clean.reset_index(drop=True)

    st_scaler = preprocessing.StandardScaler()
    minmax_scaler = preprocessing.MinMaxScaler()

    df_clean[['location_easting_osgr','location_northing_osgr']] = minmax_scaler.fit_transform(df_clean[['location_easting_osgr','location_northing_osgr']])
    df_clean[['longitude','latitude']] = st_scaler.fit_transform(df_clean[['longitude','latitude']])
    df_clean[['Week_Number']] = minmax_scaler.fit_transform(df_clean[['Week_Number']])

    lookup_df = pd.DataFrame(columns=['feature_name','feature_original','feature_label'])
    for i in range(len(lookup)):
        for j in range(len(lookup[i][1])):
            lookup_df.loc[len(lookup_df)] = {'feature_name': lookup[i][0], 'feature_original': lookup[i][1][j], 'feature_label': lookup[i][2][j]}
    lookup_df.to_csv('/opt/airflow/data/lookup_test.csv',index=False)
    df_clean.to_csv('/opt/airflow/data/2010_UK_cleaned.csv', index=False)


def extract_feature(df_filename, lookup_filename, drivers_filename):
    accidents_df = pd.read_csv(df_filename)
    vehicles_df = pd.read_csv(drivers_filename,encoding='unicode_escape')
    lookup = pd.read_csv(lookup_filename)
    vehicles_df = vehicles_df[['Accident_Index', 'Age_Band_of_Driver','Year']]
    vehicles_df.drop_duplicates(subset=['Accident_Index'], inplace=True)
    df = pd.merge(left=accidents_df, right=vehicles_df, left_on=['accident_index','accident_year'], right_on=['Accident_Index','Year'], how='inner')
    df.drop(columns=['Accident_Index'], inplace=True)
    df = df[df['Age_Band_of_Driver'] != 'Data missing or out of range']
    df = df.reset_index(drop=True)
    label_encoder = preprocessing.LabelEncoder()
    df['Age_Band_of_Driver'] = label_encoder.fit_transform(df['Age_Band_of_Driver'])
    classes = list(label_encoder.classes_)
    for i in range(len(classes)):
        lookup.loc[len(lookup.index)] = ['Age_Band_of_Driver', classes[i], i]
    st_scaler = preprocessing.StandardScaler()
    df['Age_Band_of_Driver'] = st_scaler.fit_transform(np.array(df['Age_Band_of_Driver']).reshape(-1,1))
    df.to_csv('/opt/airflow/data/2010_UK_cleaned_featured.csv', index=False)
    lookup.to_csv('/opt/airflow/data/lookup_table.csv',index=False)
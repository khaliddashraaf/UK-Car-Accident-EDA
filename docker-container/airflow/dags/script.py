from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
from sqlalchemy import create_engine

from preprocessing import clean_data, extract_feature
from dashboard import create_dashboard

def load_to_postgres(df_filename, lookup_filename): 
    df = pd.read_csv(df_filename)
    lookup_df = pd.read_csv(lookup_filename)
    engine = create_engine('postgresql://root:root@UK_accidents_postgres:5432/UK_accidents')
    if(engine.connect()):
        print('connected succesfully')
    else:
        print('failed to connect')
    df.to_sql(name = 'UK_accidents_2010',con = engine,if_exists='replace')
    lookup_df.to_sql(name = 'lookup_table',con = engine,if_exists='replace')

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    'start_date': days_ago(2),
    "retries": 1,
}

dag = DAG(
    'UK_accidents_pipeline',
    default_args=default_args,
    description='UK accidents pipeline',
)
with DAG(
    dag_id = 'UK_accidents_pipeline',
    schedule_interval = '@once',
    default_args = default_args,
    tags = ['accidents-pipeline'],
)as dag:
    clean_task= PythonOperator(
        task_id = 'extract_clean_load',
        python_callable = clean_data,
        op_kwargs={
            "filename": '/opt/airflow/data/2010_Accidents_UK.csv'
        },
    )
    extract_task= PythonOperator(
        task_id = 'add_feature',
        python_callable = extract_feature,
        op_kwargs={
            "df_filename": '/opt/airflow/data/2010_UK_cleaned.csv',
            "lookup_filename": '/opt/airflow/data/lookup_test.csv',
            "drivers_filename": '/opt/airflow/data/Vehicle_Information.csv',
        },
    )
    load_to_postgres_task=PythonOperator(
        task_id = 'load_to_postgres',
        python_callable = load_to_postgres,
        op_kwargs={
            "df_filename": "/opt/airflow/data/2010_UK_cleaned_featured.csv",
            "lookup_filename": "/opt/airflow/data/lookup_table.csv",
        },
    )
    create_dashboard_task=PythonOperator(
        task_id = 'create_dashboard_task',
        python_callable = create_dashboard,
        op_kwargs={
            "filename": "/opt/airflow/data/2010_Accidents_UK.csv",
            "df_filename": "/opt/airflow/data/2010_UK_cleaned_featured.csv",
        },
    )


    clean_task >> extract_task >> load_to_postgres_task >> create_dashboard_task

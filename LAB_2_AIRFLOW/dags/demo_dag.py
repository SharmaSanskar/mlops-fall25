# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import logging

# Define default arguments for the demo DAG
default_args = {
    'owner': 'Sanskar Sharma',
    'start_date': datetime(2025, 10, 6),
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

# Simple Python functions for demonstration
def print_hello():
    """Simple function to print hello message"""
    print("Hello from Airflow Demo DAG!")
    logging.info("Demo task: Hello message printed successfully")
    return "Hello task completed"

def print_date():
    """Function to print current date and time"""
    current_time = datetime.now()
    print(f"Current date and time: {current_time}")
    logging.info(f"Demo task: Current time is {current_time}")
    return current_time.strftime("%Y-%m-%d %H:%M:%S")

def calculate_sum():
    """Simple calculation function"""
    numbers = [1, 2, 3, 4, 5]
    result = sum(numbers)
    print(f"Sum of {numbers} = {result}")
    logging.info(f"Demo task: Calculated sum = {result}")
    return result

def print_goodbye():
    """Function to print goodbye message"""
    print("Goodbye from Airflow Demo DAG!")
    logging.info("Demo task: Goodbye message printed successfully")
    return "Goodbye task completed"

# Create the demo DAG
with DAG(
    'demo_dag',
    default_args=default_args,
    description='A simple demonstration DAG for learning Airflow basics',
    schedule_interval=timedelta(hours=1),  # Run every hour
    catchup=False,
) as dag:

    # Task 1: Print hello message
    hello_task = PythonOperator(
        task_id='print_hello_task',
        python_callable=print_hello,
    )

    # Task 2: Print current date and time
    date_task = PythonOperator(
        task_id='print_date_task',
        python_callable=print_date,
    )

    # Task 3: Simple bash command
    bash_task = BashOperator(
        task_id='bash_command_task',
        bash_command='echo "This is a bash command from demo DAG" && date',
    )

    # Task 4: Calculate sum
    calculation_task = PythonOperator(
        task_id='calculate_sum_task',
        python_callable=calculate_sum,
    )

    # Task 5: Print goodbye message
    goodbye_task = PythonOperator(
        task_id='print_goodbye_task',
        python_callable=print_goodbye,
    )

    # Set task dependencies - create a simple linear flow
    hello_task >> date_task >> bash_task >> calculation_task >> goodbye_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()

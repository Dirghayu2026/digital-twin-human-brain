import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Database file stored in the project folder
DATABASE_PATH = Path(__file__).parent / "brain_database.db"


def get_connection():
    """Create a connection to the SQLite database."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    """Create the database tables if they do not exist."""

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS simulations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                neuron_count INTEGER NOT NULL,
                connection_density REAL NOT NULL,
                input_strength REAL NOT NULL,
                noise_level REAL NOT NULL,
                simulation_steps INTEGER NOT NULL,
                random_seed INTEGER NOT NULL,
                average_activity REAL NOT NULL,
                active_neurons INTEGER NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS neural_activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                simulation_id INTEGER NOT NULL,
                time_step INTEGER NOT NULL,
                neuron_id INTEGER NOT NULL,
                activity REAL NOT NULL,
                FOREIGN KEY (simulation_id)
                    REFERENCES simulations(id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS ai_training_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                dataset_size INTEGER NOT NULL,
                accuracy REAL NOT NULL
            )
        """)


def save_simulation(
    neuron_count,
    connection_density,
    input_strength,
    noise_level,
    simulation_steps,
    random_seed,
    activity
):
    """Save simulation settings and activity; return its ID."""

    activity = activity.tolist()

    average_activity = sum(activity[-1]) / len(activity[-1])

    active_neurons = sum(
        value > 0.5 for value in activity[-1]
    )

    with get_connection() as connection:

        cursor = connection.execute("""
            INSERT INTO simulations (
                created_at,
                neuron_count,
                connection_density,
                input_strength,
                noise_level,
                simulation_steps,
                random_seed,
                average_activity,
                active_neurons
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            neuron_count,
            connection_density,
            input_strength,
            noise_level,
            simulation_steps,
            random_seed,
            average_activity,
            active_neurons
        ))

        simulation_id = cursor.lastrowid

        records = []

        for time_index, row in enumerate(activity, start=1):
            for neuron_index, value in enumerate(row, start=1):
                records.append((
                    simulation_id,
                    time_index,
                    neuron_index,
                    float(value)
                ))

        connection.executemany("""
            INSERT INTO neural_activity (
                simulation_id,
                time_step,
                neuron_id,
                activity
            )
            VALUES (?, ?, ?, ?)
        """, records)

    return simulation_id


def get_simulations():
    """Return saved experiments, newest first."""

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM simulations
            ORDER BY id DESC
        """).fetchall()

    return [dict(row) for row in rows]


def get_simulation_activity(simulation_id):
    """Return activity records for a saved experiment."""

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT time_step, neuron_id, activity
            FROM neural_activity
            WHERE simulation_id = ?
            ORDER BY time_step, neuron_id
        """, (simulation_id,)).fetchall()

    return [dict(row) for row in rows]


def save_ai_result(dataset_size, accuracy):
    """Save AI model training results."""

    with get_connection() as connection:
        connection.execute("""
            INSERT INTO ai_training_results (
                created_at,
                dataset_size,
                accuracy
            )
            VALUES (?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            dataset_size,
            accuracy
        ))


def get_ai_results():
    """Return saved AI training results, newest first."""

    with get_connection() as connection:
        rows = connection.execute("""
            SELECT *
            FROM ai_training_results
            ORDER BY id DESC
        """).fetchall()

    return [dict(row) for row in rows]


# Create the database and tables automatically.
initialize_database()
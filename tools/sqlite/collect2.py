#!/usr/bin/env python3
import os
import csv
import sqlite3
import argparse
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from queue import Queue
from threading import Thread
import logging
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the database schema.
# Do not define a primary key here, let the merge tools deal with dupes.
CREATE_TABLE_QUERY = '''
CREATE TABLE IF NOT EXISTS position(epd text, depth integer, score integer)
'''

INSERT_QUERY = '''
INSERT OR REPLACE INTO position (epd, depth, score) VALUES (?, ?, ?)
'''

def parse_arguments():
    parser = argparse.ArgumentParser(description='Watch directory for new files and process them.')
    parser.add_argument('--dir', required=True, help='Directory to watch')
    parser.add_argument('--db', required=True, help='SQLite database file')
    return parser.parse_args()

def create_table(connection):
    with connection:
        connection.execute(CREATE_TABLE_QUERY)

def process_file(file_path, db_queue):
    logging.info(f"Processing file: {file_path}")
    data = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            epd, uci_move, depth, score = row
            depth = int(depth)
            score = int(score)
            data.append((epd, depth, score))
    db_queue.put(data)
    os.remove(file_path)
    logging.info(f"Finished processing and deleted file: {file_path}")

class FileHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue

    def on_created(self, event):
        if not event.is_directory:
            logging.info(f"Detected new file: {event.src_path}")
            self.queue.put(event.src_path)

def is_file_open(file_path):
    result = subprocess.run(['lsof', file_path], capture_output=True, text=True)
    return bool(result.stdout)

def process_queue(file_queue, db_queue):
    while True:
        file_path = file_queue.get()
        if file_path is None:
            file_queue.task_done()
            break
        while is_file_open(file_path):
            logging.warning(f"File {file_path} is open by another process, retrying...")
            time.sleep(1)
        try:
            process_file(file_path, db_queue)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
        file_queue.task_done()

def insert_into_db(db_queue, db_path):
    connection = sqlite3.connect(db_path)
    create_table(connection)
    while True:
        data = db_queue.get()
        if data is None:
            db_queue.task_done()
            break
        with connection:
            try:
                for epd, depth, score in data:
                    connection.execute(INSERT_QUERY, (epd, depth, score))
            except Exception as e:
                logging.error(f"Error inserting into database: {e}")
        db_queue.task_done()
    connection.close()

def main():
    args = parse_arguments()

    # Ensure the directory exists
    if not os.path.exists(args.dir):
        os.makedirs(args.dir)
        logging.info(f"Created directory: {args.dir}")

    logging.info("Starting the directory watcher")

    # Set up the queues
    file_queue = Queue()
    db_queue = Queue()

    # Set up the worker threads
    file_worker = Thread(target=process_queue, args=(file_queue, db_queue))
    file_worker.daemon = True
    file_worker.start()

    db_worker = Thread(target=insert_into_db, args=(db_queue, args.db))
    db_worker.daemon = True
    db_worker.start()

    # Set up the watchdog observer
    event_handler = FileHandler(file_queue)
    observer = Observer()
    observer.schedule(event_handler, args.dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        logging.info("Stopping the directory watcher")
        file_queue.put(None)
        db_queue.put(None)
    observer.join()

    # Wait for all threads to finish
    file_queue.join()
    db_queue.join()

if __name__ == '__main__':
    main()

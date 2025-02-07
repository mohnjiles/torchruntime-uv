# Converts the txt file at https://github.com/pciutils/pciids/blob/master/pci.ids to an sqlite3 db

import sqlite3
import re
import sys

from torchruntime.device_db import get_gpu_type, is_gpu_vendor

# Compile all regular expressions
SKIP_LINE_REGEX = re.compile(r"^\t{2,}")
VENDOR_REGEX = re.compile(r"^([0-9a-fA-F]{4})\s+(.*)")
DEVICE_REGEX = re.compile(r"^\t([0-9a-fA-F]{4})\s+(.*)")


def process_pci_file(input_file: str, output_db: str):
    # Connect to the SQLite database (create if it doesn't exist)
    conn = sqlite3.connect(output_db)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pci_ids (
            vendor_id TEXT,
            vendor_name TEXT,
            device_id TEXT,
            device_name TEXT,
            is_discrete INTEGER
        )
    """
    )

    vendor_id, vendor_name = None, None  # Initialize the current vendor context

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            # Skip comments and deeply indented lines (2 or more indents)
            if line.startswith("#") or SKIP_LINE_REGEX.match(line):
                continue

            # Match vendor lines (no tab)
            match_vendor = VENDOR_REGEX.match(line)
            if match_vendor:
                vendor_id, vendor_name = match_vendor.groups()
                continue  # Move to next line for processing

            if not is_gpu_vendor(vendor_id):
                continue

            # Match device lines (single tab)
            match_device = DEVICE_REGEX.match(line)
            if match_device and vendor_id and vendor_name:
                device_id, device_name = match_device.groups()

                gpu_type = get_gpu_type(vendor_id, device_id, device_name)
                if gpu_type == "NONE":
                    continue

                is_discrete = 1 if gpu_type == "DISCRETE" else 0

                # Insert the vendor and device information into the database
                cursor.execute(
                    """
                    INSERT INTO pci_ids (vendor_id, vendor_name, device_id, device_name, is_discrete)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (vendor_id, vendor_name, device_id, device_name, is_discrete),
                )

    # Commit the changes and close the connection
    conn.commit()
    conn.close()


if __name__ == "__main__":
    # Check if input and output arguments are provided
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_db>")
        print("Get the text file at https://github.com/pciutils/pciids/blob/master/pci.ids")
        sys.exit(1)

    input_file = sys.argv[1]
    output_db = sys.argv[2]

    # Process the input file and output to the SQLite database
    process_pci_file(input_file, output_db)
